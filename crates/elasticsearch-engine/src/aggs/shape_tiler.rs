//! ES 9.5.5's `GeoGridTiler`s (`x-pack/spatial`): the cells of a
//! `geo_shape` doc value for `geotile_grid`, `geohash_grid` and
//! `geohex_grid` (`GeoShapeCellIdSource` / `GeoShapeCellValues`), each with
//! its bounded and unbounded form: the doc's bounding box picks one cell,
//! a brute-force scan of a few, or a recursive descent from the coarsest
//! cells, every candidate related to the doc's triangle tree
//! (`GeoShapeValue.relate`: `Tile2DVisitor` for tiles and hashes, ES's
//! `GeoHexVisitor` for H3 cells).
//!
//! Cells come out as ES's longs (`GeoTileUtils.longEncodeTiles`,
//! `Geohash.longEncode`, the H3 index) and keys as ES prints them.

use rustic_core::fieldtypes::geo2d::{line_crosses_line, line_crosses_line_with_boundary, orient, point_in_triangle};
use rustic_core::fieldtypes::shape_dv::{ab_from_triangle, bc_from_triangle, ca_from_triangle, GeoRelation, ShapeValue, TreeVisitor};
use rustic_core::geo::{decode_lat, decode_lon, encode_lat, encode_lon};

/// The `bounds` of a grid aggregation (`GeoBoundingBox`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub top: f64,
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
}

// ---------------------------------------------------------------------------
// GeoTileUtils

const MAX_ZOOM: u32 = 29;
const LATITUDE_MASK: f64 = 85.0511287798066;
const MAX_TILES_FULL_PRECISION: u64 = 1 << 20;
const LUCENE_LAT_RES: f64 = 180.0 / 4294967296.0;

fn encoded_lat_mask() -> i32 {
    encode_lat(LATITUDE_MASK)
}
fn encoded_neg_lat_mask() -> i32 {
    encode_lat(-LATITUDE_MASK)
}
fn normalized_lat_mask() -> f64 {
    decode_lat(encoded_lat_mask())
}
fn normalized_neg_lat_mask() -> f64 {
    decode_lat(encoded_neg_lat_mask())
}

/// `GeoTileUtils.getXTile`.
pub(crate) fn get_x_tile(lon: f64, tiles: u64) -> u32 {
    let x = (lon + 180.0) / 360.0 * tiles as f64;
    (x.floor() as i64).clamp(0, tiles as i64 - 1) as u32
}

/// `GeoTileUtils.getYTile` (Lucene `SloppyMath.cos`, ES `FastMath.log`).
pub(crate) fn get_y_tile(lat: f64, tiles: u64) -> u32 {
    let lat_sin = crate::sloppy_math::cos(std::f64::consts::FRAC_PI_2 - lat.to_radians());
    let y = (0.5 - (crate::sloppy_math::es_log((1.0 + lat_sin) / (1.0 - lat_sin)) / (std::f64::consts::PI * 4.0))) * tiles as f64;
    let f = y.floor();
    // Java `(int) Math.floor(..)` then the clamp (NaN reads 0).
    let i = if f.is_nan() { 0 } else { f.clamp(i32::MIN as f64, i32::MAX as f64) as i64 };
    i.clamp(0, tiles as i64 - 1) as u32
}

/// `GeoTileUtils.longEncodeTiles`.
pub(crate) fn long_encode_tiles(precision: u32, x: u32, y: u32) -> u64 {
    ((precision as u64) << (MAX_ZOOM * 2)) | ((x as u64) << MAX_ZOOM) | y as u64
}

/// The key of a tile long.
pub(crate) fn tile_key(h: u64) -> String {
    let mask = (1u64 << MAX_ZOOM) - 1;
    format!("{}/{}/{}", h >> (MAX_ZOOM * 2), (h >> MAX_ZOOM) & mask, h & mask)
}

/// `GeoTileUtils.tileToLon`.
pub(crate) fn tile_to_lon(x: u32, tiles: u64) -> f64 {
    (x as f64 / tiles as f64 * 360.0) - 180.0
}

fn tile_to_lat_f(y: f64, tiles: u64) -> f64 {
    let n = std::f64::consts::PI - (std::f64::consts::PI * 2.0 * y) / tiles as f64;
    crate::fieldtypes::geo::fast_math::atan(crate::fieldtypes::geo::fast_math::sinh(n)).to_degrees()
}

/// `GeoTileUtils.tileToLat` (refined to the Lucene grid above 2^20 tiles).
pub(crate) fn tile_to_lat(y: u32, tiles: u64) -> f64 {
    let lat = tile_to_lat_f(y as f64, tiles);
    if tiles < MAX_TILES_FULL_PRECISION || y == 0 || y as u64 == tiles {
        return lat;
    }
    let q = decode_lat(encode_lat(lat));
    let computed = get_y_tile(q, tiles);
    find_boundary_point(q, computed, tiles, if computed == y { LUCENE_LAT_RES } else { -LUCENE_LAT_RES })
}

fn find_boundary_point(q: f64, y: u32, tiles: u64, step: f64) -> f64 {
    let mut q = q;
    loop {
        let next = q + step;
        let ny = get_y_tile(next, tiles);
        if ny != y {
            return if step > 0.0 { q } else { next };
        }
        q = next;
    }
}

/// `GeoTileBoundedPredicate`.
#[derive(Clone, Debug)]
struct TilePred {
    crosses: bool,
    max_tiles: u64,
    precision: u32,
    left_x: i64,
    right_x: i64,
    min_y: i64,
    max_y: i64,
}

impl TilePred {
    fn new(precision: u32, b: &Bounds) -> TilePred {
        let crosses = b.right < b.left;
        if b.bottom > normalized_lat_mask() || b.top < normalized_neg_lat_mask() {
            return TilePred { crosses, max_tiles: 0, precision, left_x: -1, right_x: -1, min_y: -1, max_y: -1 };
        }
        let tiles = 1u64 << precision;
        let min_x = get_x_tile(b.left, tiles);
        let min_y = get_y_tile(b.top, tiles);
        // `toBoundingBox(minX, minY)`: max x is the next tile's left, min y
        // the next row's top.
        let left_x = if encode_lon(tile_to_lon(min_x + 1, tiles)) == encode_lon(b.left) { min_x as i64 + 1 } else { min_x as i64 };
        let min_y2 = if encode_lat(tile_to_lat(min_y + 1, tiles)) == encode_lat(b.top) { min_y as i64 + 1 } else { min_y as i64 };
        let max_x = get_x_tile(b.right, tiles);
        let max_y = get_y_tile(b.bottom, tiles);
        let right_x = if encode_lon(tile_to_lon(max_x, tiles)) == encode_lon(b.right) { max_x as i64 } else { max_x as i64 + 1 };
        let max_y2 = if encode_lat(tile_to_lat(max_y, tiles)) == encode_lat(b.bottom) { max_y as i64 } else { max_y as i64 + 1 };
        let max_tiles = if crosses { ((tiles as i64 + right_x - left_x) * (max_y2 - min_y2)) as u64 } else { ((right_x - left_x) * (max_y2 - min_y2)).max(0) as u64 };
        TilePred { crosses, max_tiles, precision, left_x, right_x, min_y: min_y2, max_y: max_y2 }
    }

    fn valid_tile(&self, x: u32, y: u32, precision: u32) -> bool {
        let splits = 1i64 << (self.precision - precision);
        let y_min = y as i64 * splits;
        if self.max_y > y_min && self.min_y < y_min + splits {
            let x_min = x as i64 * splits;
            if self.crosses {
                return self.right_x > x_min || self.left_x < x_min + splits;
            }
            return self.right_x > x_min && self.left_x < x_min + splits;
        }
        false
    }
}

/// `GeoTileGridTiler` (bounded when `pred` is set).
pub(crate) struct TileTiler {
    precision: u32,
    tiles: u64,
    pred: Option<TilePred>,
}

impl TileTiler {
    pub(crate) fn new(precision: u32, bounds: Option<&Bounds>) -> TileTiler {
        TileTiler { precision, tiles: 1u64 << precision, pred: bounds.map(|b| TilePred::new(precision, b)) }
    }

    fn valid_tile(&self, x: u32, y: u32, z: u32) -> bool {
        self.pred.as_ref().is_none_or(|p| p.valid_tile(x, y, z))
    }

    fn max_cells(&self) -> u64 {
        self.pred.as_ref().map_or(self.tiles * self.tiles, |p| p.max_tiles)
    }

    /// `setValues`: the doc's tiles (unsorted, possibly repeated).
    pub(crate) fn set_values(&self, v: &ShapeValue<'_>, out: &mut Vec<u64>) {
        let Some(b) = v.bounding_box() else { return };
        if b.bottom > normalized_lat_mask() || b.top < normalized_neg_lat_mask() {
            return;
        }
        if self.precision == 0 {
            self.brute(v, 0, 0, 0, 0, out);
            return;
        }
        let tiles = self.tiles;
        let min_xt = get_x_tile(b.min_x(), tiles);
        let min_yt = get_y_tile(b.max_y(), tiles);
        let max_xt = get_x_tile(b.max_x(), tiles);
        let max_yt = get_y_tile(b.min_y(), tiles);
        let count = (max_xt as i64 - min_xt as i64 + 1) * (max_yt as i64 - min_yt as i64 + 1);
        if count == 1 {
            if self.valid_tile(min_xt, min_yt, self.precision) {
                out.push(long_encode_tiles(self.precision, min_xt, min_yt));
            }
        } else if count <= 8 * self.precision as i64 {
            self.brute(v, min_xt, min_yt, max_xt, max_yt, out);
        } else {
            self.raster(0, 0, 0, v, out);
        }
    }

    fn relate_tile(&self, v: &ShapeValue<'_>, x: u32, y: u32, z: u32) -> GeoRelation {
        if !self.valid_tile(x, y, z) {
            return GeoRelation::QueryDisjoint;
        }
        let tiles = 1u64 << z;
        let min_x = encode_lon(tile_to_lon(x, tiles));
        let max_x = encode_lon(tile_to_lon(x + 1, tiles));
        let min_y = encode_lat(tile_to_lat(y + 1, tiles));
        let max_y = encode_lat(tile_to_lat(y, tiles));
        v.relate_tile(min_x, if max_x == i32::MAX { max_x } else { max_x - 1 }, if min_y == encoded_neg_lat_mask() { min_y } else { min_y + 1 }, max_y)
    }

    fn brute(&self, v: &ShapeValue<'_>, min_x: u32, min_y: u32, max_x: u32, max_y: u32, out: &mut Vec<u64>) {
        for i in min_x..=max_x {
            for j in min_y..=max_y {
                if self.relate_tile(v, i, j, self.precision) != GeoRelation::QueryDisjoint {
                    out.push(long_encode_tiles(self.precision, i, j));
                }
            }
        }
    }

    fn raster(&self, x: u32, y: u32, z: u32, v: &ShapeValue<'_>, out: &mut Vec<u64>) {
        let z = z + 1;
        for i in 0..2 {
            for j in 0..2 {
                let (nx, ny) = (2 * x + i, 2 * y + j);
                let r = self.relate_tile(v, nx, ny, z);
                if r == GeoRelation::QueryInside {
                    if z == self.precision {
                        out.push(long_encode_tiles(z, nx, ny));
                    } else {
                        let _ = self.max_cells();
                        self.fully_contained(nx, ny, z, out);
                    }
                } else if r != GeoRelation::QueryDisjoint {
                    if z == self.precision {
                        out.push(long_encode_tiles(z, nx, ny));
                    } else {
                        self.raster(nx, ny, z, v, out);
                    }
                }
            }
        }
    }

    /// `setValuesForFullyContainedTile`.
    fn fully_contained(&self, x: u32, y: u32, z: u32, out: &mut Vec<u64>) {
        let splits = 1i64 << (self.precision - z);
        let push = |out: &mut Vec<u64>, min_y: i64, max_y: i64, min_x: i64, max_x: i64| {
            for i in min_x..max_x {
                for j in min_y..max_y {
                    out.push(long_encode_tiles(self.precision, i as u32, j as u32));
                }
            }
        };
        match &self.pred {
            None => {
                let (min_x, min_y) = (x as i64 * splits, y as i64 * splits);
                push(out, min_y, min_y + splits, min_x, min_x + splits);
            }
            Some(p) => {
                let min_y = p.min_y.max(y as i64 * splits);
                let max_y = p.max_y.min(y as i64 * splits + splits);
                if p.crosses {
                    let west_min_x = p.left_x.max(x as i64 * splits);
                    let west_max_x = x as i64 * splits + splits;
                    push(out, min_y, max_y, west_min_x, west_max_x);
                    let east_max_x = west_min_x.min(p.right_x.min(x as i64 * splits + splits));
                    let east_min_x = x as i64 * splits;
                    push(out, min_y, max_y, east_min_x, east_max_x);
                } else {
                    let min_x = p.left_x.max(x as i64 * splits);
                    let max_x = p.right_x.min(x as i64 * splits + splits);
                    push(out, min_y, max_y, min_x, max_x);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Geohash (ES `org.elasticsearch.geometry.utils.Geohash`)

const BASE_32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";
const GH_PRECISION: u32 = 12;
const MORTON_OFFSET: u32 = 64 - GH_PRECISION * 5;
const GH_LAT_SCALE: f64 = (1u64 << 31) as f64 / 180.0;
const GH_LON_SCALE: f64 = (1u64 << 31) as f64 / 360.0;
const MAX_LAT_BITS: u64 = (1u64 << (GH_PRECISION * 5 / 2)) - 1;

fn gh_lat_height(p: u32) -> f64 {
    let (mut h, mut even) = (180.0f64, false);
    for _ in 0..p {
        h /= if even { 8.0 } else { 4.0 };
        even = !even;
    }
    h
}

fn gh_lon_width(p: u32) -> f64 {
    let (mut w, mut even) = (360.0f64, false);
    for _ in 0..p {
        w /= if even { 4.0 } else { 8.0 };
        even = !even;
    }
    w
}

fn interleave(even: u32, odd: u32) -> u64 {
    rustic_core::fieldtypes::tessellator::interleave(even as i32, odd as i32)
}

/// Lucene `BitUtil.deinterleave`.
fn deinterleave(b: u64) -> u64 {
    let mut b = b & 0x5555555555555555;
    b = (b ^ (b >> 1)) & 0x3333333333333333;
    b = (b ^ (b >> 2)) & 0x0F0F0F0F0F0F0F0F;
    b = (b ^ (b >> 4)) & 0x00FF00FF00FF00FF;
    b = (b ^ (b >> 8)) & 0x0000FFFF0000FFFF;
    b = (b ^ (b >> 16)) & 0x00000000FFFFFFFF;
    b
}

/// Lucene `BitUtil.flipFlop`.
fn flip_flop(b: u64) -> u64 {
    ((b & 0xAAAAAAAAAAAAAAAA) >> 1) | ((b & 0x5555555555555555) << 1)
}

fn encode_lat_lon(lat: f64, lon: f64) -> u64 {
    let lat_enc = (encode_lat(lat) as u32) ^ 0x8000_0000;
    let lon_enc = (encode_lon(lon) as u32) ^ 0x8000_0000;
    interleave(lat_enc, lon_enc) >> 2
}

/// `Geohash.stringEncode(lon, lat, level)`.
pub(crate) fn gh_string_encode(lon: f64, lat: f64, level: u32) -> String {
    let interleaved = encode_lat_lon(lat, lon) >> (((GH_PRECISION - level) * 5) + (MORTON_OFFSET - 2));
    gh_string_of_long((interleaved << 4) | level as u64)
}

fn gh_string_of_long(h: u64) -> String {
    let level = (h & 15) as usize;
    let mut v = h >> 4;
    let mut chars = vec![0u8; level];
    for i in (0..level).rev() {
        chars[i] = BASE_32[(v & 31) as usize];
        v >>= 5;
    }
    String::from_utf8(chars).unwrap_or_default()
}

fn b32(c: u8) -> u64 {
    BASE_32.iter().position(|&x| x == c).unwrap_or(0) as u64
}

/// `Geohash.longEncode(String)`.
pub(crate) fn gh_long_encode(hash: &str) -> u64 {
    let len = hash.len() as u32;
    let mut level = len as i64 - 1;
    let mut l = 0u64;
    for c in hash.bytes() {
        l |= b32(c) << (level as u64 * 5);
        level -= 1;
        if level < 0 {
            break;
        }
    }
    (l << 4) | len as u64
}

/// The key of a geohash long.
pub(crate) fn gh_key(h: u64) -> String {
    gh_string_of_long(h)
}

fn morton_encode(hash: &str) -> u64 {
    let mut level: i64 = 11;
    let mut l = 0u64;
    for c in hash.bytes() {
        l |= b32(c) << ((level as u64 * 5) + (MORTON_OFFSET as u64 - 2));
        level -= 1;
        if level < 0 {
            break;
        }
    }
    flip_flop(l)
}

fn gh_decode_lon(h: u64) -> f64 {
    (deinterleave(h) as f64 / GH_LON_SCALE) - 180.0
}

fn gh_decode_lat(h: u64) -> f64 {
    (deinterleave(h >> 1) as f64 / GH_LAT_SCALE) - 90.0
}

/// `Geohash.toBoundingBox`: `(min_lon, max_lon, max_lat, min_lat)`.
pub(crate) fn gh_bbox(hash: &str) -> (f64, f64, f64, f64) {
    let m = morton_encode(hash);
    let (bl_x, bl_y) = (gh_decode_lon(m), gh_decode_lat(m));
    let len = hash.len().min(12) as u32;
    let mut gh = gh_long_encode(&hash[..len as usize]) >> 4;
    let lon = deinterleave(gh >> 1);
    let lat = deinterleave(gh);
    let shift = (12 - len) * 5 + 2;
    if lat < MAX_LAT_BITS {
        gh = interleave((lat + 1) as u32, (lon + 1) as u32) << 4 | len as u64;
        let mh = flip_flop((gh >> 4) << shift);
        (bl_x, gh_decode_lon(mh), gh_decode_lat(mh), bl_y)
    } else {
        gh = interleave(lat as u32, (lon + 1) as u32) << 4 | len as u64;
        let mh = flip_flop((gh >> 4) << shift);
        (bl_x, gh_decode_lon(mh), 90.0, bl_y)
    }
}

fn encode_base32(x: i32, y: i32) -> u8 {
    BASE_32[(((x & 1) + ((y & 1) * 2) + ((x & 2) * 2) + ((y & 2) * 4) + ((x & 4) * 4)) % 32) as usize]
}

/// `Geohash.getNeighbor`.
fn gh_neighbor(hash: &str, level: usize, dx: i32, dy: i32) -> Option<String> {
    let cell = b32(hash.as_bytes()[level - 1]) as i32;
    let (x0, y0, x1, y1, x2) = (cell & 1, cell & 2, cell & 4, cell & 8, cell & 16);
    let x = x0 + (x1 / 2) + (x2 / 4);
    let y = (y0 / 2) + (y1 / 4);
    if level == 1 {
        if (dy < 0 && y == 0) || (dy > 0 && y == 3) {
            None
        } else {
            Some((encode_base32(x + dx, y + dy) as char).to_string())
        }
    } else {
        let nx = if level % 2 == 1 { x + dx } else { x + dy };
        let ny = if level % 2 == 1 { y + dy } else { y + dx };
        if (0..=7).contains(&nx) && (0..=3).contains(&ny) {
            let mut s = hash[..level - 1].to_string();
            s.push(encode_base32(nx, ny) as char);
            Some(s)
        } else {
            gh_neighbor(hash, level - 1, dx, dy).map(|mut n| {
                n.push(encode_base32(nx, ny) as char);
                n
            })
        }
    }
}

/// `GeoHashBoundedPredicate`.
struct HashPred {
    b: Bounds,
    crosses: bool,
}

impl HashPred {
    fn valid_hash(&self, hash: &str) -> bool {
        let (min_x, max_x, max_y, min_y) = gh_bbox(hash);
        if max_x < min_x {
            self.intersects(-180.0, max_x, min_y, max_y) || self.intersects(min_x, 180.0, min_y, max_y)
        } else {
            self.intersects(min_x, max_x, min_y, max_y)
        }
    }

    fn intersects(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        if self.b.top > min_y && self.b.bottom < max_y {
            if self.crosses {
                return self.b.left < max_x || self.b.right > min_x;
            }
            return self.b.left < max_x && self.b.right > min_x;
        }
        false
    }
}

/// `GeoHashGridTiler` (bounded when `pred` is set).
pub(crate) struct HashTiler {
    precision: u32,
    pred: Option<HashPred>,
}

impl HashTiler {
    pub(crate) fn new(precision: u32, bounds: Option<&Bounds>) -> HashTiler {
        HashTiler { precision, pred: bounds.map(|b| HashPred { b: *b, crosses: b.right < b.left }) }
    }

    fn valid_hash(&self, h: &str) -> bool {
        self.pred.as_ref().is_none_or(|p| p.valid_hash(h))
    }

    pub(crate) fn set_values(&self, v: &ShapeValue<'_>, out: &mut Vec<u64>) {
        if self.precision == 0 {
            return;
        }
        let Some(b) = v.bounding_box() else { return };
        let p = self.precision;
        if b.min_x() == b.max_x() && b.min_y() == b.max_y() {
            self.set_value(v, &gh_string_encode(b.min_x(), b.min_y(), p), out);
            return;
        }
        let dx = ((b.max_x() - b.min_x()) / gh_lon_width(p)).ceil() as i64;
        let dy = ((b.max_y() - b.min_y()) / gh_lat_height(p)).ceil() as i64;
        if dx.saturating_mul(dy) <= 32 * p as i64 {
            self.brute(v, &b, out);
        } else {
            self.raster("", v, out);
        }
    }

    fn set_value(&self, v: &ShapeValue<'_>, hash: &str, out: &mut Vec<u64>) {
        if self.relate_tile(v, hash) != GeoRelation::QueryDisjoint {
            out.push(gh_long_encode(hash));
        }
    }

    fn relate_tile(&self, v: &ShapeValue<'_>, hash: &str) -> GeoRelation {
        if !self.valid_hash(hash) {
            return GeoRelation::QueryDisjoint;
        }
        let (min_lon, max_lon, max_lat, min_lat) = gh_bbox(hash);
        let (min_x, min_y, max_x, max_y) = (encode_lon(min_lon), encode_lat(min_lat), encode_lon(max_lon), encode_lat(max_lat));
        v.relate_tile(min_x, if max_x == i32::MAX { max_x } else { max_x - 1 }, min_y, if max_y == i32::MAX { max_y } else { max_y - 1 })
    }

    fn brute(&self, v: &ShapeValue<'_>, b: &rustic_core::fieldtypes::shape_dv::BoundingBox, out: &mut Vec<u64>) {
        let p = self.precision;
        let stop = gh_string_encode(b.max_x(), b.max_y(), p);
        let move_down = |hash: Option<String>, x: f64, y: f64| -> Option<String> {
            match hash {
                None => Some(gh_string_encode(x, y, p)),
                Some(h) => gh_neighbor(&h, p as usize, 0, 1),
            }
        };
        let mut first_in_row: Option<String> = None;
        let mut last_in_row: Option<String> = None;
        // Bounded by the number of cells of the box (guards a malformed walk).
        let mut guard = 0u64;
        loop {
            last_in_row = move_down(last_in_row, b.max_x(), b.min_y());
            let Some(last) = last_in_row.clone() else { return };
            let mut current: Option<String> = None;
            loop {
                current = match current {
                    None => {
                        first_in_row = move_down(first_in_row, b.min_x(), b.min_y());
                        first_in_row.clone()
                    }
                    Some(c) => gh_neighbor(&c, p as usize, 1, 0),
                };
                let Some(c) = current.clone() else { return };
                self.set_value(v, &c, out);
                guard += 1;
                if c == last || guard > 1 << 20 {
                    break;
                }
            }
            if last == stop || guard > 1 << 20 {
                return;
            }
        }
    }

    fn raster(&self, hash: &str, v: &ShapeValue<'_>, out: &mut Vec<u64>) {
        for &c in BASE_32 {
            let mut s = hash.to_string();
            s.push(c as char);
            let r = self.relate_tile(v, &s);
            if r == GeoRelation::QueryInside {
                if s.len() as u32 == self.precision {
                    out.push(gh_long_encode(&s));
                } else {
                    self.fully_contained(&s, out);
                }
            } else if r != GeoRelation::QueryDisjoint {
                if s.len() as u32 == self.precision {
                    out.push(gh_long_encode(&s));
                } else {
                    self.raster(&s, v, out);
                }
            }
        }
    }

    fn fully_contained(&self, hash: &str, out: &mut Vec<u64>) {
        for &c in BASE_32 {
            let mut s = hash.to_string();
            s.push(c as char);
            if self.valid_hash(&s) {
                if s.len() as u32 == self.precision {
                    out.push(gh_long_encode(&s));
                } else {
                    self.fully_contained(&s, out);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// H3 (ES `H3` / `H3CartesianUtil` over [`crate::h3`], the port of ES's H3)

use crate::h3;

fn h3_res(h: u64) -> u32 {
    h3::resolution(h) as u32
}

fn is_pentagon(h: u64) -> bool {
    h3::is_pentagon(h)
}

/// `H3.geoToH3`.
pub(crate) fn geo_to_h3(lat: f64, lon: f64, res: u32) -> u64 {
    h3::geo_to_h3(lat, lon, res.min(15) as i32).unwrap_or(0)
}

fn hex_ring_size(h: u64) -> usize {
    h3::hex_ring_size(h)
}

/// `H3.hexRingPosToH3` (0 where ES has no neighbour).
fn hex_ring_pos(h: u64, pos: usize) -> u64 {
    h3::hex_ring_pos_to_h3(h, pos).unwrap_or(0)
}

/// `H3.childPosToH3`.
fn child_pos(h: u64, pos: usize) -> u64 {
    h3::child_pos_to_h3(h, pos).unwrap_or(0)
}

fn children_size(h: u64) -> usize {
    h3::children_size(h)
}

fn not_intersecting_children_size(h: u64) -> usize {
    h3::not_intersecting_children_size(h)
}

/// `H3.noChildIntersectingPosToH3`.
fn no_child_intersecting_pos(h: u64, pos: usize) -> u64 {
    h3::no_child_intersecting_pos_to_h3(h, pos).unwrap_or(0)
}

fn h3_parent(h: u64) -> u64 {
    h3::parent(h).unwrap_or(0)
}

/// Per resolution: the north and south polar cells and their bounds
/// (`H3CartesianUtil.NORTH` / `SOUTH`, computed once).
struct PolarTable {
    north: [u64; 16],
    south: [u64; 16],
    north_bound: [f64; 16],
    south_bound: [f64; 16],
}

fn polar_table() -> &'static PolarTable {
    static T: std::sync::OnceLock<PolarTable> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = PolarTable { north: [0; 16], south: [0; 16], north_bound: [0.0; 16], south_bound: [0.0; 16] };
        for r in 0..16u32 {
            t.north[r as usize] = geo_to_h3(90.0, 0.0, r);
            t.south[r as usize] = geo_to_h3(-90.0, 0.0, r);
        }
        for r in 0..16usize {
            t.north_bound[r] = h3_bbox_polar(t.north[r], t.north[r], t.south[r]).3;
            t.south_bound[r] = h3_bbox_polar(t.south[r], t.north[r], t.south[r]).2;
        }
        t
    })
}

fn polar(lat: f64, res: u32) -> u64 {
    let t = polar_table();
    let r = (res as usize).min(15);
    if lat > 0.0 {
        t.north[r]
    } else {
        t.south[r]
    }
}

/// `H3CartesianUtil.isPolar`.
fn is_polar(h: u64) -> bool {
    let r = h3_res(h);
    polar(90.0, r) == h || polar(-90.0, r) == h
}

fn boundary(h: u64) -> Vec<(f64, f64)> {
    h3::boundary(h).iter().map(|l| (l.lon_deg(), l.lat_deg())).collect()
}

/// `H3CartesianUtil.computePoints`: the cell's boundary on the Lucene grid,
/// closed; polar cells closed along the dateline through the pole.
pub(crate) fn compute_points(h: u64) -> (Vec<f64>, Vec<f64>) {
    let b = boundary(h);
    let n = b.len();
    let res = h3_res(h);
    let north = polar(90.0, res) == h;
    let south = polar(-90.0, res) == h;
    let total = if north || south { n + 5 } else { n + 1 };
    let mut xs = vec![0.0; total];
    let mut ys = vec![0.0; total];
    for (i, p) in b.iter().enumerate() {
        xs[i] = decode_lon(encode_lon(p.0));
        ys[i] = decode_lat(encode_lat(p.1));
    }
    if north || south {
        let pole = decode_lat(encode_lat(if north { 90.0 } else { -90.0 }));
        // `sort` by x (an `IntroSorter`; boundary longitudes are distinct).
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &c| xs[a].total_cmp(&xs[c]));
        let (sx, sy): (Vec<f64>, Vec<f64>) = idx.iter().map(|&i| (xs[i], ys[i])).unzip();
        xs[..n].copy_from_slice(&sx);
        ys[..n].copy_from_slice(&sy);
        let norm_pos = |x: f64| if x < 0.0 { x + 360.0 } else { x };
        let t = (180.0 - norm_pos(xs[0])) / (norm_pos(xs[n - 1]) - norm_pos(xs[0]));
        let y = ys[0] + t * (ys[n - 1] - ys[0]);
        let max_lon = decode_lon(encode_lon(180.0));
        let min_lon = decode_lon(encode_lon(-180.0));
        xs[n] = max_lon;
        ys[n] = y;
        xs[n + 1] = max_lon;
        ys[n + 1] = pole;
        xs[n + 2] = min_lon;
        ys[n + 2] = pole;
        xs[n + 3] = min_lon;
        ys[n + 3] = y;
        xs[n + 4] = xs[0];
        ys[n + 4] = ys[0];
        if south {
            xs.reverse();
            ys.reverse();
        }
    } else if n > 0 {
        xs[n] = xs[0];
        ys[n] = ys[0];
    }
    (xs, ys)
}

/// `H3CartesianUtil.toBoundingBox`: `(min_x, max_x, max_y, min_y)`
/// (`min_x > max_x` across the dateline).
fn h3_bbox(h: u64) -> (f64, f64, f64, f64) {
    let res = h3_res(h);
    h3_bbox_polar(h, polar(90.0, res), polar(-90.0, res))
}

fn h3_bbox_polar(h: u64, north: u64, south: u64) -> (f64, f64, f64, f64) {
    let (mut min_lat, mut min_lon, mut max_lat, mut max_lon) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let (mut max_neg_lon, mut min_pos_lon) = (f64::NEG_INFINITY, f64::INFINITY);
    for p in boundary(h) {
        let lon = decode_lon(encode_lon(p.0));
        let lat = decode_lat(encode_lat(p.1));
        min_lat = min_lat.min(lat);
        min_lon = min_lon.min(lon);
        max_lat = max_lat.max(lat);
        max_lon = max_lon.max(lon);
        if lon < 0.0 {
            max_neg_lon = max_neg_lon.max(lon);
        } else {
            min_pos_lon = min_pos_lon.min(lon);
        }
    }
    if h == north {
        (-180.0, 180.0, 90.0, min_lat)
    } else if h == south {
        (-180.0, 180.0, max_lat, -90.0)
    } else if max_lon - min_lon > 180.0 {
        (min_pos_lon, max_neg_lon, max_lat, min_lat)
    } else {
        (min_lon, max_lon, max_lat, min_lat)
    }
}

fn north_bound(res: u32) -> f64 {
    polar_table().north_bound[(res as usize).min(15)]
}
fn south_bound(res: u32) -> f64 {
    polar_table().south_bound[(res as usize).min(15)]
}

/// A cell's boundary as `GeoHexVisitor` relates to it (`computePoints`
/// and its box), computed once per cell and thread.
struct CellGeom {
    xs: Vec<f64>,
    ys: Vec<f64>,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    crosses: bool,
    convex: bool,
}

impl CellGeom {
    fn new(h: u64) -> CellGeom {
        let (xs, ys) = compute_points(h);
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..xs.len() {
            min_x = min_x.min(xs[i]);
            max_x = max_x.max(xs[i]);
            min_y = min_y.min(ys[i]);
            max_y = max_y.max(ys[i]);
        }
        let crosses = max_x - min_x > 180.0 && !is_polar(h);
        // Strictly convex (every turn the same way): a box with its corners
        // inside is inside.
        let n = xs.len().saturating_sub(1);
        let turn = |i: usize| {
            let (a, b, c) = (i % n, (i + 1) % n, (i + 2) % n);
            (xs[b] - xs[a]) * (ys[c] - ys[b]) - (ys[b] - ys[a]) * (xs[c] - xs[b])
        };
        let convex = n >= 3 && !crosses && ((0..n).all(|i| turn(i) > 0.0) || (0..n).all(|i| turn(i) < 0.0));
        CellGeom { xs, ys, min_x, max_x, min_y, max_y, crosses, convex }
    }

    /// The geometry of cell `h` (memoized: the same cells come up for
    /// doc after doc; dropped when the memo outgrows `MAX_CELLS`).
    fn of(h: u64) -> std::rc::Rc<CellGeom> {
        const MAX_CELLS: usize = 1 << 16;
        thread_local! {
            static CELLS: std::cell::RefCell<std::collections::HashMap<u64, std::rc::Rc<CellGeom>>> = Default::default();
        }
        CELLS.with(|c| {
            let mut c = c.borrow_mut();
            if let Some(g) = c.get(&h) {
                return g.clone();
            }
            if c.len() >= MAX_CELLS {
                c.clear();
            }
            let g = std::rc::Rc::new(CellGeom::new(h));
            c.insert(h, g.clone());
            g
        })
    }
}

/// ES `GeoHexVisitor` over one cell.
struct HexVisitor {
    relation: Option<GeoRelation>,
    g: std::rc::Rc<CellGeom>,
}

impl HexVisitor {
    fn new(h: u64) -> HexVisitor {
        HexVisitor { relation: None, g: CellGeom::of(h) }
    }

    /// Whether the doc's whole box is strictly inside the cell: then every
    /// point, line and triangle of the doc relates `QUERY_CONTAINS` and so
    /// does the visit (a convex cell, the box's corners inside with a
    /// margin, none on the boundary).
    fn contains_box(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        const EPS: f64 = 1e-9;
        if !self.g.convex || min_x - EPS <= self.g.min_x || max_x + EPS >= self.g.max_x || min_y - EPS <= self.g.min_y || max_y + EPS >= self.g.max_y {
            return false;
        }
        [(min_x - EPS, min_y - EPS), (max_x + EPS, min_y - EPS), (max_x + EPS, max_y + EPS), (min_x - EPS, max_y + EPS)].iter().all(|&(x, y)| self.relate_point_raw(x, y) == GeoRelation::QueryContains)
    }

    /// The doc's relation to the cell (`geoValue.visit(visitor)`).
    fn relate_value(&mut self, v: &ShapeValue<'_>) -> GeoRelation {
        #[cfg(test)]
        let shortcut = !NO_SHORTCUT.with(|c| c.get());
        #[cfg(not(test))]
        let shortcut = true;
        if let Some(e) = v.extent().filter(|_| shortcut) {
            let (max_x, min_x, max_y, min_y) = (e.neg_right.max(e.pos_right), e.neg_left.min(e.pos_left), e.top, e.bottom);
            if self.contains_box(decode_lon(min_x), decode_lon(max_x), decode_lat(min_y), decode_lat(max_y)) {
                return GeoRelation::QueryContains;
            }
        }
        v.visit(self);
        self.relation.unwrap_or(GeoRelation::QueryDisjoint)
    }

    fn update(&mut self, r: GeoRelation) {
        if r != GeoRelation::QueryDisjoint {
            if r == GeoRelation::QueryInside && self.relation != Some(GeoRelation::QueryContains) {
                self.relation = Some(GeoRelation::QueryInside);
            } else if r == GeoRelation::QueryContains && matches!(self.relation, None | Some(GeoRelation::QueryContains)) {
                self.relation = Some(GeoRelation::QueryContains);
            } else {
                self.relation = Some(GeoRelation::QueryCrosses);
            }
        } else {
            self.not_intersecting();
        }
    }

    fn not_intersecting(&mut self) {
        match self.relation {
            None => self.relation = Some(GeoRelation::QueryDisjoint),
            Some(GeoRelation::QueryContains) => self.relation = Some(GeoRelation::QueryCrosses),
            _ => {}
        }
    }

    fn boxes_disjoint(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        if !(max_y < self.g.min_y || min_y > self.g.max_y) {
            if self.g.crosses {
                return max_x < self.g.min_x && min_x > self.g.max_x;
            }
            return max_x < self.g.min_x || min_x > self.g.max_x;
        }
        true
    }

    fn norm(&self, pos: bool, x: f64) -> f64 {
        if pos {
            if x < 0.0 {
                x + 360.0
            } else {
                x
            }
        } else if x > 0.0 {
            x - 360.0
        } else {
            x
        }
    }

    fn relate_point_raw(&self, x: f64, y: f64) -> GeoRelation {
        let n = |x: f64| if self.g.crosses && x < 0.0 { x + 360.0 } else { x };
        let x = n(x);
        let mut res = false;
        for i in 0..self.g.xs.len().saturating_sub(1) {
            let (x1, x2, y1, y2) = (n(self.g.xs[i]), n(self.g.xs[i + 1]), self.g.ys[i], self.g.ys[i + 1]);
            if (y == y1 && y == y2) || ((y <= y1 && y >= y2) != (y >= y1 && y <= y2)) {
                if (x == x1 && x == x2) || (((x <= x1 && x >= x2) != (x >= x1 && x <= x2)) && orient(x1, y1, x2, y2, x, y) == 0) {
                    return GeoRelation::QueryCrosses;
                } else if (y1 > y) != (y2 > y) {
                    res ^= x < (x2 - x1) * (y - y1) / (y2 - y1) + x1;
                }
            }
        }
        if res {
            GeoRelation::QueryContains
        } else {
            GeoRelation::QueryDisjoint
        }
    }

    fn relate_point(&self, x: f64, y: f64) -> GeoRelation {
        if self.boxes_disjoint(x, x, y, y) {
            return GeoRelation::QueryDisjoint;
        }
        self.relate_point_raw(x, y)
    }

    #[allow(clippy::too_many_arguments)]
    fn crosses_line_n(&self, pos: Option<bool>, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, include_boundary: bool) -> bool {
        for i in 0..self.g.xs.len().saturating_sub(1) {
            let (cy, dy) = (self.g.ys[i], self.g.ys[i + 1]);
            let (cx, dx) = match pos {
                None => (self.g.xs[i], self.g.xs[i + 1]),
                Some(p) => (self.norm(p, self.g.xs[i]), self.norm(p, self.g.xs[i + 1])),
            };
            let (l_min_x, l_max_x, l_min_y, l_max_y) = (cx.min(dx), cx.max(dx), cy.min(dy), cy.max(dy));
            if l_max_x < min_x || l_min_x > max_x || l_min_y > max_y || l_max_y < min_y {
                continue;
            }
            let hit = if include_boundary { line_crosses_line_with_boundary(cx, cy, dx, dy, ax, ay, bx, by) } else { line_crosses_line(cx, cy, dx, dy, ax, ay, bx, by) };
            if hit {
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn crosses_line(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        if self.g.crosses {
            self.crosses_line_n(Some(true), min_x, max_x, min_y, max_y, ax, ay, bx, by, true) || self.crosses_line_n(Some(false), min_x, max_x, min_y, max_y, ax, ay, bx, by, true)
        } else {
            self.crosses_line_n(None, min_x, max_x, min_y, max_y, ax, ay, bx, by, true)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn crosses_triangle_n(&self, pos: Option<bool>, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
        for i in 0..self.g.xs.len().saturating_sub(1) {
            let (dy, ey) = (self.g.ys[i], self.g.ys[i + 1]);
            let (dx, ex) = match pos {
                None => (self.g.xs[i], self.g.xs[i + 1]),
                Some(p) => (self.norm(p, self.g.xs[i]), self.norm(p, self.g.xs[i + 1])),
            };
            let outside = (dy < min_y && ey < min_y) || (dy > max_y && ey > max_y) || (dx < min_x && ex < min_x) || (dx > max_x && ex > max_x);
            if !outside && (line_crosses_line_with_boundary(dx, dy, ex, ey, ax, ay, bx, by) || line_crosses_line_with_boundary(dx, dy, ex, ey, bx, by, cx, cy) || line_crosses_line_with_boundary(dx, dy, ex, ey, cx, cy, ax, ay)) {
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn crosses_triangle(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
        if self.g.crosses {
            self.crosses_triangle_n(Some(true), min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy) || self.crosses_triangle_n(Some(false), min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy)
        } else {
            self.crosses_triangle_n(None, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy)
        }
    }

    fn crosses_box_n(&self, pos: Option<bool>, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        for i in 0..self.g.xs.len().saturating_sub(1) {
            let (cy, dy) = (self.g.ys[i], self.g.ys[i + 1]);
            let (cx, dx) = match pos {
                None => (self.g.xs[i], self.g.xs[i + 1]),
                Some(p) => (self.norm(p, self.g.xs[i]), self.norm(p, self.g.xs[i + 1])),
            };
            let inb = |x: f64, y: f64| y >= min_y && y <= max_y && x >= min_x && x <= max_x;
            if inb(cx, cy) || inb(dx, dy) {
                return true;
            }
            let outside = (cy < min_y && dy < min_y) || (cy > max_y && dy > max_y) || (cx < min_x && dx < min_x) || (cx > max_x && dx > max_x);
            if !outside
                && (line_crosses_line_with_boundary(cx, cy, dx, dy, min_x, min_y, max_x, min_y)
                    || line_crosses_line_with_boundary(cx, cy, dx, dy, max_x, min_y, max_x, max_y)
                    || line_crosses_line_with_boundary(cx, cy, dx, dy, max_x, max_y, min_x, max_y)
                    || line_crosses_line_with_boundary(cx, cy, dx, dy, min_x, max_y, min_x, min_y))
            {
                return true;
            }
        }
        false
    }

    fn relate_bbox(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        if self.boxes_disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        if min_x <= self.g.xs[0] && max_x >= self.g.xs[0] && min_y <= self.g.ys[0] && max_y >= self.g.ys[0] {
            return true;
        }
        let crosses_box = if self.g.crosses { self.crosses_box_n(Some(true), min_x, max_x, min_y, max_y) || self.crosses_box_n(Some(false), min_x, max_x, min_y, max_y) } else { self.crosses_box_n(None, min_x, max_x, min_y, max_y) };
        self.relate_point(min_x, min_y) != GeoRelation::QueryDisjoint || crosses_box
    }

    /// `intersectsBbox(minX, maxX, minY, maxY)` (a box may cross the dateline).
    fn intersects_bbox(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        if self.g.xs.is_empty() {
            return false;
        }
        if min_x > max_x {
            self.relate_bbox(min_x, 180.0, min_y, max_y) || self.relate_bbox(-180.0, max_x, min_y, max_y)
        } else {
            self.relate_bbox(min_x, max_x, min_y, max_y)
        }
    }

    fn edge_intersects_query(&self, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        let (min_x, max_x, min_y, max_y) = (ax.min(bx), ax.max(bx), ay.min(by), ay.max(by));
        !self.boxes_disjoint(min_x, max_x, min_y, max_y) && self.crosses_line(min_x, max_x, min_y, max_y, ax, ay, bx, by)
    }

    fn relate_line(&self, ax: f64, ay: f64, bx: f64, by: f64) -> GeoRelation {
        let r1 = self.relate_point(ax, ay);
        let r2 = self.relate_point(bx, by);
        if r1 != r2 || r1 == GeoRelation::QueryCrosses {
            return GeoRelation::QueryCrosses;
        } else if r1 == GeoRelation::QueryContains {
            if self.g.crosses {
                let (min_x, max_x, min_y, max_y) = (ax.min(bx), ax.max(bx), ay.min(by), ay.max(by));
                if self.crosses_line(min_x, max_x, min_y, max_y, ax, ay, bx, by) {
                    return GeoRelation::QueryCrosses;
                }
            }
            return GeoRelation::QueryContains;
        }
        if self.edge_intersects_query(ax, ay, bx, by) {
            return GeoRelation::QueryCrosses;
        }
        GeoRelation::QueryDisjoint
    }

    #[allow(clippy::too_many_arguments)]
    fn relate_triangle(&self, ax: f64, ay: f64, ab: bool, bx: f64, by: f64, bc: bool, cx: f64, cy: f64, ca: bool) -> GeoRelation {
        let (t_min_x, t_max_x, t_min_y, t_max_y) = (ax.min(bx).min(cx), ax.max(bx).max(cx), ay.min(by).min(cy), ay.max(by).max(cy));
        if self.boxes_disjoint(t_min_x, t_max_x, t_min_y, t_max_y) {
            return GeoRelation::QueryDisjoint;
        }
        let (r1, r2, r3) = (self.relate_point(ax, ay), self.relate_point(bx, by), self.relate_point(cx, cy));
        if r1 != r2 || r1 != r3 || r1 == GeoRelation::QueryCrosses {
            return GeoRelation::QueryCrosses;
        } else if r1 == GeoRelation::QueryContains {
            if self.g.crosses && self.crosses_triangle(t_min_x, t_max_x, t_min_y, t_max_y, ax, ay, bx, by, cx, cy) {
                return GeoRelation::QueryCrosses;
            }
            return GeoRelation::QueryContains;
        }
        let mut within = false;
        for (x1, y1, x2, y2, from_poly) in [(ax, ay, bx, by, ab), (bx, by, cx, cy, bc), (cx, cy, ax, ay, ca)] {
            if self.edge_intersects_query(x1, y1, x2, y2) {
                if from_poly {
                    return GeoRelation::QueryCrosses;
                }
                within = true;
            }
        }
        if within || point_in_triangle(t_min_x, t_max_x, t_min_y, t_max_y, self.g.xs[0], self.g.ys[0], ax, ay, bx, by, cx, cy) {
            return GeoRelation::QueryInside;
        }
        GeoRelation::QueryDisjoint
    }
}

impl TreeVisitor for HexVisitor {
    fn visit_point(&mut self, x: i32, y: i32) {
        let r = self.relate_point(decode_lon(x), decode_lat(y));
        self.update(r);
    }
    fn visit_line(&mut self, ax: i32, ay: i32, bx: i32, by: i32, _m: u8) {
        let r = self.relate_line(decode_lon(ax), decode_lat(ay), decode_lon(bx), decode_lat(by));
        self.update(r);
    }
    fn visit_triangle(&mut self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, m: u8) {
        let r = self.relate_triangle(decode_lon(ax), decode_lat(ay), ab_from_triangle(m), decode_lon(bx), decode_lat(by), bc_from_triangle(m), decode_lon(cx), decode_lat(cy), ca_from_triangle(m));
        self.update(r);
    }
    fn push(&mut self) -> bool {
        self.relation != Some(GeoRelation::QueryCrosses)
    }
    fn push_x(&mut self, min_x: i32) -> bool {
        if self.g.crosses || self.g.max_x >= decode_lon(min_x) {
            return true;
        }
        self.not_intersecting();
        false
    }
    fn push_y(&mut self, min_y: i32) -> bool {
        if self.g.max_y >= decode_lat(min_y) {
            return true;
        }
        self.not_intersecting();
        false
    }
    fn push_max(&mut self, max_x: i32, max_y: i32) -> bool {
        if self.g.min_y <= decode_lat(max_y) && (self.g.crosses || self.g.min_x <= decode_lon(max_x)) {
            return true;
        }
        self.not_intersecting();
        false
    }
    fn push_box(&mut self, min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> bool {
        if self.boxes_disjoint(decode_lon(min_x), decode_lon(max_x), decode_lat(min_y), decode_lat(max_y)) {
            self.relation = Some(GeoRelation::QueryDisjoint);
            return false;
        }
        self.relation = None;
        true
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: relate every cell by visiting the doc (no box shortcut).
    static NO_SHORTCUT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A doc's H3 cells at `precision` by brute force: from the cells of its
/// vertices, every neighbour reached through cells the doc touches, each
/// related by visiting the doc (`bounds` applied as at the precision).
#[cfg(test)]
pub(crate) fn brute_force_hex_cells(precision: u32, bounds: Option<&Bounds>, v: &ShapeValue<'_>) -> std::collections::BTreeSet<String> {
    NO_SHORTCUT.with(|c| c.set(true));
    let t = HexTiler::new(precision, bounds);
    let mut seeds = std::collections::BTreeSet::new();
    v.for_each_triangle(&mut |tr| {
        for (x, y) in [(tr.ax, tr.ay), (tr.bx, tr.by), (tr.cx, tr.cy)] {
            seeds.insert(geo_to_h3(decode_lat(y), decode_lon(x), precision));
        }
    });
    let mut seen: std::collections::HashSet<u64> = seeds.iter().copied().collect();
    let mut queue: Vec<u64> = seeds.iter().copied().collect();
    let mut out = std::collections::BTreeSet::new();
    let unbounded = HexTiler::new(precision, None);
    while let Some(h) = queue.pop() {
        // Reach on the doc alone (a vertex's own cell is expanded even when
        // its planar boundary misses the vertex: a neighbour holds it);
        // keep the cells the bounds allow.
        if unbounded.relate_tile(v, h) == GeoRelation::QueryDisjoint && !seeds.contains(&h) {
            continue;
        }
        if t.relate_tile(v, h) != GeoRelation::QueryDisjoint {
            out.insert(hex_key(h));
        }
        if h3::is_valid(h) {
            for n in std::iter::once(h).chain(h3::hex_ring(h)) {
                if seen.insert(n) {
                    queue.push(n);
                }
            }
        }
    }
    NO_SHORTCUT.with(|c| c.set(false));
    out
}

/// Test counters: hex cells related, docs celled by recursion.
#[cfg(test)]
pub(crate) static HEX_STATS: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// `GeoHexGridTiler` (bounded when `bounds` is set).
pub(crate) struct HexTiler {
    precision: u32,
    bounds: Option<Bounds>,
    /// `inflatedBboxes` of the bounded tiler, per coarser resolution.
    inflated: Vec<Bounds>,
}

fn normalize_lon(lon: f64) -> f64 {
    // ES `GeoUtils.normalizeLon`.
    if lon > 180.0 || lon <= -180.0 {
        let m = (lon + 180.0).rem_euclid(360.0) - 180.0;
        if m == -180.0 {
            180.0
        } else {
            m
        }
    } else {
        lon
    }
}

impl HexTiler {
    pub(crate) fn new(precision: u32, bounds: Option<&Bounds>) -> HexTiler {
        let inflated = match bounds {
            Some(b) => (0..precision).map(|r| Self::inflate(r, b, 0.37)).collect(),
            None => Vec::new(),
        };
        HexTiler { precision, bounds: bounds.copied(), inflated }
    }

    fn inflate(res: u32, b: &Bounds, factor: f64) -> Bounds {
        let height = |r: (f64, f64, f64, f64)| r.2 - r.3;
        let width = |r: (f64, f64, f64, f64)| if r.0 > r.1 { 360.0 + r.1 - r.0 } else { r.1 - r.0 };
        let min_min = h3_bbox(geo_to_h3(b.bottom, b.left, res));
        let max_max = h3_bbox(geo_to_h3(b.top, b.right, res));
        let h = height(min_min).max(height(max_max));
        let w = width(min_min).max(width(max_max));
        let min_y = (b.bottom - factor * h).max(-90.0);
        let max_y = (b.top + factor * h).min(90.0);
        let left = normalize_lon(b.left - factor * w);
        let right = normalize_lon(b.right + factor * w);
        let bw = if b.left > b.right { 360.0 + b.right - b.left } else { b.right - b.left };
        if 2.0 * factor * w + bw >= 360.0 {
            Bounds { top: max_y, left: -180.0, bottom: min_y, right: 180.0 }
        } else {
            Bounds { top: max_y, left, bottom: min_y, right }
        }
    }

    fn h3_intersects_bounds(&self, h: u64) -> bool {
        let Some(b) = &self.bounds else { return true };
        let v = HexVisitor::new(h);
        let res = h3_res(h);
        let bb = if res != self.precision { &self.inflated[res as usize] } else { b };
        v.intersects_bbox(bb.left, bb.right, bb.bottom, bb.top)
    }

    fn relate_tile(&self, v: &ShapeValue<'_>, h: u64) -> GeoRelation {
        #[cfg(test)]
        HEX_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut vis = HexVisitor::new(h);
        let res = h3_res(h);
        match &self.bounds {
            Some(b) => {
                if res != self.precision {
                    let ib = &self.inflated[res as usize];
                    if vis.intersects_bbox(ib.left, ib.right, ib.bottom, ib.top) {
                        if vis.g.max_y > north_bound(res) || vis.g.min_y < south_bound(res) {
                            return GeoRelation::QueryCrosses;
                        }
                        return vis.relate_value(v);
                    }
                    return GeoRelation::QueryDisjoint;
                }
                if vis.intersects_bbox(b.left, b.right, b.bottom, b.top) {
                    return vis.relate_value(v);
                }
                GeoRelation::QueryDisjoint
            }
            None => {
                if res != self.precision && (vis.g.max_y > north_bound(res) || vis.g.min_y < south_bound(res)) {
                    return GeoRelation::QueryCrosses;
                }
                vis.relate_value(v)
            }
        }
    }

    fn value_inside_bounds(&self, v: &ShapeValue<'_>) -> bool {
        let Some(b) = &self.bounds else { return true };
        let Some(bb) = v.bounding_box() else { return false };
        if b.bottom <= bb.min_y() && b.top >= bb.max_y() {
            if b.right < b.left {
                return b.left <= bb.min_x() || b.right >= bb.max_x();
            }
            return b.left <= bb.min_x() && b.right >= bb.max_x();
        }
        false
    }

    pub(crate) fn set_values(&self, v: &ShapeValue<'_>, out: &mut Vec<u64>) {
        let Some(bb) = v.bounding_box() else { return };
        if bb.max_x() - bb.min_x() < 180.0 {
            if let Some(single) = Self::bounds_in_same_cell(&bb, self.precision) {
                self.from_point_resolution(single, v, out);
                return;
            }
        }
        self.by_recursion(v, &bb, out);
    }

    fn bounds_in_same_cell(bb: &rustic_core::fieldtypes::shape_dv::BoundingBox, res: u32) -> Option<u64> {
        let min = geo_to_h3(bb.min_y(), bb.min_x(), res);
        // A point's corners are one point: its cell, computed once.
        if bb.min_x() == bb.max_x() && bb.min_y() == bb.max_y() {
            return (min > 0).then_some(min);
        }
        let max = geo_to_h3(bb.max_y(), bb.max_x(), res);
        if min != max {
            return None;
        }
        if is_polar(min) {
            let min_max = geo_to_h3(bb.min_y(), bb.max_x(), res);
            let max_min = geo_to_h3(bb.max_y(), bb.min_x(), res);
            if min_max != min || max_min != min {
                return None;
            }
        }
        (min > 0).then_some(min)
    }

    fn maybe_add(h: u64, r: GeoRelation, out: &mut Vec<u64>) {
        if r != GeoRelation::QueryDisjoint {
            out.push(h);
        }
    }

    fn from_point_resolution(&self, h: u64, v: &ShapeValue<'_>, out: &mut Vec<u64>) {
        let r = self.relate_tile(v, h);
        Self::maybe_add(h, r, out);
        if r == GeoRelation::QueryContains {
            return;
        }
        for i in 0..hex_ring_size(h) {
            let n = hex_ring_pos(h, i);
            let r = self.relate_tile(v, n);
            Self::maybe_add(n, r, out);
            if r == GeoRelation::QueryContains {
                return;
            }
        }
    }

    fn by_recursion(&self, v: &ShapeValue<'_>, bb: &rustic_core::fieldtypes::shape_dv::BoundingBox, out: &mut Vec<u64>) {
        #[cfg(test)]
        HEX_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The doc's relation to each cell, once (the recursion asks again
        // for the parents of the cells no child covers).
        let mut memo = std::collections::HashMap::new();
        if bb.max_x() - bb.min_x() < 180.0 {
            if let Some(single) = Self::bounds_in_same_cell(bb, 0) {
                self.recurse(v, single, 0, out, &mut memo);
                for i in 0..hex_ring_size(single) {
                    self.recurse(v, hex_ring_pos(single, i), 0, out, &mut memo);
                }
                return;
            }
        }
        for c in h3::res0_cells() {
            self.recurse(v, c, 0, out, &mut memo);
        }
    }

    fn recurse(&self, v: &ShapeValue<'_>, h: u64, precision: u32, out: &mut Vec<u64>, memo: &mut std::collections::HashMap<u64, GeoRelation>) {
        let r = self.relate_memo(v, h, memo);
        if precision == self.precision {
            Self::maybe_add(h, r, out);
            return;
        }
        if r != GeoRelation::QueryDisjoint {
            let mut i = 0;
            if r == GeoRelation::QueryInside {
                let center = child_pos(h, i);
                i += 1;
                self.all_values(center, precision + 1, self.value_inside_bounds(v), out);
            }
            let n = children_size(h);
            while i < n {
                self.recurse(v, child_pos(h, i), precision + 1, out, memo);
                i += 1;
            }
            for j in 0..not_intersecting_children_size(h) {
                let no_child = no_child_intersecting_pos(h, j);
                if no_child != 0 && self.relate_memo(v, h3_parent(no_child), memo) == GeoRelation::QueryDisjoint {
                    self.recurse(v, no_child, precision + 1, out, memo);
                }
            }
        }
    }

    fn relate_memo(&self, v: &ShapeValue<'_>, h: u64, memo: &mut std::collections::HashMap<u64, GeoRelation>) -> GeoRelation {
        *memo.entry(h).or_insert_with(|| self.relate_tile(v, h))
    }

    fn all_values(&self, h: u64, precision: u32, inside: bool, out: &mut Vec<u64>) {
        if inside || self.h3_intersects_bounds(h) {
            if precision == self.precision {
                out.push(h);
            } else {
                for i in 0..children_size(h) {
                    self.all_values(child_pos(h, i), precision + 1, inside, out);
                }
            }
        }
    }
}

/// The key of an H3 cell.
pub(crate) fn hex_key(h: u64) -> String {
    if h3::is_valid(h) {
        h3::to_string(h)
    } else {
        String::new()
    }
}

/// One grid's tiler.
pub(crate) enum Tiler {
    Tile(TileTiler),
    Hash(HashTiler),
    Hex(HexTiler),
}

impl Tiler {
    /// The doc's cells, sorted and distinct (`GeoShapeCellValues` sorted,
    /// the aggregator skipping repeats).
    pub(crate) fn cells(&self, v: &ShapeValue<'_>, out: &mut Vec<u64>) {
        out.clear();
        match self {
            Tiler::Tile(t) => t.set_values(v, out),
            Tiler::Hash(t) => t.set_values(v, out),
            Tiler::Hex(t) => t.set_values(v, out),
        }
        out.sort_unstable();
        out.dedup();
    }

    pub(crate) fn key(&self, h: u64) -> String {
        match self {
            Tiler::Tile(_) => tile_key(h),
            Tiler::Hash(_) => gh_key(h),
            Tiler::Hex(_) => hex_key(h),
        }
    }
}

/// The cells of a doc by a brute-force scan over its bounding box (every
/// cell at `precision` related, `bounds` applied; no descent), for tests.
#[cfg(test)]
pub(crate) fn brute_force_cells(grid: &str, precision: u32, bounds: Option<&Bounds>, v: &ShapeValue<'_>, out: &mut std::collections::BTreeSet<String>) {
    let Some(b) = v.bounding_box() else { return };
    let mut cells = Vec::new();
    match grid {
        "geotile_grid" => {
            if b.bottom > normalized_lat_mask() || b.top < normalized_neg_lat_mask() {
                return;
            }
            let t = TileTiler::new(precision, bounds);
            let tiles = 1u64 << precision;
            t.brute(v, get_x_tile(b.min_x(), tiles), get_y_tile(b.max_y(), tiles), get_x_tile(b.max_x(), tiles), get_y_tile(b.min_y(), tiles), &mut cells);
            out.extend(cells.into_iter().map(tile_key));
        }
        _ => {
            let t = HashTiler::new(precision, bounds);
            if b.min_x() == b.max_x() && b.min_y() == b.max_y() {
                t.set_value(v, &gh_string_encode(b.min_x(), b.min_y(), precision), &mut cells);
            } else {
                t.brute(v, &b, &mut cells);
            }
            out.extend(cells.into_iter().map(gh_key));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geohash_utils_match_es() {
        // ES `Geohash.stringEncode` / `toBoundingBox` round trip.
        let h = gh_string_encode(-0.1278, 51.5074, 7);
        assert_eq!(h, "gcpvj0d");
        let (min_x, max_x, max_y, min_y) = gh_bbox(&h);
        assert!(min_x <= -0.1278 && max_x >= -0.1278 && min_y <= 51.5074 && max_y >= 51.5074);
        assert_eq!(gh_key(gh_long_encode(&h)), h);
        assert_eq!(gh_neighbor("u", 1, 1, 0).as_deref(), Some("v"));
        assert_eq!(gh_neighbor("gcpvj0d", 7, 0, 1).map(|s| gh_bbox(&s).3), Some(max_y));
    }

    #[test]
    fn tiles_match_es_formulas() {
        let tiles = 1u64 << 9;
        assert_eq!(get_x_tile(-180.0, tiles), 0);
        assert_eq!(get_x_tile(180.0, tiles), 511);
        assert_eq!(get_y_tile(85.0511287798066, tiles), 0);
        let k = long_encode_tiles(9, 255, 170);
        assert_eq!(tile_key(k), "9/255/170");
        assert!((tile_to_lat(0, tiles) - LATITUDE_MASK).abs() < 1e-9);
    }
}
