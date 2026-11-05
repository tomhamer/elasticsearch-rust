//! A port of ES 9.5.5's own H3 library (`libs/h3`, `org.elasticsearch.h3`:
//! `H3`, `H3Index`, `FaceIJK`, `CoordIJK`, `Vec2d`, `Vec3d`, `LatLng`,
//! `BaseCells`, `HexRing`, `FastMath`), so geohex cells, boundaries,
//! parents, children and neighbours are ES's to the bit, including at cell
//! edges where its table-based trigonometry ([`fast_math`]) differs from
//! the exact maths of the reference H3 library.
//!
//! Cells are `u64` H3 indexes; coordinates are degrees at the API and
//! radians inside, as in ES.

pub mod fast_math;
mod tables;

use fast_math as fm;
use tables::*;

pub const MAX_H3_RES: i32 = 15;

const M_SQRT3_2: f64 = 0.8660254037844386467637231707529361834714;
const M_2PI: f64 = 2.0 * std::f64::consts::PI;
const NUM_BASE_CELLS: i32 = 122;
const NUM_HEX_VERTS: usize = 6;
const NUM_PENT_VERTS: usize = 5;
const H3_CELL_MODE: u64 = 1;
const M_SQRT7: f64 = 2.6457513110645905905016157536392604257102;
const M_RSQRT7: f64 = 1.0 / M_SQRT7;
const RES0_U_GNOMONIC: f64 = 0.38196601125010500003;
const M_AP7_ROT_RADS: f64 = 0.333473172251832115336090755351601070065900389;
const EPSILON: f64 = 0.0000000000000001;
const M_PI_2: f64 = 1.5707963267948966;
const MINIMUM_ANGULAR_RESOLUTION: f64 = std::f64::consts::PI * 1.0e-12;

// Digits (`CoordIJK.Direction`).
const CENTER_DIGIT: i32 = 0;
const K_AXES_DIGIT: i32 = 1;
const J_AXES_DIGIT: i32 = 2;
const JK_AXES_DIGIT: i32 = 3;
const I_AXES_DIGIT: i32 = 4;
const IK_AXES_DIGIT: i32 = 5;
const IJ_AXES_DIGIT: i32 = 6;
const INVALID_DIGIT: i32 = 7;
const NUM_DIGITS: i32 = 7;

/// `Math.round(double)` (ties toward positive infinity), as an `int` cast.
#[inline]
fn java_round(x: f64) -> i32 {
    if x.is_nan() {
        return 0;
    }
    let f = x.floor();
    let r = if x - f >= 0.5 { f + 1.0 } else { f };
    r as i64 as i32
}

/// `Math.toRadians`.
#[inline]
fn to_radians(deg: f64) -> f64 {
    deg * 0.017453292519943295
}

/// `Math.toDegrees`.
#[inline]
fn to_degrees(rad: f64) -> f64 {
    rad * 57.29577951308232
}

// ---------------------------------------------------------------------------
// LatLng

/// A point, radians (`LatLng`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatLng {
    pub lat: f64,
    pub lon: f64,
}

impl LatLng {
    pub fn lat_deg(&self) -> f64 {
        to_degrees(self.lat)
    }

    pub fn lon_deg(&self) -> f64 {
        to_degrees(self.lon)
    }

    fn geo_azimuth_rads(&self, lat: f64, lon: f64) -> f64 {
        let cos_lat = fm::cos(lat);
        fm::atan2(cos_lat * fm::sin(lon - self.lon), fm::cos(self.lat) * fm::sin(lat) - fm::sin(self.lat) * cos_lat * fm::cos(lon - self.lon))
    }

    /// `greatCircleMaxLatitude`.
    pub fn great_circle_max_latitude(&self, other: &LatLng) -> f64 {
        if self.is_numerically_identical(other) {
            return other.lat;
        }
        if other.lat > self.lat {
            gc_max_lat(other, self)
        } else {
            gc_max_lat(self, other)
        }
    }

    /// `greatCircleMinLatitude`.
    pub fn great_circle_min_latitude(&self, other: &LatLng) -> f64 {
        if self.is_numerically_identical(other) {
            return other.lat;
        }
        if other.lat < self.lat {
            gc_min_lat(other, self)
        } else {
            gc_min_lat(self, other)
        }
    }

    fn is_numerically_identical(&self, o: &LatLng) -> bool {
        (self.lat - o.lat).abs() < MINIMUM_ANGULAR_RESOLUTION && (self.lon - o.lon).abs() < MINIMUM_ANGULAR_RESOLUTION
    }
}

fn gc_max_lat(a: &LatLng, b: &LatLng) -> f64 {
    let az = a.geo_azimuth_rads(b.lat, b.lon);
    if az.abs() < M_PI_2 {
        return fm::acos((fm::sin(az) * fm::cos(a.lat)).abs());
    }
    a.lat
}

fn gc_min_lat(a: &LatLng, b: &LatLng) -> f64 {
    let az = a.geo_azimuth_rads(b.lat, b.lon);
    if az.abs() > M_PI_2 {
        return -fm::acos((fm::sin(az) * fm::cos(a.lat)).abs());
    }
    a.lat
}

// ---------------------------------------------------------------------------
// Vec2d / Vec3d

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vec2d {
    x: f64,
    y: f64,
}

const VEC2D_RESOLUTION: f64 = 1e-7;
const M_RSIN60: f64 = 1.0 / M_SQRT3_2;
const M_ONETHIRD: f64 = 1.0 / 3.0;

#[inline]
fn pos_angle_rads(rads: f64) -> f64 {
    if rads < 0.0 {
        rads + M_2PI
    } else if rads >= M_2PI {
        rads - M_2PI
    } else {
        rads
    }
}

#[inline]
fn is_class_iii(res: i32) -> bool {
    res & 1 == 1
}

impl Vec2d {
    fn hex2d_to_geo(&self, face: usize, res: i32, substrate: bool) -> LatLng {
        hex2d_to_geo(self.x, self.y, face, res, substrate)
    }

    fn numerically_identical(&self, o: &Vec2d) -> bool {
        (o.x - self.x).abs() < VEC2D_RESOLUTION && (o.y - self.y).abs() < VEC2D_RESOLUTION
    }
}

fn hex2d_to_geo(x: f64, y: f64, face: usize, res: i32, substrate: bool) -> LatLng {
    let mut r = (x * x + y * y).sqrt();
    if r < EPSILON {
        let (lat, lon) = FACE_CENTER_GEO[face];
        return LatLng { lat, lon };
    }
    let mut theta = fm::atan2(y, x);
    for _ in 0..res {
        r *= M_RSQRT7;
    }
    if substrate {
        r *= M_ONETHIRD;
        if is_class_iii(res) {
            r *= M_RSQRT7;
        }
    }
    r *= RES0_U_GNOMONIC;
    r = fm::atan(r);
    if !substrate && is_class_iii(res) {
        theta = pos_angle_rads(theta + M_AP7_ROT_RADS);
    }
    theta = pos_angle_rads(FACE_AXES_AZ_RADS_CII[face][0] - theta);
    geo_az_distance_rads(&FACE_CENTER_POINT[face], theta, r)
}

fn hex2d_to_coord_ijk(x: f64, y: f64) -> CoordIJK {
    let a1 = x.abs();
    let a2 = y.abs();
    let x2 = a2 * M_RSIN60;
    let x1 = a1 + x2 * 0.5;
    let m1 = x1 as i32;
    let m2 = x2 as i32;
    let r1 = x1 - m1 as f64;
    let r2 = x2 - m2 as f64;
    let (mut i, mut j);
    if r1 < 0.5 {
        if r1 < M_ONETHIRD {
            if r2 < (1.0 + r1) * 0.5 {
                i = m1;
                j = m2;
            } else {
                i = m1;
                j = m2 + 1;
            }
        } else {
            j = if r2 < (1.0 - r1) { m2 } else { m2 + 1 };
            i = if (1.0 - r1) <= r2 && r2 < (2.0 * r1) { m1 + 1 } else { m1 };
        }
    } else if r1 < 2.0 * M_ONETHIRD {
        j = if r2 < (1.0 - r1) { m2 } else { m2 + 1 };
        i = if (2.0 * r1 - 1.0) < r2 && r2 < (1.0 - r1) { m1 } else { m1 + 1 };
    } else if r2 < (r1 * 0.5) {
        i = m1 + 1;
        j = m2;
    } else {
        i = m1 + 1;
        j = m2 + 1;
    }
    if x < 0.0 {
        if j % 2 == 0 {
            let axisi = j / 2;
            let diff = i - axisi;
            i -= 2 * diff;
        } else {
            let axisi = (j + 1) / 2;
            let diff = i - axisi;
            i -= 2 * diff + 1;
        }
    }
    if y < 0.0 {
        i -= (2 * j + 1) / 2;
        j = -j;
    }
    let mut c = CoordIJK { i, j, k: 0 };
    c.normalize();
    c
}

fn v2d_intersect(p0: Vec2d, p1: Vec2d, p2: Vec2d, p3: Vec2d) -> Vec2d {
    let s1x = p1.x - p0.x;
    let s1y = p1.y - p0.y;
    let s2x = p3.x - p2.x;
    let s2y = p3.y - p2.y;
    let t = (s2x * (p0.y - p2.y) - s2y * (p0.x - p2.x)) / (-s2x * s1y + s1x * s2y);
    Vec2d { x: p0.x + (t * s1x), y: p0.y + (t * s1y) }
}

#[inline]
fn square(x: f64) -> f64 {
    x * x
}

#[inline]
fn magnitude(x: f64, y: f64, z: f64) -> f64 {
    (square(x) + square(y) + square(z)).sqrt()
}

/// `Vec3d.geoAzimuthRads` of face centre `c` toward the unit vector `(x, y, z)`.
fn vec3_geo_azimuth_rads(c: &[f64; 3], x: f64, y: f64, z: f64) -> f64 {
    let (tx, ty, tz) = (c[0], c[1], c[2]);
    let c1x = ty * z - tz * y;
    let c1y = tz * x - tx * z;
    let c1z = tx * y - ty * x;
    let c2x = ty;
    let c2y = -tx;
    let c2z = 0.0;
    let c1c2x = c1y * c2z - c1z * c2y;
    let c1c2y = c1z * c2x - c1x * c2z;
    let c1c2z = c1x * c2y - c1y * c2x;
    let dot = tx * c1c2x + ty * c1c2y + tz * c1c2z;
    let sign = java_signum(dot);
    fm::atan2(sign * magnitude(c1c2x, c1c2y, c1c2z), c1x * c2x + c1y * c2y + c1z * c2z)
}

/// `Math.signum`.
#[inline]
fn java_signum(d: f64) -> f64 {
    if d == 0.0 || d.is_nan() {
        d
    } else if d > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// `Vec3d.geoAzDistanceRads` from face centre `c`.
fn geo_az_distance_rads(c: &[f64; 3], az: f64, distance: f64) -> LatLng {
    let az = pos_angle_rads(az);
    let (tx, ty, tz) = (c[0], c[1], c[2]);
    let m = magnitude(tx, ty, 0.0);
    let de_x = -ty / m;
    let de_y = tx / m;
    let dn_x = -tz * de_y;
    let dn_y = tz * de_x;
    let dn_z = tx * de_y - ty * de_x;
    let sin_az = fm::sin(az);
    let cos_az = fm::cos(az);
    let sin_d = fm::sin(distance);
    let cos_d = fm::cos(distance);
    let d_x = dn_x * cos_az + de_x * sin_az;
    let d_y = dn_y * cos_az + de_y * sin_az;
    let d_z = dn_z * cos_az;
    let n2x = tx * cos_d + d_x * sin_d;
    let n2y = ty * cos_d + d_y * sin_d;
    let n2z = tz * cos_d + d_z * sin_d;
    LatLng { lat: fm::asin(n2z), lon: fm::atan2(n2y, n2x) }
}

/// `Vec3d.geoToH3`.
fn vec3_geo_to_h3(res: i32, lat: f64, lon: f64) -> u64 {
    let cos_lat = fm::cos(lat);
    let z = fm::sin(lat);
    let x = fm::cos(lon) * cos_lat;
    let y = fm::sin(lon) * cos_lat;
    let dist = |c: &[f64; 3]| square(x - c[0]) + square(y - c[1]) + square(z - c[2]);
    let mut face = 0usize;
    let mut sqd = dist(&FACE_CENTER_POINT[0]);
    for (i, c) in FACE_CENTER_POINT.iter().enumerate().skip(1) {
        let t = dist(c);
        if t < sqd {
            face = i;
            sqd = t;
        }
    }
    let mut r = fm::acos(1.0 - sqd * 0.5);
    if r < EPSILON {
        return face_ijk_to_h3(res, face, CoordIJK { i: 0, j: 0, k: 0 });
    }
    let mut theta = pos_angle_rads(FACE_AXES_AZ_RADS_CII[face][0] - pos_angle_rads(vec3_geo_azimuth_rads(&FACE_CENTER_POINT[face], x, y, z)));
    if is_class_iii(res) {
        theta = pos_angle_rads(theta - M_AP7_ROT_RADS);
    }
    r = fm::tan(r);
    r /= RES0_U_GNOMONIC;
    for _ in 0..res {
        r *= M_SQRT7;
    }
    face_ijk_to_h3(res, face, hex2d_to_coord_ijk(r * fm::cos(theta), r * fm::sin(theta)))
}

// ---------------------------------------------------------------------------
// CoordIJK

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CoordIJK {
    i: i32,
    j: i32,
    k: i32,
}

const UNIT_VECS: [[i32; 3]; 7] = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]];
const M_ONESEVENTH: f64 = 1.0 / 7.0;

impl CoordIJK {
    fn reset(&mut self, i: i32, j: i32, k: i32) {
        *self = CoordIJK { i, j, k };
    }

    fn to_hex2d(&self) -> Vec2d {
        let i = self.i - self.k;
        let j = self.j - self.k;
        Vec2d { x: i as f64 - 0.5 * j as f64, y: j as f64 * M_SQRT3_2 }
    }

    fn to_geo(&self, face: usize, res: i32, substrate: bool) -> LatLng {
        let i = self.i - self.k;
        let j = self.j - self.k;
        hex2d_to_geo(i as f64 - 0.5 * j as f64, j as f64 * M_SQRT3_2, face, res, substrate)
    }

    fn add(&mut self, i: i32, j: i32, k: i32) {
        self.i += i;
        self.j += j;
        self.k += k;
    }

    fn sub(&mut self, i: i32, j: i32, k: i32) {
        self.i -= i;
        self.j -= j;
        self.k -= k;
    }

    fn normalize(&mut self) {
        let min = self.i.min(self.j.min(self.k));
        self.sub(min, min, min);
    }

    fn down_ap7(&mut self) {
        let (i, j, k) = (self.i * 3 + self.j, self.j * 3 + self.k, self.k * 3 + self.i);
        self.reset(i, j, k);
        self.normalize();
    }

    fn down_ap7r(&mut self) {
        let (i, j, k) = (self.i * 3 + self.k, self.j * 3 + self.i, self.k * 3 + self.j);
        self.reset(i, j, k);
        self.normalize();
    }

    fn down_ap3(&mut self) {
        let (i, j, k) = (self.i * 2 + self.j, self.j * 2 + self.k, self.k * 2 + self.i);
        self.reset(i, j, k);
        self.normalize();
    }

    fn down_ap3r(&mut self) {
        let (i, j, k) = (self.i * 2 + self.k, self.j * 2 + self.i, self.k * 2 + self.j);
        self.reset(i, j, k);
        self.normalize();
    }

    fn rotate60cw(&mut self) {
        let (i, j, k) = (self.i + self.j, self.j + self.k, self.i + self.k);
        self.reset(i, j, k);
        self.normalize();
    }

    fn rotate60ccw(&mut self) {
        let (i, j, k) = (self.i + self.k, self.i + self.j, self.j + self.k);
        self.reset(i, j, k);
        self.normalize();
    }

    fn neighbor(&mut self, digit: i32) {
        if digit > CENTER_DIGIT && digit < NUM_DIGITS {
            let u = UNIT_VECS[digit as usize];
            self.add(u[0], u[1], u[2]);
            self.normalize();
        }
    }

    fn up_ap7r(&mut self) {
        let i = self.i - self.k;
        let j = self.j - self.k;
        self.i = java_round((2 * i + j) as f64 * M_ONESEVENTH);
        self.j = java_round((3 * j - i) as f64 * M_ONESEVENTH);
        self.k = 0;
        self.normalize();
    }

    fn up_ap7(&mut self) {
        let i = self.i - self.k;
        let j = self.j - self.k;
        self.i = java_round((3 * i - j) as f64 * M_ONESEVENTH);
        self.j = java_round((2 * j + i) as f64 * M_ONESEVENTH);
        self.k = 0;
        self.normalize();
    }

    fn unit_ijk_to_digit(&self) -> i32 {
        if self.i.min(self.j.min(self.k)) < 0 || self.i.max(self.j.max(self.k)) > 1 {
            return INVALID_DIGIT;
        }
        self.i << 2 | self.j << 1 | self.k
    }
}

fn digit_rotate60cw(d: i32) -> i32 {
    match d {
        K_AXES_DIGIT => JK_AXES_DIGIT,
        JK_AXES_DIGIT => J_AXES_DIGIT,
        J_AXES_DIGIT => IJ_AXES_DIGIT,
        IJ_AXES_DIGIT => I_AXES_DIGIT,
        I_AXES_DIGIT => IK_AXES_DIGIT,
        IK_AXES_DIGIT => K_AXES_DIGIT,
        d => d,
    }
}

fn digit_rotate60ccw(d: i32) -> i32 {
    match d {
        K_AXES_DIGIT => IK_AXES_DIGIT,
        IK_AXES_DIGIT => I_AXES_DIGIT,
        I_AXES_DIGIT => IJ_AXES_DIGIT,
        IJ_AXES_DIGIT => J_AXES_DIGIT,
        J_AXES_DIGIT => JK_AXES_DIGIT,
        JK_AXES_DIGIT => K_AXES_DIGIT,
        d => d,
    }
}

// ---------------------------------------------------------------------------
// BaseCells

fn is_base_cell_pentagon(bc: i32) -> bool {
    (0..NUM_BASE_CELLS).contains(&bc) && BASE_CELL_DATA[bc as usize].4
}

fn base_face_ijk(bc: i32) -> FaceIJK {
    let d = BASE_CELL_DATA[bc as usize];
    FaceIJK { face: d.0 as usize, coord: CoordIJK { i: d.1, j: d.2, k: d.3 } }
}

fn face_ijk_base_cell(face: usize, c: &CoordIJK) -> (i32, i32) {
    FACE_IJK_BASE_CELLS[((face * 3 + c.i as usize) * 3 + c.j as usize) * 3 + c.k as usize]
}

fn base_cell_is_cw_offset(bc: i32, test_face: i32) -> bool {
    let o = BASE_CELL_DATA[bc as usize].5;
    o[0] == test_face || o[1] == test_face
}

// ---------------------------------------------------------------------------
// FaceIJK

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Overage {
    None,
    FaceEdge,
    NewFace,
}

const IJ: i32 = 1;
const KI: i32 = 2;
const JK: i32 = 3;

const MAX_DIM_BY_CII_RES: [i32; 17] = [2, -1, 14, -1, 98, -1, 686, -1, 4802, -1, 33614, -1, 235298, -1, 1647086, -1, 11529602];
const UNIT_SCALE_BY_CII_RES: [i32; 17] = [1, -1, 7, -1, 49, -1, 343, -1, 2401, -1, 16807, -1, 117649, -1, 823543, -1, 5764801];
const MAX_FACE_COORD: i32 = 2;
const VERTEX_CLASSIII: [[i32; 3]; 6] = [[5, 4, 0], [1, 5, 0], [0, 5, 4], [0, 1, 5], [4, 0, 5], [5, 0, 1]];
const VERTEX_CLASSII: [[i32; 3]; 6] = [[2, 1, 0], [1, 2, 0], [0, 2, 1], [0, 1, 2], [1, 0, 2], [2, 0, 1]];

/// `FaceIJK.maxDimByCIIVec2d[res][n]`.
fn max_dim_vec2d(res: i32, n: usize) -> Vec2d {
    let m = MAX_DIM_BY_CII_RES[res as usize] as f64;
    match n {
        0 => Vec2d { x: 3.0 * m, y: 0.0 },
        1 => Vec2d { x: -1.5 * m, y: 3.0 * M_SQRT3_2 * m },
        _ => Vec2d { x: -1.5 * m, y: -3.0 * M_SQRT3_2 * m },
    }
}

#[derive(Clone, Copy, Debug)]
struct FaceIJK {
    face: usize,
    coord: CoordIJK,
}

impl FaceIJK {
    fn adjust_overage_class_ii(&mut self, res: i32, pent_leading4: bool, substrate: bool) -> Overage {
        let mut overage = Overage::None;
        let mut max_dim = MAX_DIM_BY_CII_RES[res as usize];
        if substrate {
            max_dim *= 3;
        }
        let sum = self.coord.i + self.coord.j + self.coord.k;
        if substrate && sum == max_dim {
            overage = Overage::FaceEdge;
        } else if sum > max_dim {
            overage = Overage::NewFace;
            let orient = if self.coord.k > 0 {
                if self.coord.j > 0 {
                    FACE_NEIGHBORS[self.face][JK as usize]
                } else {
                    if pent_leading4 {
                        self.coord.sub(max_dim, 0, 0);
                        self.coord.rotate60cw();
                        self.coord.add(max_dim, 0, 0);
                    }
                    FACE_NEIGHBORS[self.face][KI as usize]
                }
            } else {
                FACE_NEIGHBORS[self.face][IJ as usize]
            };
            self.face = orient.0 as usize;
            for _ in 0..orient.4 {
                self.coord.rotate60ccw();
            }
            let mut unit_scale = UNIT_SCALE_BY_CII_RES[res as usize];
            if substrate {
                unit_scale *= 3;
            }
            self.coord.add(orient.1 * unit_scale, orient.2 * unit_scale, orient.3 * unit_scale);
            self.coord.normalize();
            if substrate && self.coord.i + self.coord.j + self.coord.k == max_dim {
                overage = Overage::FaceEdge;
            }
        }
        overage
    }

    fn to_geo(&self, res: i32) -> LatLng {
        self.coord.to_geo(self.face, res, false)
    }

    fn adjust_pent_vert_overage(&mut self, res: i32) {
        while self.adjust_overage_class_ii(res, false, true) == Overage::NewFace {}
    }

    fn pent_to_cell_boundary(&mut self, res: i32) -> Vec<LatLng> {
        self.coord.down_ap3();
        self.coord.down_ap3r();
        let adj_res = adjust_res(&mut self.coord, res);
        if is_class_iii(res) {
            self.pent_boundary_class_iii(adj_res)
        } else {
            self.pent_boundary_class_ii(adj_res)
        }
    }

    fn pent_boundary_class_ii(&self, adj_res: i32) -> Vec<LatLng> {
        let mut points = Vec::with_capacity(NUM_PENT_VERTS);
        let mut fijk = FaceIJK { face: self.face, coord: CoordIJK { i: 0, j: 0, k: 0 } };
        for v in VERTEX_CLASSII.iter().take(NUM_PENT_VERTS) {
            fijk.coord.reset(v[0] + self.coord.i, v[1] + self.coord.j, v[2] + self.coord.k);
            fijk.coord.normalize();
            fijk.face = self.face;
            fijk.adjust_pent_vert_overage(adj_res);
            points.push(fijk.coord.to_geo(fijk.face, adj_res, true));
        }
        points
    }

    fn pent_boundary_class_iii(&self, adj_res: i32) -> Vec<LatLng> {
        let mut points = Vec::with_capacity(10);
        let mut fijk = FaceIJK { face: self.face, coord: CoordIJK { i: 0, j: 0, k: 0 } };
        let mut last_coord = CoordIJK { i: 0, j: 0, k: 0 };
        let mut last_face = self.face;
        for vert in 0..NUM_PENT_VERTS + 1 {
            let v = vert % NUM_PENT_VERTS;
            let vc = VERTEX_CLASSIII[v];
            fijk.coord.reset(vc[0] + self.coord.i, vc[1] + self.coord.j, vc[2] + self.coord.k);
            fijk.coord.normalize();
            fijk.face = self.face;
            fijk.adjust_pent_vert_overage(adj_res);
            if vert > 0 {
                let orig2d0 = last_coord.to_hex2d();
                let current_to_last_dir = ADJACENT_FACE_DIR[fijk.face][last_face];
                let orient = FACE_NEIGHBORS[fijk.face][current_to_last_dir as usize];
                last_coord.reset(fijk.coord.i, fijk.coord.j, fijk.coord.k);
                for _ in 0..orient.4 {
                    last_coord.rotate60ccw();
                }
                let unit_scale = UNIT_SCALE_BY_CII_RES[adj_res as usize] * 3;
                last_coord.add(orient.1 * unit_scale, orient.2 * unit_scale, orient.3 * unit_scale);
                last_coord.normalize();
                let orig2d1 = last_coord.to_hex2d();
                if let Some(inter) = find_intersection_point(orig2d0, orig2d1, adj_res, ADJACENT_FACE_DIR[orient.0 as usize][fijk.face]) {
                    points.push(inter.hex2d_to_geo(orient.0 as usize, adj_res, true));
                }
            }
            if vert < NUM_PENT_VERTS {
                points.push(fijk.coord.to_geo(fijk.face, adj_res, true));
            }
            last_face = fijk.face;
            last_coord.reset(fijk.coord.i, fijk.coord.j, fijk.coord.k);
        }
        points
    }

    fn to_cell_boundary(&mut self, res: i32) -> Vec<LatLng> {
        self.coord.down_ap3();
        self.coord.down_ap3r();
        let adj_res = adjust_res(&mut self.coord, res);
        if is_class_iii(res) {
            self.boundary_class_iii(adj_res)
        } else {
            self.boundary_class_ii(adj_res)
        }
    }

    fn boundary_class_ii(&self, adj_res: i32) -> Vec<LatLng> {
        let mut points = Vec::with_capacity(NUM_HEX_VERTS);
        let mut fijk = FaceIJK { face: self.face, coord: CoordIJK { i: 0, j: 0, k: 0 } };
        for v in VERTEX_CLASSII.iter() {
            fijk.coord.reset(v[0] + self.coord.i, v[1] + self.coord.j, v[2] + self.coord.k);
            fijk.coord.normalize();
            fijk.face = self.face;
            fijk.adjust_overage_class_ii(adj_res, false, true);
            points.push(fijk.coord.to_geo(fijk.face, adj_res, true));
        }
        points
    }

    fn boundary_class_iii(&self, adj_res: i32) -> Vec<LatLng> {
        let mut points = Vec::with_capacity(10);
        let mut fijk = FaceIJK { face: self.face, coord: CoordIJK { i: 0, j: 0, k: 0 } };
        let mut last_face: i64 = -1;
        let mut last_overage = Overage::None;
        for vert in 0..NUM_HEX_VERTS + 1 {
            let v = vert % NUM_HEX_VERTS;
            let vc = VERTEX_CLASSIII[v];
            fijk.coord.reset(vc[0] + self.coord.i, vc[1] + self.coord.j, vc[2] + self.coord.k);
            fijk.coord.normalize();
            fijk.face = self.face;
            let overage = fijk.adjust_overage_class_ii(adj_res, false, true);
            if vert > 0 && fijk.face as i64 != last_face && last_overage != Overage::FaceEdge {
                let last_v = (v + 5) % NUM_HEX_VERTS;
                let orig2d0 = self.orig(VERTEX_CLASSIII[last_v]);
                let orig2d1 = self.orig(VERTEX_CLASSIII[v]);
                let face2 = if last_face == self.face as i64 { fijk.face } else { last_face as usize };
                if let Some(inter) = find_intersection_point(orig2d0, orig2d1, adj_res, ADJACENT_FACE_DIR[self.face][face2]) {
                    points.push(inter.hex2d_to_geo(self.face, adj_res, true));
                }
            }
            if vert < NUM_HEX_VERTS {
                points.push(fijk.coord.to_geo(fijk.face, adj_res, true));
            }
            last_face = fijk.face as i64;
            last_overage = overage;
        }
        points
    }

    fn orig(&self, v: [i32; 3]) -> Vec2d {
        let mut s = CoordIJK { i: v[0] + self.coord.i, j: v[1] + self.coord.j, k: v[2] + self.coord.k };
        s.normalize();
        s.to_hex2d()
    }
}

fn adjust_res(coord: &mut CoordIJK, res: i32) -> i32 {
    if is_class_iii(res) {
        coord.down_ap7r();
        res + 1
    } else {
        res
    }
}

fn find_intersection_point(orig2d0: Vec2d, orig2d1: Vec2d, adj_res: i32, face_dir: i32) -> Option<Vec2d> {
    let (e0, e1) = match face_dir {
        IJ => (max_dim_vec2d(adj_res, 0), max_dim_vec2d(adj_res, 1)),
        JK => (max_dim_vec2d(adj_res, 1), max_dim_vec2d(adj_res, 2)),
        _ => (max_dim_vec2d(adj_res, 2), max_dim_vec2d(adj_res, 0)),
    };
    let inter = v2d_intersect(orig2d0, orig2d1, e0, e1);
    if orig2d0.numerically_identical(&inter) || orig2d1.numerically_identical(&inter) {
        None
    } else {
        Some(inter)
    }
}

/// `FaceIJK.faceIjkToH3`. Out-of-range coordinates (ES throws) give 0.
fn face_ijk_to_h3(res: i32, face: usize, mut coord: CoordIJK) -> u64 {
    let mut h = H3_INIT;
    h = set_mode(h, H3_CELL_MODE);
    h = set_resolution(h, res);
    if res == 0 {
        if coord.i > MAX_FACE_COORD || coord.j > MAX_FACE_COORD || coord.k > MAX_FACE_COORD {
            return 0;
        }
        return set_base_cell(h, face_ijk_base_cell(face, &coord).0);
    }
    let mut scratch = CoordIJK { i: 0, j: 0, k: 0 };
    for r in (1..=res).rev() {
        let (li, lj, lk) = (coord.i, coord.j, coord.k);
        if is_class_iii(r) {
            coord.up_ap7();
            scratch = coord;
            scratch.down_ap7();
        } else {
            coord.up_ap7r();
            scratch = coord;
            scratch.down_ap7r();
        }
        scratch.reset(li - scratch.i, lj - scratch.j, lk - scratch.k);
        scratch.normalize();
        h = set_index_digit(h, r, scratch.unit_ijk_to_digit());
    }
    if coord.i > MAX_FACE_COORD || coord.j > MAX_FACE_COORD || coord.k > MAX_FACE_COORD {
        return 0;
    }
    let (base_cell, num_rots) = face_ijk_base_cell(face, &coord);
    h = set_base_cell(h, base_cell);
    if is_base_cell_pentagon(base_cell) {
        if leading_non_zero_digit(h) == K_AXES_DIGIT {
            if base_cell_is_cw_offset(base_cell, face as i32) {
                h = rotate60cw(h);
            } else {
                h = rotate60ccw(h);
            }
        }
        for _ in 0..num_rots {
            h = rotate_pent60ccw(h);
        }
    } else {
        for _ in 0..num_rots {
            h = rotate60ccw(h);
        }
    }
    h
}

// ---------------------------------------------------------------------------
// H3Index

const H3_INIT: u64 = 35184372088831;
const H3_MODE_OFFSET: u32 = 59;
const H3_MODE_MASK: u64 = 15 << H3_MODE_OFFSET;
const H3_BC_OFFSET: u32 = 45;
const H3_BC_MASK: u64 = 127 << H3_BC_OFFSET;
const H3_RES_OFFSET: u32 = 52;
const H3_RES_MASK: u64 = 15 << H3_RES_OFFSET;
const H3_HIGH_BIT_MASK: u64 = 1 << 63;
const H3_RESERVED_OFFSET: u32 = 56;
const H3_RESERVED_MASK: u64 = 7 << H3_RESERVED_OFFSET;
const H3_PER_DIGIT_OFFSET: i32 = 3;
const H3_DIGIT_MASK: u64 = 7;

#[inline]
fn base_cell_of(h: u64) -> i32 {
    ((h & H3_BC_MASK) >> H3_BC_OFFSET) as i32
}

#[inline]
fn set_mode(h: u64, mode: u64) -> u64 {
    (h & !H3_MODE_MASK) | (mode << H3_MODE_OFFSET)
}

#[inline]
fn set_base_cell(h: u64, bc: i32) -> u64 {
    (h & !H3_BC_MASK) | ((bc as u64) << H3_BC_OFFSET)
}

#[inline]
fn mode_of(h: u64) -> u64 {
    (h & H3_MODE_MASK) >> H3_MODE_OFFSET
}

#[inline]
fn set_resolution(h: u64, res: i32) -> u64 {
    (h & !H3_RES_MASK) | ((res as u64) << H3_RES_OFFSET)
}

#[inline]
fn index_digit(h: u64, res: i32) -> i32 {
    ((h >> ((MAX_H3_RES - res) * H3_PER_DIGIT_OFFSET)) & H3_DIGIT_MASK) as i32
}

#[inline]
fn set_index_digit(h: u64, res: i32, digit: i32) -> u64 {
    let x = (MAX_H3_RES - res) * H3_PER_DIGIT_OFFSET;
    (h & !(H3_DIGIT_MASK << x)) | ((digit as u64) << x)
}

fn leading_non_zero_digit(h: u64) -> i32 {
    for r in 1..=resolution(h) {
        let d = index_digit(h, r);
        if d != CENTER_DIGIT {
            return d;
        }
    }
    CENTER_DIGIT
}

fn rotate60cw(mut h: u64) -> u64 {
    for r in 1..=resolution(h) {
        h = set_index_digit(h, r, digit_rotate60cw(index_digit(h, r)));
    }
    h
}

fn rotate60ccw(mut h: u64) -> u64 {
    for r in 1..=resolution(h) {
        h = set_index_digit(h, r, digit_rotate60ccw(index_digit(h, r)));
    }
    h
}

fn rotate_pent60ccw(mut h: u64) -> u64 {
    let mut found = false;
    for r in 1..=resolution(h) {
        h = set_index_digit(h, r, digit_rotate60ccw(index_digit(h, r)));
        if !found && index_digit(h, r) != 0 {
            found = true;
            if leading_non_zero_digit(h) == K_AXES_DIGIT {
                h = rotate60ccw(h);
            }
        }
    }
    h
}

/// `H3Index.h3ToFaceIjk`.
fn h3_to_face_ijk(mut h: u64) -> FaceIJK {
    let base_cell = base_cell_of(h);
    if is_base_cell_pentagon(base_cell) && leading_non_zero_digit(h) == 5 {
        h = rotate60cw(h);
    }
    let mut fijk = base_face_ijk(base_cell.clamp(0, NUM_BASE_CELLS - 1));
    if !face_ijk_with_initialized(h, &mut fijk) {
        return fijk;
    }
    let orig = fijk.coord;
    let mut res = resolution(h);
    if is_class_iii(res) {
        fijk.coord.down_ap7r();
        res += 1;
    }
    let pent_leading4 = is_base_cell_pentagon(base_cell) && leading_non_zero_digit(h) == 4;
    if fijk.adjust_overage_class_ii(res, pent_leading4, false) != Overage::None {
        if is_base_cell_pentagon(base_cell) {
            while fijk.adjust_overage_class_ii(res, false, false) != Overage::None {}
        }
        if res != resolution(h) {
            fijk.coord.up_ap7r();
        }
    } else if res != resolution(h) {
        fijk.coord = orig;
    }
    fijk
}

fn face_ijk_with_initialized(h: u64, fijk: &mut FaceIJK) -> bool {
    let res = resolution(h);
    let possible_overage = is_base_cell_pentagon(base_cell_of(h)) || (res != 0 && (fijk.coord.i != 0 || fijk.coord.j != 0 || fijk.coord.k != 0));
    for r in 1..=res {
        if is_class_iii(r) {
            fijk.coord.down_ap7();
        } else {
            fijk.coord.down_ap7r();
        }
        fijk.coord.neighbor(index_digit(h, r));
    }
    possible_overage
}

// ---------------------------------------------------------------------------
// HexRing

const INVALID_BASE_CELL: i32 = 127;
const RING_DIRECTIONS: [i32; 6] = [J_AXES_DIGIT, JK_AXES_DIGIT, K_AXES_DIGIT, IK_AXES_DIGIT, I_AXES_DIGIT, IJ_AXES_DIGIT];

/// `HexRing.h3NeighborInDirection`: `None` where ES returns -1 (the
/// deleted direction of a pentagon) or throws.
pub fn neighbor_in_direction(origin: u64, dir: i32) -> Option<u64> {
    let mut current = origin;
    let mut dir = dir;
    let mut new_rotations = 0;
    let old_base_cell = base_cell_of(current);
    if !(0..NUM_BASE_CELLS).contains(&old_base_cell) {
        return None;
    }
    let old_leading_digit = leading_non_zero_digit(current);
    let mut r = resolution(current) - 1;
    loop {
        if r == -1 {
            current = set_base_cell(current, BASE_CELL_NEIGHBORS[old_base_cell as usize][dir as usize]);
            new_rotations = BASE_CELL_NEIGHBOR_60CCW_ROTS[old_base_cell as usize][dir as usize];
            if base_cell_of(current) == INVALID_BASE_CELL {
                current = set_base_cell(current, BASE_CELL_NEIGHBORS[old_base_cell as usize][IK_AXES_DIGIT as usize]);
                new_rotations = BASE_CELL_NEIGHBOR_60CCW_ROTS[old_base_cell as usize][IK_AXES_DIGIT as usize];
                current = rotate60ccw(current);
            }
            break;
        }
        let old_digit = index_digit(current, r + 1);
        if old_digit == INVALID_DIGIT {
            return None;
        }
        let next_dir;
        if is_class_iii(r + 1) {
            current = set_index_digit(current, r + 1, NEW_DIGIT_II[old_digit as usize][dir as usize]);
            next_dir = NEW_ADJUSTMENT_II[old_digit as usize][dir as usize];
        } else {
            current = set_index_digit(current, r + 1, NEW_DIGIT_III[old_digit as usize][dir as usize]);
            next_dir = NEW_ADJUSTMENT_III[old_digit as usize][dir as usize];
        }
        if next_dir != CENTER_DIGIT {
            dir = next_dir;
            r -= 1;
        } else {
            break;
        }
    }
    let new_base_cell = base_cell_of(current);
    if is_base_cell_pentagon(new_base_cell) {
        if leading_non_zero_digit(current) == K_AXES_DIGIT {
            if old_base_cell != new_base_cell {
                if base_cell_is_cw_offset(new_base_cell, base_face_ijk(old_base_cell).face as i32) {
                    current = rotate60cw(current);
                } else {
                    current = rotate60ccw(current);
                }
            } else if old_leading_digit == CENTER_DIGIT {
                return None;
            } else if old_leading_digit == JK_AXES_DIGIT {
                current = rotate60ccw(current);
            } else if old_leading_digit == IK_AXES_DIGIT {
                current = rotate60cw(current);
            } else {
                return None;
            }
        }
        for _ in 0..new_rotations {
            current = rotate_pent60ccw(current);
        }
    } else {
        for _ in 0..new_rotations {
            current = rotate60ccw(current);
        }
    }
    Some(current)
}

/// `HexRing.areNeighbours` (`None` where ES throws).
pub fn are_neighbor_cells(origin: u64, destination: u64) -> Option<bool> {
    if mode_of(origin) != H3_CELL_MODE || mode_of(destination) != H3_CELL_MODE {
        return None;
    }
    if origin == destination {
        return Some(false);
    }
    let res = resolution(origin);
    if res != resolution(destination) {
        return Some(false);
    }
    if res > 1 {
        let (op, dp) = (parent(origin)?, parent(destination)?);
        if op == dp {
            let od = index_digit(origin, res);
            let dd = index_digit(destination, res);
            if od == CENTER_DIGIT || dd == CENTER_DIGIT {
                return Some(true);
            }
            if od >= INVALID_DIGIT {
                return None;
            }
            if (od == K_AXES_DIGIT || dd == K_AXES_DIGIT) && is_pentagon(op) {
                return None;
            }
            if NEIGHBORSETCLOCKWISE[od as usize] == dd || NEIGHBORSETCOUNTERCLOCKWISE[od as usize] == dd {
                return Some(true);
            }
        }
    }
    for d in RING_DIRECTIONS {
        if neighbor_in_direction(origin, d) == Some(destination) {
            return Some(true);
        }
    }
    Some(false)
}

// ---------------------------------------------------------------------------
// H3 (the public API)

/// `H3.h3ToString`: lower-case hex.
pub fn to_string(h: u64) -> String {
    format!("{h:x}")
}

/// `H3.stringToH3` (`Long.parseUnsignedLong(s, 16)`).
pub fn from_string(s: &str) -> Option<u64> {
    if s.is_empty() || s.starts_with('-') {
        return None;
    }
    u64::from_str_radix(s.strip_prefix('+').unwrap_or(s), 16).ok()
}

/// `H3.getResolution`.
#[inline]
pub fn resolution(h: u64) -> i32 {
    ((h & H3_RES_MASK) >> H3_RES_OFFSET) as i32
}

/// `H3.isPentagon`.
pub fn is_pentagon(h: u64) -> bool {
    is_base_cell_pentagon(base_cell_of(h)) && leading_non_zero_digit(h) == 0
}

/// `H3.h3IsValid`.
pub fn is_valid(h: u64) -> bool {
    if h & H3_HIGH_BIT_MASK != 0 || mode_of(h) != H3_CELL_MODE || (h & H3_RESERVED_MASK) >> H3_RESERVED_OFFSET != 0 {
        return false;
    }
    let base_cell = base_cell_of(h);
    if !(0..NUM_BASE_CELLS).contains(&base_cell) {
        return false;
    }
    let res = resolution(h);
    let mut found = false;
    for r in 1..=res {
        let d = index_digit(h, r);
        if !found && d != CENTER_DIGIT {
            found = true;
            if is_base_cell_pentagon(base_cell) && d == K_AXES_DIGIT {
                return false;
            }
        }
        if !(CENTER_DIGIT..NUM_DIGITS).contains(&d) {
            return false;
        }
    }
    for r in res + 1..=MAX_H3_RES {
        if index_digit(h, r) != INVALID_DIGIT {
            return false;
        }
    }
    true
}

/// `H3.getLongRes0Cells`.
pub fn res0_cells() -> impl Iterator<Item = u64> {
    (0..NUM_BASE_CELLS).map(|bc| set_base_cell(set_mode(H3_INIT, H3_CELL_MODE), bc))
}

/// `H3.h3ToLatLng`: the cell's centre.
pub fn to_lat_lng(h: u64) -> LatLng {
    h3_to_face_ijk(h).to_geo(resolution(h))
}

/// `H3.h3ToGeoBoundary`: the cell's boundary vertices.
pub fn boundary(h: u64) -> Vec<LatLng> {
    let mut fijk = h3_to_face_ijk(h);
    if is_pentagon(h) {
        fijk.pent_to_cell_boundary(resolution(h))
    } else {
        fijk.to_cell_boundary(resolution(h))
    }
}

/// `H3.geoToH3` (degrees; `None` for a resolution out of range).
pub fn geo_to_h3(lat: f64, lng: f64, res: i32) -> Option<u64> {
    if !(0..=MAX_H3_RES).contains(&res) {
        return None;
    }
    Some(vec3_geo_to_h3(res, to_radians(lat), to_radians(lng)))
}

/// `H3.h3ToParent` (`None` for a base cell).
pub fn parent(h: u64) -> Option<u64> {
    let child_res = resolution(h);
    if child_res == 0 {
        return None;
    }
    let p = set_resolution(h, child_res - 1);
    Some(set_index_digit(p, child_res, H3_DIGIT_MASK as i32))
}

/// The ancestor of `h` at resolution `res` (`res` at most `h`'s).
pub fn parent_at(mut h: u64, res: i32) -> Option<u64> {
    while resolution(h) > res {
        h = parent(h)?;
    }
    (resolution(h) == res).then_some(h)
}

/// `H3.h3ToChildrenSize(h3)`.
pub fn children_size(h: u64) -> usize {
    if is_pentagon(h) {
        6
    } else {
        7
    }
}

/// `H3.childPosToH3` (`None` past the last resolution or position).
pub fn child_pos_to_h3(h: u64, child_pos: usize) -> Option<u64> {
    let children_res = resolution(h) + 1;
    if children_res > MAX_H3_RES {
        return None;
    }
    let child = set_resolution(h, children_res);
    if child_pos == 0 {
        return Some(set_index_digit(child, children_res, CENTER_DIGIT));
    }
    let pent = is_pentagon(h);
    if child_pos > if pent { 5 } else { 6 } {
        return None;
    }
    Some(set_index_digit(child, children_res, child_pos as i32 + i32::from(pent)))
}

/// `H3.h3ToChildren`.
pub fn children(h: u64) -> Vec<u64> {
    (0..children_size(h)).filter_map(|i| child_pos_to_h3(h, i)).collect()
}

/// The centre child of `h` at resolution `res` (digits 0 down to it).
pub fn center_child(h: u64, res: i32) -> Option<u64> {
    let mut c = h;
    while resolution(c) < res {
        c = child_pos_to_h3(c, 0)?;
    }
    (resolution(c) == res).then_some(c)
}

const PEN_INTERSECTING_CHILDREN_DIRECTIONS: [i32; 5] = [3, 1, 6, 4, 2];
const HEX_INTERSECTING_CHILDREN_DIRECTIONS: [i32; 6] = [3, 6, 2, 5, 1, 4];

/// `H3.h3ToNotIntersectingChildrenSize`.
pub fn not_intersecting_children_size(h: u64) -> usize {
    if is_pentagon(h) {
        5
    } else {
        6
    }
}

/// `H3.noChildIntersectingPosToH3`.
pub fn no_child_intersecting_pos_to_h3(h: u64, child_pos: usize) -> Option<u64> {
    let children_res = resolution(h) + 1;
    if children_res > MAX_H3_RES {
        return None;
    }
    let pent = is_pentagon(h);
    if child_pos > if pent { 4 } else { 5 } {
        return None;
    }
    let child_h = set_resolution(h, children_res);
    if pent {
        neighbor_in_direction(set_index_digit(child_h, children_res, child_pos as i32 + 2), PEN_INTERSECTING_CHILDREN_DIRECTIONS[child_pos])
    } else {
        neighbor_in_direction(set_index_digit(child_h, children_res, child_pos as i32 + 1), HEX_INTERSECTING_CHILDREN_DIRECTIONS[child_pos])
    }
}

/// `H3.h3ToNoChildrenIntersecting`.
pub fn no_children_intersecting(h: u64) -> Vec<u64> {
    (0..not_intersecting_children_size(h)).filter_map(|i| no_child_intersecting_pos_to_h3(h, i)).collect()
}

/// `H3.hexRingSize`.
pub fn hex_ring_size(h: u64) -> usize {
    if is_pentagon(h) {
        5
    } else {
        6
    }
}

/// `H3.hexRingPosToH3`.
pub fn hex_ring_pos_to_h3(h: u64, ring_pos: usize) -> Option<u64> {
    let pos = if is_pentagon(h) && ring_pos >= 2 { ring_pos + 1 } else { ring_pos };
    if pos > 5 {
        return None;
    }
    neighbor_in_direction(h, RING_DIRECTIONS[pos])
}

/// `H3.hexRing`: the cell's neighbours.
pub fn hex_ring(h: u64) -> Vec<u64> {
    (0..hex_ring_size(h)).filter_map(|i| hex_ring_pos_to_h3(h, i)).collect()
}

/// `H3.northPolarH3`.
pub fn north_polar_h3(res: i32) -> Option<u64> {
    polar(res).map(|p| p.0)
}

/// `H3.southPolarH3`.
pub fn south_polar_h3(res: i32) -> Option<u64> {
    polar(res).map(|p| p.1)
}

fn polar(res: i32) -> Option<(u64, u64)> {
    static P: std::sync::OnceLock<Vec<(u64, u64)>> = std::sync::OnceLock::new();
    let p = P.get_or_init(|| (0..=MAX_H3_RES).map(|r| (geo_to_h3(90.0, 0.0, r).unwrap(), geo_to_h3(-90.0, 0.0, r).unwrap())).collect());
    usize::try_from(res).ok().and_then(|r| p.get(r)).copied()
}

#[cfg(test)]
mod tests;
