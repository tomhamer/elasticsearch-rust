//! `geo_grid` support for `geohex` (H3) cells, on [`crate::h3`] (the port
//! of ES's own H3 library): parent, children, non-intersecting children
//! (`h3ToNoChildrenIntersecting`) and boundary are ES's.

use crate::error::EsError;
use crate::h3;

type P = (f64, f64);

#[derive(Debug)]
pub struct Hex {
    pub precision: u8,
    /// Open ring of (lon, lat) vertices.
    pub boundary: Vec<P>,
    pub parent: Option<String>,
    pub children: Vec<String>,
    pub non_children: Vec<String>,
}

fn ia(reason: impl Into<String>) -> EsError {
    EsError::illegal_argument(reason)
}

fn invalid(cause: EsError) -> EsError {
    ia("invalid tile definition").caused_by(cause)
}

fn poly(c: u64) -> Vec<P> {
    h3::boundary(c).iter().map(|l| (l.lon_deg(), l.lat_deg())).collect()
}

/// Parse an H3 index string the way ES does (`Long.parseUnsignedLong(s, 16)`).
fn parse_index(id: &str) -> Result<u64, EsError> {
    let raw = u64::from_str_radix(id.strip_prefix('+').unwrap_or(id), 16)
        .map_err(|_| invalid(EsError::new(500, "number_format_exception", format!("For input string: \"{id}\" under radix 16"))))?;
    if h3::is_valid(raw) {
        Ok(raw)
    } else {
        Err(invalid(ia("")))
    }
}

/// The cell index of an address, as [`hex_info`] parses it.
pub fn parse_cell_id(id: &str) -> Result<u64, EsError> {
    parse_index(id)
}

pub fn hex_info(id: &str) -> Result<Hex, EsError> {
    let cell = parse_index(id)?;
    let res = h3::resolution(cell);
    if res == h3::MAX_H3_RES {
        return Err(invalid(ia(format!("Invalid child resolution [{}]", h3::MAX_H3_RES))));
    }
    Ok(Hex {
        precision: res as u8,
        boundary: poly(cell),
        parent: h3::parent(cell).map(h3::to_string),
        children: h3::children(cell).into_iter().map(h3::to_string).collect(),
        non_children: h3::no_children_intersecting(cell).into_iter().map(h3::to_string).collect(),
    })
}



#[cfg(test)]
mod tests {
    use super::*;

    /// The geo_grid processor's geohex shapes equal ES 9.5.5's
    /// (`_ingest/pipeline/_simulate`, WKT) for cells near the dateline and
    /// the poles.
    #[test]
    fn geohex_shapes_match_es() {
        use std::io::Read;
        let mut text = String::new();
        flate2::read::GzDecoder::new(&include_bytes!("../h3/fixtures/es_ingest_hex.json.gz")[..]).read_to_string(&mut text).unwrap();
        let want: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text).unwrap();
        let p = crate::ingest::processors::test_geohex_wkt;
        let mut bad = Vec::new();
        for (cell, w) in &want {
            let got = p(cell);
            if got.as_deref() != w.as_str() {
                bad.push(format!("{cell}:\n ours {got:?}\n ES   {w}"));
            }
        }
        assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), want.len(), bad[..bad.len().min(6)].join("\n"));
    }

    #[test]
    fn es_sample_cell() {
        let h = hex_info("8a2a1072b59ffff").unwrap();
        assert_eq!(h.precision, 10);
        assert_eq!(h.parent.as_deref(), Some("892a1072b5bffff"));
        assert_eq!(h.children.len(), 7);
        assert_eq!(h.children[0], "8b2a1072b598fff");
        assert_eq!(h.non_children, ["8b2a1072b4a6fff", "8b2a1070c96dfff", "8b2a1072b4b4fff", "8b2a1072b583fff", "8b2a1072b58afff", "8b2a1072b591fff"]);
        assert_eq!(h.boundary.len(), 6);
        assert!((h.boundary[0].0 - -74.04415179044008).abs() < 1e-6);
    }

    #[test]
    fn edge_cells() {
        assert!(hex_info("8001fffffffffff").unwrap().parent.is_none());
        assert_eq!(hex_info("8009fffffffffff").unwrap().children.len(), 6); // pentagon
        assert_eq!(hex_info("bogus").unwrap_err().caused_by.unwrap().reason, "For input string: \"bogus\" under radix 16");
        assert_eq!(hex_info("8f2a1072b59a000").unwrap_err().caused_by.unwrap().reason, "Invalid child resolution [15]");
    }
}
