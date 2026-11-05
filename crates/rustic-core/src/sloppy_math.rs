//! Port of Lucene 9.12 `SloppyMath` (table-based `cos` / `asin`), which ES
//! uses for `arc` geo distances (`GeoUtils.arcDistance`). Tables are built
//! with fdlibm (`libm`), as Java's `StrictMath`, so results match ES bit for
//! bit.

use std::sync::OnceLock;

const TO_METERS: f64 = 6_371_008.7714;
const ONE_DIV_F2: f64 = 1.0 / 2.0;
const ONE_DIV_F3: f64 = 1.0 / 6.0;
const ONE_DIV_F4: f64 = 1.0 / 24.0;
const SIN_COS_TABS_SIZE: usize = (1 << 11) + 1;
const ASIN_TABS_SIZE: usize = (1 << 13) + 1;

fn bits(b: u64) -> f64 {
    f64::from_bits(b)
}

struct Tabs {
    delta_hi: f64,
    delta_lo: f64,
    indexer: f64,
    max_for_int_modulo: f64,
    sin: Vec<f64>,
    cos: Vec<f64>,
    asin_max: f64,
    asin_delta: f64,
    asin_indexer: f64,
    asin: Vec<f64>,
    d1: Vec<f64>,
    d2: Vec<f64>,
    d3: Vec<f64>,
    d4: Vec<f64>,
}

fn tabs() -> &'static Tabs {
    static T: OnceLock<Tabs> = OnceLock::new();
    T.get_or_init(|| {
        let pio2_hi = bits(0x3FF921FB54400000);
        let pio2_lo = bits(0x3DD0B4611A626331);
        let (twopi_hi, twopi_lo) = (4.0 * pio2_hi, 4.0 * pio2_lo);
        let delta_hi = twopi_hi / (SIN_COS_TABS_SIZE - 1) as f64;
        let delta_lo = twopi_lo / (SIN_COS_TABS_SIZE - 1) as f64;
        let indexer = 1.0 / (delta_hi + delta_lo);
        let max_for_int_modulo = (((i32::MAX >> 9) as f64) / indexer) * 0.99;
        let pi_index = (SIN_COS_TABS_SIZE - 1) / 2;
        let (mut sin, mut cos) = (vec![0.0; SIN_COS_TABS_SIZE], vec![0.0; SIN_COS_TABS_SIZE]);
        for i in 0..SIN_COS_TABS_SIZE {
            let angle = i as f64 * delta_hi + i as f64 * delta_lo;
            let mut s = libm::sin(angle);
            let mut c = libm::cos(angle);
            if i == pi_index || i == 2 * pi_index {
                s = 0.0;
            } else if i == pi_index / 2 || i == 3 * pi_index / 2 {
                c = 0.0;
            }
            sin[i] = s;
            cos[i] = c;
        }
        let asin_max = libm::sin(73.0f64.to_radians());
        let asin_delta = asin_max / (ASIN_TABS_SIZE - 1) as f64;
        let asin_indexer = 1.0 / asin_delta;
        let mut t = (vec![0.0; ASIN_TABS_SIZE], vec![0.0; ASIN_TABS_SIZE], vec![0.0; ASIN_TABS_SIZE], vec![0.0; ASIN_TABS_SIZE], vec![0.0; ASIN_TABS_SIZE]);
        for i in 0..ASIN_TABS_SIZE {
            let x = i as f64 * asin_delta;
            t.0[i] = libm::asin(x);
            let inv = 1.0 / (1.0 - x * x);
            let inv0_5 = libm::sqrt(inv);
            let inv1_5 = inv0_5 * inv;
            let inv2_5 = inv1_5 * inv;
            let inv3_5 = inv2_5 * inv;
            t.1[i] = inv0_5;
            t.2[i] = (x * inv1_5) * ONE_DIV_F2;
            t.3[i] = ((1.0 + 2.0 * x * x) * inv2_5) * ONE_DIV_F3;
            t.4[i] = ((5.0 + 2.0 * x * (2.0 + x * (5.0 - 2.0 * x))) * inv3_5) * ONE_DIV_F4;
        }
        Tabs { delta_hi, delta_lo, indexer, max_for_int_modulo, sin, cos, asin_max, asin_delta, asin_indexer, asin: t.0, d1: t.1, d2: t.2, d3: t.3, d4: t.4 }
    })
}

/// `SloppyMath.cos`.
pub fn cos(a: f64) -> f64 {
    let t = tabs();
    let a = a.abs();
    if a > t.max_for_int_modulo {
        return a.cos();
    }
    let mut index = (a * t.indexer + 0.5) as i32;
    let delta = (a - index as f64 * t.delta_hi) - index as f64 * t.delta_lo;
    index &= (SIN_COS_TABS_SIZE - 2) as i32;
    let (ic, is) = (t.cos[index as usize], t.sin[index as usize]);
    ic + delta * (-is + delta * (-ic * ONE_DIV_F2 + delta * (is * ONE_DIV_F3 + delta * ic * ONE_DIV_F4)))
}

/// `SloppyMath.asin`.
pub fn asin(a: f64) -> f64 {
    let t = tabs();
    let (a, neg) = if a < 0.0 { (-a, true) } else { (a, false) };
    let r = if a <= t.asin_max {
        let index = (a * t.asin_indexer + 0.5) as usize;
        let delta = a - index as f64 * t.asin_delta;
        t.asin[index] + delta * (t.d1[index] + delta * (t.d2[index] + delta * (t.d3[index] + delta * t.d4[index])))
    } else if a < 1.0 {
        let tt = (1.0 - a) * 0.5;
        let p = tt
            * (bits(0x3fc5555555555555)
                + tt * (bits(0xbfd4d61203eb6f7d) + tt * (bits(0x3fc9c1550e884455) + tt * (bits(0xbfa48228b5688f3b) + tt * (bits(0x3f49efe07501b288) + tt * bits(0x3f023de10dfdf709))))));
        let q = 1.0 + tt * (bits(0xc0033a271c8a2d4b) + tt * (bits(0x40002ae59c598ac8) + tt * (bits(0xbfe6066c1b8d0159) + tt * bits(0x3fb3b8c5b12e9282))));
        let s = tt.sqrt();
        let z = s + s * (p / q);
        bits(0x3FF921FB54442D18) - ((z + z) - bits(0x3C91A62633145C07))
    } else if a == 1.0 {
        std::f64::consts::PI / 2.0
    } else {
        f64::NAN
    };
    if neg { -r } else { r }
}

/// `SloppyMath.haversinSortKey`.
pub fn haversin_sort_key(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let x1 = lat1.to_radians();
    let x2 = lat2.to_radians();
    let h1 = 1.0 - cos(x1 - x2);
    let h2 = 1.0 - cos((lon1 - lon2).to_radians());
    let h = h1 + cos(x1) * cos(x2) * h2;
    f64::from_bits(h.to_bits() & 0xFFFF_FFFF_FFFF_FFF8)
}

/// `SloppyMath.haversinMeters` (ES `GeoUtils.arcDistance`).
pub fn haversin_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let k = haversin_sort_key(lat1, lon1, lat2, lon2);
    TO_METERS * 2.0 * asin(1f64.min((k * 0.5).sqrt()))
}

/// ES `org.elasticsearch.core.FastMath.log` (`ESSloppyMath.log`): table
/// based, as ES uses for spherical mercator.
pub fn es_log(value: f64) -> f64 {
    const LOG_BITS: u32 = 12;
    const LOG_TAB_SIZE: usize = 1 << LOG_BITS;
    struct LogTabs {
        log_x_log: Vec<f64>,
        log_x_inv: Vec<f64>,
        log2: f64,
    }
    static T: OnceLock<LogTabs> = OnceLock::new();
    let t = T.get_or_init(|| {
        let mut log_x_log = vec![0.0; LOG_TAB_SIZE];
        let mut log_x_inv = vec![0.0; LOG_TAB_SIZE];
        for i in 0..LOG_TAB_SIZE {
            let x = 1.0 + i as f64 * (1.0 / LOG_TAB_SIZE as f64);
            log_x_log[i] = libm::log(x);
            log_x_inv[i] = 1.0 / x;
        }
        LogTabs { log_x_log, log_x_inv, log2: libm::log(2.0) }
    });
    if value > 0.0 {
        if value == f64::INFINITY {
            return f64::INFINITY;
        }
        let mut value = value;
        let h;
        if value > 0.95 {
            if value < 1.14 {
                let z = (value - 1.0) / (value + 1.0);
                let z2 = z * z;
                return z * (2.0 + z2 * ((2.0 / 3.0) + z2 * ((2.0 / 5.0) + z2 * ((2.0 / 7.0) + z2 * ((2.0 / 9.0) + z2 * (2.0 / 11.0))))));
            }
            h = 0.0;
        } else if value < f64::from_bits(0x0010000000000000) {
            value *= f64::from_bits(0x4330000000000000);
            h = -52.0 * t.log2;
        } else {
            h = 0.0;
        }
        let hi = (value.to_bits() >> 32) as i32;
        let exp = (hi >> 20) - 1023;
        let x_index = (((hi << 12) as u32) >> (32 - LOG_BITS)) as usize;
        let mut z = (value * 2f64.powi(-exp)) * t.log_x_inv[x_index] - 1.0;
        z *= 1.0 - z * ((1.0 / 2.0) - z * (1.0 / 3.0));
        h + exp as f64 * t.log2 + (t.log_x_log[x_index] + z)
    } else if value == 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::NAN
    }
}
