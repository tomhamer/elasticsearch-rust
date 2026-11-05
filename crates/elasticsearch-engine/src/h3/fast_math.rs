//! ES 9.5.5's `org.elasticsearch.h3.FastMath` (a Jafama port): table based
//! `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, the same tables
//! (built from `StrictMath`, i.e. fdlibm: here the `libm` crate's port of
//! it, with the few entries it rounds differently set to ES's) and the same
//! arithmetic in the same order, so every result is ES's
//! to the bit.

use std::sync::OnceLock;

const ONE_DIV_F2: f64 = 1.0 / 2.0;
const ONE_DIV_F3: f64 = 1.0 / 6.0;
const ONE_DIV_F4: f64 = 1.0 / 24.0;

const TWO_POW_66: f64 = f64::from_bits(0x4410000000000000);

const M_HALF_PI: f64 = std::f64::consts::PI * 0.5;
const M_QUARTER_PI: f64 = std::f64::consts::PI * 0.25;
const M_2PI: f64 = 2.0 * std::f64::consts::PI;

const INVPIO2: f64 = f64::from_bits(0x3FE45F306DC9C883);
const PIO2_HI: f64 = f64::from_bits(0x3FF921FB54400000);
const PIO2_LO: f64 = f64::from_bits(0x3DD0B4611A626331);
const INVTWOPI: f64 = INVPIO2 / 4.0;
const TWOPI_HI: f64 = 4.0 * PIO2_HI;
const TWOPI_LO: f64 = 4.0 * PIO2_LO;

const SIN_COS_TABS_SIZE: usize = (1 << 11) + 1;
const SIN_COS_DELTA_HI: f64 = TWOPI_HI / (SIN_COS_TABS_SIZE - 1) as f64;
const SIN_COS_DELTA_LO: f64 = TWOPI_LO / (SIN_COS_TABS_SIZE - 1) as f64;
const SIN_COS_INDEXER: f64 = 1.0 / (SIN_COS_DELTA_HI + SIN_COS_DELTA_LO);
const SIN_COS_MAX_VALUE_FOR_INT_MODULO: f64 = ((i32::MAX >> 9) as f64 / SIN_COS_INDEXER) * 0.99;

const TAN_VIRTUAL_TABS_SIZE: usize = (1 << 12) + 1;
const TAN_DELTA_HI: f64 = PIO2_HI / (TAN_VIRTUAL_TABS_SIZE - 1) as f64;
const TAN_DELTA_LO: f64 = PIO2_LO / (TAN_VIRTUAL_TABS_SIZE - 1) as f64;
const TAN_INDEXER: f64 = 1.0 / (TAN_DELTA_HI + TAN_DELTA_LO);
const TAN_MAX_VALUE_FOR_INT_MODULO: f64 = ((i32::MAX >> 9) as f64 / TAN_INDEXER) * 0.99;

const ASIN_TABS_SIZE: usize = (1 << 13) + 1;
const ASIN_POWTABS_POWER: i32 = 84;
const ASIN_POWTABS_SIZE: usize = (1 << 12) + 1;
const ASIN_POWTABS_SIZE_MINUS_ONE: usize = ASIN_POWTABS_SIZE - 1;

const ASIN_PIO2_HI: f64 = f64::from_bits(0x3FF921FB54442D18);
const ASIN_PIO2_LO: f64 = f64::from_bits(0x3C91A62633145C07);
const ASIN_PS0: f64 = f64::from_bits(0x3fc5555555555555);
const ASIN_PS1: f64 = f64::from_bits(0xbfd4d61203eb6f7d);
const ASIN_PS2: f64 = f64::from_bits(0x3fc9c1550e884455);
const ASIN_PS3: f64 = f64::from_bits(0xbfa48228b5688f3b);
const ASIN_PS4: f64 = f64::from_bits(0x3f49efe07501b288);
const ASIN_PS5: f64 = f64::from_bits(0x3f023de10dfdf709);
const ASIN_QS1: f64 = f64::from_bits(0xc0033a271c8a2d4b);
const ASIN_QS2: f64 = f64::from_bits(0x40002ae59c598ac8);
const ASIN_QS3: f64 = f64::from_bits(0xbfe6066c1b8d0159);
const ASIN_QS4: f64 = f64::from_bits(0x3fb3b8c5b12e9282);

const ATAN_TABS_SIZE: usize = (1 << 12) + 1;
const ATAN_HI3: f64 = f64::from_bits(0x3ff921fb54442d18);
const ATAN_LO3: f64 = f64::from_bits(0x3c91a62633145c07);
const ATAN_AT0: f64 = f64::from_bits(0x3fd555555555550d);
const ATAN_AT1: f64 = f64::from_bits(0xbfc999999998ebc4);
const ATAN_AT2: f64 = f64::from_bits(0x3fc24924920083ff);
const ATAN_AT3: f64 = f64::from_bits(0xbfbc71c6fe231671);
const ATAN_AT4: f64 = f64::from_bits(0x3fb745cdc54c206e);
const ATAN_AT5: f64 = f64::from_bits(0xbfb3b0f2af749a6d);
const ATAN_AT6: f64 = f64::from_bits(0x3fb10d66a0d03d51);
const ATAN_AT7: f64 = f64::from_bits(0xbfadde2d52defd9a);
const ATAN_AT8: f64 = f64::from_bits(0x3fa97b4b24760deb);
const ATAN_AT9: f64 = f64::from_bits(0xbfa2b4442c6a6c2f);
const ATAN_AT10: f64 = f64::from_bits(0x3f90ad3ae322da11);

/// Table entries where the `libm` crate's `sin` / `cos` round one ulp away
/// from `StrictMath` (fdlibm): `(table, index, ES's bits)`; tables 0 `sin`,
/// 1 `cos`, 2-5 `tan`'s derivatives (from `sin` / `cos` of their angles).
/// Checked against ES's own tables by `tables_match_es_dump`.
const STRICT_MATH_FIXES: [(u8, u16, u64); 199] = [
    (0, 300, 0x3fe9777ef4c7d742),
    (0, 305, 0x3fe9c2d110f075c2),
    (0, 386, 0x3feda383a9668988),
    (0, 560, 0x3fefa7557f08a516),
    (0, 594, 0x3feefe220c0b95ec),
    (0, 699, 0x3feae068f345ecf0),
    (0, 715, 0x3fe9fdf4f13149de),
    (0, 743, 0x3fe84b7111af83fa),
    (0, 765, 0x3fe6d5afef4aafce),
    (0, 1405, 0xbfed733f508c0e00),
    (0, 1589, 0xbfef93f14f85ac08),
    (0, 1602, 0xbfef58a2b1789e84),
    (0, 1611, 0xbfef2817fc4609ce),
    (0, 1636, 0xbfee817bab4cd10c),
    (0, 1682, 0xbfecd7d9898b32f6),
    (0, 1767, 0xbfe84b7111af83fc),
    (0, 1777, 0xbfe7a4f707bf97d0),
    (0, 1778, 0xbfe79400574f55e6),
    (1, 237, 0x3fe7e83f87b03686),
    (1, 242, 0x3fe79400574f55e4),
    (1, 688, 0xbfe073879922ffec),
    (1, 775, 0xbfe71bac960e41be),
    (1, 797, 0xbfe88c66e7481ba0),
    (1, 934, 0xbfeec9b2d3c3bf84),
    (1, 943, 0xbfef045a14cf738c),
    (1, 954, 0xbfef43d085ff92dc),
    (1, 964, 0xbfef7599a3a12078),
    (1, 972, 0xbfef97f924c9099a),
    (1, 1149, 0xbfedacf42ce68aba),
    (1, 1204, 0xbfeb3e4d3ef55710),
    (1, 1228, 0xbfe9ef43ef29af94),
    (1, 1824, 0x3fe8bc806b151742),
    (1, 1826, 0x3fe8dc45331698cc),
    (1, 1910, 0x3fed2cb220e0ef9e),
    (1, 1911, 0x3fed36fc7bcbfbdc),
    (1, 1951, 0x3fee97ec36016b32),
    (2, 645, 0x3ff10532e5d5359e),
    (2, 789, 0x3ff18f232ee76448),
    (2, 842, 0x3ff1ca9d9b221715),
    (2, 1037, 0x3ff2d2c3629ccd01),
    (2, 1555, 0x3ff75f24f0daf76d),
    (2, 1650, 0x3ff89ae8026e52a1),
    (2, 1657, 0x3ff8b3d392fa9197),
    (2, 1790, 0x3ffabe1841eb8974),
    (2, 1846, 0x3ffbb8dcb32b6874),
    (2, 1867, 0x3ffc1c3148538dce),
    (2, 1896, 0x3ffcaa68166bf78c),
    (2, 1898, 0x3ffcb46fce69e399),
    (2, 1929, 0x3ffd53b561394738),
    (2, 1933, 0x3ffd68cabd985675),
    (2, 1936, 0x3ffd78afaae5b098),
    (2, 1966, 0x3ffe1b90e114f43a),
    (2, 1977, 0x3ffe59217d335827),
    (2, 1990, 0x3ffea334e0847b75),
    (2, 2029, 0x3fff8a569b3cba91),
    (3, 645, 0x3fd131381deabaa6),
    (3, 789, 0x3fd5ecdc33a3d7e4),
    (3, 842, 0x3fd7d02858564d91),
    (3, 1037, 0x3fdfa0cdd46314b3),
    (3, 1555, 0x3fefba5b4b776667),
    (3, 1650, 0x3ff20b5fe7672e64),
    (3, 1657, 0x3ff237cecee3073e),
    (3, 1790, 0x3ff5e9a9b10c66dc),
    (3, 1846, 0x3ff7ba7bfa32ad37),
    (3, 1867, 0x3ff8749dee22c1da),
    (3, 1896, 0x3ff981221e233873),
    (3, 1898, 0x3ff9942979093ff1),
    (3, 1929, 0x3ffac3effb8d0a62),
    (3, 1933, 0x3ffaec60ac6d90a4),
    (3, 1936, 0x3ffb0ae65a8f24db),
    (3, 1966, 0x3ffc4570f2bb3084),
    (3, 1977, 0x3ffcbd2af13d5bf6),
    (3, 1990, 0x3ffd4dd5b3aecbf9),
    (3, 2029, 0x3fff158584a188d0),
    (3, 2091, 0x400119b840b58045),
    (3, 2095, 0x400135257aafc09b),
    (3, 2105, 0x40017aaa628e3a76),
    (3, 2118, 0x4001d71f18fcd306),
    (3, 2150, 0x4002c534f9b3ddb2),
    (3, 2183, 0x4003cb759a9be495),
    (3, 2191, 0x40040dc25aa42119),
    (3, 2195, 0x40042f52d1bc3b27),
    (3, 2211, 0x4004b8699cd81f6e),
    (3, 2255, 0x40064a25dfde0666),
    (3, 2350, 0x400a3c549eb0f63f),
    (3, 2400, 0x400cad8d1517daca),
    (3, 2405, 0x400cf02855267807),
    (3, 2415, 0x400d77be4c45c4c1),
    (3, 2440, 0x400ed91d0cc10804),
    (3, 2458, 0x400fe4fa94d5a7c3),
    (3, 2706, 0x401a4092e7111a0b),
    (3, 2753, 0x401d210f786e3699),
    (3, 2983, 0x4029a8c7bc7f6f59),
    (3, 3023, 0x402ca58aab83370e),
    (3, 3030, 0x402d375cf22d863a),
    (3, 3052, 0x402f1b8426c98c77),
    (3, 3118, 0x4032ed8f39caf4e8),
    (3, 3139, 0x4034343434ef932d),
    (3, 3257, 0x403dffdcd81ce30e),
    (3, 3274, 0x403fe6c1333ed70a),
    (3, 3351, 0x40456e185045790b),
    (3, 3397, 0x4049f2b738375d67),
    (3, 3412, 0x404bb199a3a18b2c),
    (3, 3436, 0x404ed3d1840b8768),
    (3, 3438, 0x404f1c06bf4142bf),
    (3, 3478, 0x4052c6a44c534376),
    (4, 645, 0x3fdb0904fe77488c),
    (4, 789, 0x3fde41a4a66a35c9),
    (4, 842, 0x3fdfb0aea4bd388d),
    (4, 1037, 0x3fe3311b5ca2f2a7),
    (4, 1555, 0x3ff28efcb03a54a2),
    (4, 1650, 0x3ff56f4e6fa9d304),
    (4, 1657, 0x3ff5ab7d6fd9b405),
    (4, 1790, 0x3ffade912a8d83e8),
    (4, 1846, 0x3ffd8d02efe812f1),
    (4, 1867, 0x3ffea567fa01ad28),
    (4, 1896, 0x40001f9ee859c0fa),
    (4, 1898, 0x40002e427a6ecee0),
    (4, 1929, 0x40011a01ee112779),
    (4, 1933, 0x400139adb89b86c4),
    (4, 1936, 0x400151a0560f28a2),
    (4, 1966, 0x40024a971a0c1121),
    (4, 1977, 0x4002aa6138c54c4f),
    (4, 1990, 0x40031edc927d8e7e),
    (4, 2029, 0x400492eba4ec11c0),
    (4, 2074, 0x40066e846490c5d4),
    (4, 2091, 0x4007306c757256f8),
    (4, 2105, 0x4007d66d531410dc),
    (4, 2124, 0x4008c14f3f2b23d5),
    (4, 2191, 0x400c5f48649d1361),
    (4, 2211, 0x400d949acfc6b682),
    (4, 2255, 0x40103d192299feee),
    (4, 2400, 0x40167b8d1d01dd6d),
    (4, 2415, 0x40174a52f8994d2e),
    (4, 2440, 0x4018b832679b7c9c),
    (4, 2458, 0x4019d1426b860ff5),
    (4, 2476, 0x401afa3778e567e5),
    (4, 2706, 0x4028d3fabdb5f1ad),
    (4, 2753, 0x402c7b19327ee78d),
    (4, 2886, 0x403598783fc9dc06),
    (4, 3030, 0x4041ea3c632b02ca),
    (4, 3052, 0x404378e5af948269),
    (4, 3118, 0x404947e99b2dd176),
    (4, 3139, 0x404b92a41f2e0060),
    (4, 3351, 0x4062c376fe2f0f12),
    (4, 3397, 0x4068360c1c781ce2),
    (4, 3412, 0x406a67cfd8dd06a1),
    (4, 3436, 0x406e75e182d55ce1),
    (5, 645, 0x3fc91d9dfae82b41),
    (5, 789, 0x3fd0c0db0c1cbf91),
    (5, 842, 0x3fd28aabf9deab24),
    (5, 1037, 0x3fdaaa991062c868),
    (5, 1555, 0x3ff1e28d9bd21cd3),
    (5, 1650, 0x3ff5bbf85b62bf4c),
    (5, 1657, 0x3ff60ddd325a9435),
    (5, 1790, 0x3ffd52287c8cd740),
    (5, 1846, 0x40009a028be0a0fc),
    (5, 1867, 0x400168263cb66539),
    (5, 1896, 0x4002989e7ad16eea),
    (5, 1898, 0x4002ae82d34856e6),
    (5, 1929, 0x40041198136c045e),
    (5, 1933, 0x400441a7a2b29e06),
    (5, 1936, 0x4004660d06d538f4),
    (5, 1966, 0x4005e336fc6977a2),
    (5, 1977, 0x40067732a23e3eb6),
    (5, 1990, 0x40072c1ef4776d9e),
    (5, 2029, 0x400974f95d7a4471),
    (5, 2074, 0x400c6f5d5835508e),
    (5, 2091, 0x400daac05a5eec06),
    (5, 2105, 0x400ebac706cea954),
    (5, 2118, 0x400fc1d753ffd622),
    (5, 2150, 0x40113bb84005f05e),
    (5, 2183, 0x4012c706896162c3),
    (5, 2195, 0x4013612c864770b0),
    (5, 2211, 0x401437ff71dcf629),
    (5, 2255, 0x4016c2af55ef6175),
    (5, 2286, 0x4018c850a6fa5a8c),
    (5, 2350, 0x401dae32b369cb0c),
    (5, 2400, 0x40212a1f2980a48a),
    (5, 2440, 0x402357dade6917fb),
    (5, 2458, 0x40246e50a8d84ba0),
    (5, 2476, 0x402597bfcbed3f82),
    (5, 2706, 0x40373a2c4fb7f3b1),
    (5, 2753, 0x403b967975a025a4),
    (5, 2886, 0x40473ccf5df45682),
    (5, 3023, 0x40553076056c8a4f),
    (5, 3030, 0x4055e4ee0cf03bed),
    (5, 3052, 0x40584cf2fd2b3500),
    (5, 3088, 0x405cf62335c0a9ac),
    (5, 3118, 0x4060d7a26fc10e81),
    (5, 3139, 0x4062c5f084738acf),
    (5, 3257, 0x40721fdca29894c6),
    (5, 3324, 0x407b7a74d98051c4),
    (5, 3346, 0x407fc0841682a50c),
    (5, 3397, 0x408693aa40191710),
    (5, 3412, 0x408929d0b12b556c),
    (5, 3436, 0x408e1565207eb006),
    (5, 3438, 0x408e8b26ceb675ae),
    (5, 3478, 0x4094e58c65d2370d),
];

/// The look-up tables (`FastMath`'s static initializer) and the values
/// derived from `StrictMath` calls.
struct Tabs {
    sin: Vec<f64>,
    cos: Vec<f64>,
    tan_tabs_size: usize,
    tan: [Vec<f64>; 5],
    asin_max_value_for_tabs: f64,
    asin_delta: f64,
    asin_indexer: f64,
    asin: [Vec<f64>; 5],
    asin_max_value_for_powtabs: f64,
    asin_powtabs_one_div_max_value: f64,
    asin_param_pow: Vec<f64>,
    asin_pow: [Vec<f64>; 5],
    atan_max_value_for_tabs: f64,
    atan_delta: f64,
    atan_indexer: f64,
    atan: [Vec<f64>; 5],
    normalize_angle_max_medium_double: f64,
}

fn tabs() -> &'static Tabs {
    static T: OnceLock<Tabs> = OnceLock::new();
    T.get_or_init(build)
}

/// `Math.toRadians`.
#[inline]
fn to_radians(deg: f64) -> f64 {
    deg * 0.017453292519943295
}

fn build() -> Tabs {
    // sin and cos
    let mut sin = vec![0.0; SIN_COS_TABS_SIZE];
    let mut cos = vec![0.0; SIN_COS_TABS_SIZE];
    let pi_index = (SIN_COS_TABS_SIZE - 1) / 2;
    let pi_mul_2_index = 2 * pi_index;
    let pi_mul_0_5_index = pi_index / 2;
    let pi_mul_1_5_index = 3 * pi_index / 2;
    for i in 0..SIN_COS_TABS_SIZE {
        let angle = i as f64 * SIN_COS_DELTA_HI + i as f64 * SIN_COS_DELTA_LO;
        let mut s = libm::sin(angle);
        let mut c = libm::cos(angle);
        if i == pi_index || i == pi_mul_2_index {
            s = 0.0;
        } else if i == pi_mul_0_5_index || i == pi_mul_1_5_index {
            c = 0.0;
        }
        sin[i] = s;
        cos[i] = c;
    }

    // tan
    let tan_max_value_for_tabs = to_radians(77.0);
    let tan_tabs_size = ((tan_max_value_for_tabs / (std::f64::consts::PI / 2.0)) * (TAN_VIRTUAL_TABS_SIZE - 1) as f64) as i32 as usize + 1;
    let mut tan: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; tan_tabs_size]);
    for i in 0..tan_tabs_size {
        let angle = i as f64 * TAN_DELTA_HI + i as f64 * TAN_DELTA_LO;
        tan[0][i] = libm::tan(angle);
        let cos_angle = libm::cos(angle);
        let sin_angle = libm::sin(angle);
        let inv = 1.0 / cos_angle;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let inv4 = inv2 * inv2;
        let inv5 = inv3 * inv2;
        tan[1][i] = inv2;
        tan[2][i] = ((2.0 * sin_angle) * inv3) * ONE_DIV_F2;
        tan[3][i] = ((2.0 * (1.0 + 2.0 * sin_angle * sin_angle)) * inv4) * ONE_DIV_F3;
        tan[4][i] = ((8.0 * sin_angle * (2.0 + sin_angle * sin_angle)) * inv5) * ONE_DIV_F4;
    }

    // asin
    let asin_max_value_for_tabs = libm::sin(to_radians(73.0));
    let asin_delta = asin_max_value_for_tabs / (ASIN_TABS_SIZE - 1) as f64;
    let asin_indexer = 1.0 / asin_delta;
    let derivs = |x: f64, t: &mut [Vec<f64>; 5], i: usize| {
        t[0][i] = libm::asin(x);
        let inv = 1.0 / (1.0 - x * x);
        let inv0_5 = libm::sqrt(inv);
        let inv1_5 = inv0_5 * inv;
        let inv2_5 = inv1_5 * inv;
        let inv3_5 = inv2_5 * inv;
        t[1][i] = inv0_5;
        t[2][i] = (x * inv1_5) * ONE_DIV_F2;
        t[3][i] = ((1.0 + 2.0 * x * x) * inv2_5) * ONE_DIV_F3;
        t[4][i] = ((5.0 + 2.0 * x * (2.0 + x * (5.0 - 2.0 * x))) * inv3_5) * ONE_DIV_F4;
    };
    let mut asin: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; ASIN_TABS_SIZE]);
    for i in 0..ASIN_TABS_SIZE {
        derivs(i as f64 * asin_delta, &mut asin, i);
    }
    let asin_max_value_for_powtabs = libm::sin(to_radians(88.6));
    let asin_powtabs_one_div_max_value = 1.0 / asin_max_value_for_powtabs;
    let mut asin_param_pow = vec![0.0; ASIN_POWTABS_SIZE];
    let mut asin_pow: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; ASIN_POWTABS_SIZE]);
    for i in 0..ASIN_POWTABS_SIZE {
        let x = libm::pow(i as f64 * (1.0 / ASIN_POWTABS_SIZE_MINUS_ONE as f64), 1.0 / ASIN_POWTABS_POWER as f64) * asin_max_value_for_powtabs;
        asin_param_pow[i] = x;
        derivs(x, &mut asin_pow, i);
    }

    // atan
    let atan_max_value_for_tabs = libm::tan(to_radians(74.0));
    let atan_delta = atan_max_value_for_tabs / (ATAN_TABS_SIZE - 1) as f64;
    let atan_indexer = 1.0 / atan_delta;
    let mut atan: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; ATAN_TABS_SIZE]);
    for i in 0..ATAN_TABS_SIZE {
        let x = i as f64 * atan_delta;
        let inv = 1.0 / (1.0 + x * x);
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let inv4 = inv2 * inv2;
        atan[0][i] = libm::atan(x);
        atan[1][i] = inv;
        atan[2][i] = (-2.0 * x * inv2) * ONE_DIV_F2;
        atan[3][i] = ((-2.0 + 6.0 * x * x) * inv3) * ONE_DIV_F3;
        atan[4][i] = ((24.0 * x * (1.0 - x * x)) * inv4) * ONE_DIV_F4;
    }

    for &(t, i, bits) in STRICT_MATH_FIXES.iter() {
        let v = f64::from_bits(bits);
        match t {
            0 => sin[i as usize] = v,
            1 => cos[i as usize] = v,
            k => tan[k as usize - 1][i as usize] = v,
        }
    }

    Tabs {
        sin,
        cos,
        tan_tabs_size,
        tan,
        asin_max_value_for_tabs,
        asin_delta,
        asin_indexer,
        asin,
        asin_max_value_for_powtabs,
        asin_powtabs_one_div_max_value,
        asin_param_pow,
        asin_pow,
        atan_max_value_for_tabs,
        atan_delta,
        atan_indexer,
        atan,
        normalize_angle_max_medium_double: libm::pow(2.0, 20.0) * (2.0 * std::f64::consts::PI),
    }
}

/// `remainderTwoPi` for the angles H3 uses: the medium-range reduction
/// (fdlibm's `__ieee754_rem_pio2` approach). Larger angles (never reached
/// from coordinates) reduce with `%`.
fn remainder_two_pi(angle: f64) -> f64 {
    let t = tabs();
    let (angle, negate) = if angle < 0.0 { (-angle, true) } else { (angle, false) };
    let result = if angle <= t.normalize_angle_max_medium_double {
        let fn_ = (angle * INVTWOPI + 0.5) as i32 as f64;
        (angle - fn_ * TWOPI_HI) - fn_ * TWOPI_LO
    } else if angle < f64::INFINITY {
        let r = angle % M_2PI;
        if r > std::f64::consts::PI {
            r - M_2PI
        } else {
            r
        }
    } else {
        f64::NAN
    };
    if negate {
        -result
    } else {
        result
    }
}

/// `FastMath.cos`.
pub fn cos(angle: f64) -> f64 {
    let t = tabs();
    let mut angle = angle.abs();
    if angle > SIN_COS_MAX_VALUE_FOR_INT_MODULO {
        angle = remainder_two_pi(angle);
        if angle < 0.0 {
            angle += M_2PI;
        }
    }
    let mut index = (angle * SIN_COS_INDEXER + 0.5) as i32;
    let delta = (angle - index as f64 * SIN_COS_DELTA_HI) - index as f64 * SIN_COS_DELTA_LO;
    index &= (SIN_COS_TABS_SIZE - 2) as i32;
    let index_cos = t.cos[index as usize];
    let index_sin = t.sin[index as usize];
    index_cos + delta * (-index_sin + delta * (-index_cos * ONE_DIV_F2 + delta * (index_sin * ONE_DIV_F3 + delta * index_cos * ONE_DIV_F4)))
}

/// `FastMath.sin`.
pub fn sin(angle: f64) -> f64 {
    let t = tabs();
    let (mut angle, negate) = if angle < 0.0 { (-angle, true) } else { (angle, false) };
    if angle > SIN_COS_MAX_VALUE_FOR_INT_MODULO {
        angle = remainder_two_pi(angle);
        if angle < 0.0 {
            angle += M_2PI;
        }
    }
    let mut index = (angle * SIN_COS_INDEXER + 0.5) as i32;
    let delta = (angle - index as f64 * SIN_COS_DELTA_HI) - index as f64 * SIN_COS_DELTA_LO;
    index &= (SIN_COS_TABS_SIZE - 2) as i32;
    let index_sin = t.sin[index as usize];
    let index_cos = t.cos[index as usize];
    let result = index_sin + delta * (index_cos + delta * (-index_sin * ONE_DIV_F2 + delta * (-index_cos * ONE_DIV_F3 + delta * index_sin * ONE_DIV_F4)));
    if negate {
        -result
    } else {
        result
    }
}

/// `FastMath.tan`.
pub fn tan(angle: f64) -> f64 {
    let t = tabs();
    let mut angle = angle;
    if angle.abs() > TAN_MAX_VALUE_FOR_INT_MODULO {
        angle = remainder_two_pi(angle);
        if angle < -M_HALF_PI {
            angle += std::f64::consts::PI;
        } else if angle > M_HALF_PI {
            angle -= std::f64::consts::PI;
        }
    }
    let mut negate = false;
    if angle < 0.0 {
        angle = -angle;
        negate = true;
    }
    let mut index = (angle * TAN_INDEXER + 0.5) as i32;
    let mut delta = (angle - index as f64 * TAN_DELTA_HI) - index as f64 * TAN_DELTA_LO;
    index &= (2 * (TAN_VIRTUAL_TABS_SIZE - 1) - 1) as i32;
    if index > (TAN_VIRTUAL_TABS_SIZE - 1) as i32 {
        index = (2 * (TAN_VIRTUAL_TABS_SIZE - 1)) as i32 - index;
        delta = -delta;
        negate = !negate;
    }
    let tt = &t.tan;
    let result = if (index as usize) < t.tan_tabs_size {
        let i = index as usize;
        tt[0][i] + delta * (tt[1][i] + delta * (tt[2][i] + delta * (tt[3][i] + delta * tt[4][i])))
    } else {
        let i = ((TAN_VIRTUAL_TABS_SIZE - 1) as i32 - index) as usize;
        1.0 / (tt[0][i] - delta * (tt[1][i] - delta * (tt[2][i] - delta * (tt[3][i] - delta * tt[4][i]))))
    };
    if negate {
        -result
    } else {
        result
    }
}

/// `FastMath.acos`.
pub fn acos(value: f64) -> f64 {
    M_HALF_PI - asin(value)
}

/// `FastMath.asin`.
pub fn asin(value: f64) -> f64 {
    let t = tabs();
    let (value, negate) = if value < 0.0 { (-value, true) } else { (value, false) };
    let sign = |r: f64| if negate { -r } else { r };
    if value <= t.asin_max_value_for_tabs {
        let index = (value * t.asin_indexer + 0.5) as i32 as usize;
        let delta = value - index as f64 * t.asin_delta;
        let a = &t.asin;
        sign(a[0][index] + delta * (a[1][index] + delta * (a[2][index] + delta * (a[3][index] + delta * a[4][index]))))
    } else if value <= t.asin_max_value_for_powtabs {
        let index = (pow_fast(value * t.asin_powtabs_one_div_max_value, ASIN_POWTABS_POWER) * ASIN_POWTABS_SIZE_MINUS_ONE as f64 + 0.5) as i32 as usize;
        let delta = value - t.asin_param_pow[index];
        let a = &t.asin_pow;
        sign(a[0][index] + delta * (a[1][index] + delta * (a[2][index] + delta * (a[3][index] + delta * a[4][index]))))
    } else if value < 1.0 {
        let t = (1.0 - value) * 0.5;
        let p = t * (ASIN_PS0 + t * (ASIN_PS1 + t * (ASIN_PS2 + t * (ASIN_PS3 + t * (ASIN_PS4 + t * ASIN_PS5)))));
        let q = 1.0 + t * (ASIN_QS1 + t * (ASIN_QS2 + t * (ASIN_QS3 + t * ASIN_QS4)));
        let s = t.sqrt();
        let z = s + s * (p / q);
        sign(ASIN_PIO2_HI - ((z + z) - ASIN_PIO2_LO))
    } else if value == 1.0 {
        sign(M_HALF_PI)
    } else {
        f64::NAN
    }
}

/// `FastMath.atan`.
pub fn atan(value: f64) -> f64 {
    let t = tabs();
    let (value, negate) = if value < 0.0 { (-value, true) } else { (value, false) };
    let sign = |r: f64| if negate { -r } else { r };
    if value == 1.0 {
        sign(M_QUARTER_PI)
    } else if value <= t.atan_max_value_for_tabs {
        let index = (value * t.atan_indexer + 0.5) as i32 as usize;
        let delta = value - index as f64 * t.atan_delta;
        let a = &t.atan;
        sign(a[0][index] + delta * (a[1][index] + delta * (a[2][index] + delta * (a[3][index] + delta * a[4][index]))))
    } else if value < TWO_POW_66 {
        let x = -1.0 / value;
        let x2 = x * x;
        let x4 = x2 * x2;
        let s1 = x2 * (ATAN_AT0 + x4 * (ATAN_AT2 + x4 * (ATAN_AT4 + x4 * (ATAN_AT6 + x4 * (ATAN_AT8 + x4 * ATAN_AT10)))));
        let s2 = x4 * (ATAN_AT1 + x4 * (ATAN_AT3 + x4 * (ATAN_AT5 + x4 * (ATAN_AT7 + x4 * ATAN_AT9))));
        sign(ATAN_HI3 - ((x * (s1 + s2) - ATAN_LO3) - x))
    } else if value.is_nan() {
        f64::NAN
    } else {
        sign(M_HALF_PI)
    }
}

/// `FastMath.atan2`.
pub fn atan2(y: f64, x: f64) -> f64 {
    let neg_zero = |v: f64| v == 0.0 && v.is_sign_negative();
    if x > 0.0 {
        if y == 0.0 {
            return if neg_zero(y) { -0.0 } else { 0.0 };
        }
        if x == f64::INFINITY {
            if y == f64::INFINITY {
                M_QUARTER_PI
            } else if y == f64::NEG_INFINITY {
                -M_QUARTER_PI
            } else if y > 0.0 {
                0.0
            } else if y < 0.0 {
                -0.0
            } else {
                f64::NAN
            }
        } else {
            atan(y / x)
        }
    } else if x < 0.0 {
        if y == 0.0 {
            return if neg_zero(y) { -std::f64::consts::PI } else { std::f64::consts::PI };
        }
        if x == f64::NEG_INFINITY {
            if y == f64::INFINITY {
                3.0 * M_QUARTER_PI
            } else if y == f64::NEG_INFINITY {
                -3.0 * M_QUARTER_PI
            } else if y > 0.0 {
                std::f64::consts::PI
            } else if y < 0.0 {
                -std::f64::consts::PI
            } else {
                f64::NAN
            }
        } else if y > 0.0 {
            M_HALF_PI + atan(-x / y)
        } else if y < 0.0 {
            -M_HALF_PI - atan(x / y)
        } else {
            f64::NAN
        }
    } else if x == 0.0 {
        if y == 0.0 {
            return if neg_zero(x) {
                if neg_zero(y) {
                    -std::f64::consts::PI
                } else {
                    std::f64::consts::PI
                }
            } else if neg_zero(y) {
                -0.0
            } else {
                0.0
            };
        }
        if y > 0.0 {
            M_HALF_PI
        } else if y < 0.0 {
            -M_HALF_PI
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    }
}

/// `FastMath.powFast`.
fn pow_fast(mut value: f64, mut power: i32) -> f64 {
    if power > 5 {
        let mut odd_remains = 1.0;
        loop {
            if power & 1 != 0 {
                odd_remains *= value;
            }
            value *= value;
            power >>= 1;
            if power <= 5 {
                break;
            }
        }
        if power == 3 {
            odd_remains * value * value * value
        } else {
            let v2 = value * value;
            if power == 4 {
                odd_remains * v2 * v2
            } else {
                odd_remains * v2 * v2 * value
            }
        }
    } else if power >= 0 {
        match power {
            2 => value * value,
            1 => value,
            0 => 1.0,
            3 => value * value * value,
            4 => {
                let v2 = value * value;
                v2 * v2
            }
            _ => {
                let v2 = value * value;
                v2 * v2 * value
            }
        }
    } else {
        1.0 / pow_fast(value, -power)
    }
}

#[cfg(test)]
mod tests {
    /// The tables equal ES's (`FastMath`'s, dumped by reflection to the
    /// file in `H3_TABS`; skipped without it).
    #[test]
    fn tables_match_es_dump() {
        let Ok(path) = std::env::var("H3_TABS") else { return };
        let text = std::fs::read_to_string(path).unwrap();
        let t = super::tabs();
        let mine: Vec<(&str, &Vec<f64>)> = vec![
            ("sinTab", &t.sin),
            ("cosTab", &t.cos),
            ("tanTab", &t.tan[0]),
            ("tanDer1DivF1Tab", &t.tan[1]),
            ("tanDer2DivF2Tab", &t.tan[2]),
            ("tanDer3DivF3Tab", &t.tan[3]),
            ("tanDer4DivF4Tab", &t.tan[4]),
            ("asinTab", &t.asin[0]),
            ("asinDer1DivF1Tab", &t.asin[1]),
            ("asinDer2DivF2Tab", &t.asin[2]),
            ("asinDer3DivF3Tab", &t.asin[3]),
            ("asinDer4DivF4Tab", &t.asin[4]),
            ("asinParamPowTab", &t.asin_param_pow),
            ("asinPowTab", &t.asin_pow[0]),
            ("asinDer1DivF1PowTab", &t.asin_pow[1]),
            ("asinDer2DivF2PowTab", &t.asin_pow[2]),
            ("asinDer3DivF3PowTab", &t.asin_pow[3]),
            ("asinDer4DivF4PowTab", &t.asin_pow[4]),
            ("atanTab", &t.atan[0]),
            ("atanDer1DivF1Tab", &t.atan[1]),
            ("atanDer2DivF2Tab", &t.atan[2]),
            ("atanDer3DivF3Tab", &t.atan[3]),
            ("atanDer4DivF4Tab", &t.atan[4]),
        ];
        let mut bad = Vec::new();
        for line in text.lines() {
            let mut it = line.split(' ');
            let name = it.next().unwrap();
            let vals: Vec<u64> = it.map(|h| u64::from_str_radix(h, 16).unwrap()).collect();
            if let Some((_, m)) = mine.iter().find(|(n, _)| *n == name) {
                if m.len() != vals.len() {
                    bad.push(format!("{name}: len {} vs {}", m.len(), vals.len()));
                    continue;
                }
                let diffs: Vec<usize> = (0..vals.len()).filter(|&i| m[i].to_bits() != vals[i]).collect();
                if !diffs.is_empty() {
                    let i = diffs[0];
                    bad.push(format!("{name}: {} differ, first [{i}] {:x} vs {:x}", diffs.len(), m[i].to_bits(), vals[i]));
                }
            } else if vals.len() == 1 {
                let v = f64::from_bits(vals[0]);
                let m = match name {
                    "ASIN_MAX_VALUE_FOR_TABS" => t.asin_max_value_for_tabs,
                    "ASIN_MAX_VALUE_FOR_POWTABS" => t.asin_max_value_for_powtabs,
                    "ATAN_MAX_VALUE_FOR_TABS" => t.atan_max_value_for_tabs,
                    "SIN_COS_MAX_VALUE_FOR_INT_MODULO" => super::SIN_COS_MAX_VALUE_FOR_INT_MODULO,
                    _ => v,
                };
                if m.to_bits() != v.to_bits() {
                    bad.push(format!("{name}: {m:?} vs {v:?}"));
                }
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    }
}
