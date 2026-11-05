//! The ML `p_value` significance heuristic (`PValueScore`): the negated
//! log of the likelihood-ratio test p-value that a term's frequency in the
//! foreground equals its frequency in the background.

use std::f64::consts::PI;

const EXACT_STIRLING_ERRORS: [f64; 31] = [
    0.0,
    0.1534264097200273452913848,
    0.0810614667953272582196702,
    0.0548141210519176538961390,
    0.0413406959554092940938221,
    0.03316287351993628748511048,
    0.02767792568499833914878929,
    0.02374616365629749597132920,
    0.02079067210376509311152277,
    0.01848845053267318523077934,
    0.01664469118982119216319487,
    0.01513497322191737887351255,
    0.01387612882307074799874573,
    0.01281046524292022692424986,
    0.01189670994589177009505572,
    0.01110455975820691732662991,
    0.010411265261972096497478567,
    0.009799416126158803298389475,
    0.009255462182712732917728637,
    0.008768700134139385462952823,
    0.008330563433362871256469318,
    0.007934114564314020547248100,
    0.007573675487951840794972024,
    0.007244554301320383179543912,
    0.006942840107209529865664152,
    0.006665247032707682442354394,
    0.006408994188004207068439631,
    0.006171712263039457647532867,
    0.005951370112758847735624416,
    0.005746216513010115682023589,
    0.005554733551962801371038690,
];

fn deviance_part(x: f64, mu: f64) -> f64 {
    if (x - mu).abs() < 0.1 * (x + mu) {
        let d = x - mu;
        let mut v = d / (x + mu);
        let mut s1 = v * d;
        let mut s = f64::NAN;
        let mut ej = 2.0 * x * v;
        v *= v;
        let mut j = 1;
        while s1 != s {
            s = s1;
            ej *= v;
            s1 = s + ej / ((j * 2) + 1) as f64;
            j += 1;
        }
        s1
    } else {
        x * (x / mu).ln() + mu - x
    }
}

fn stirling_error(z: f64) -> f64 {
    if z < 15.0 {
        let z2 = 2.0 * z;
        if z2.floor() == z2 {
            EXACT_STIRLING_ERRORS[z2 as usize]
        } else {
            libm::lgamma(z + 1.0) - (z + 0.5) * z.ln() + z - 0.5 * (2.0 * PI).ln()
        }
    } else {
        let z2 = z * z;
        (0.083333333333333333333 - (0.00277777777777777777778 - (0.00079365079365079365079365 - (0.000595238095238095238095238 - 0.0008417508417508417508417508 / z2) / z2) / z2) / z2) / z
    }
}

fn log_binomial(x: i64, n: i64, p: f64, q: f64) -> f64 {
    let (xf, nf) = (x as f64, n as f64);
    if x == 0 {
        if p < 0.1 {
            -deviance_part(nf, nf * q) - nf * p
        } else {
            nf * q.ln()
        }
    } else if x == n {
        if q < 0.1 {
            -deviance_part(nf, nf * p) - nf * q
        } else {
            nf * p.ln()
        }
    } else {
        let ret = stirling_error(nf) - stirling_error(xf) - stirling_error(nf - xf) - deviance_part(xf, nf * p) - deviance_part(nf - xf, nf * q);
        let f = (2.0 * PI * xf * (nf - xf)) / nf;
        -0.5 * f.ln() + ret
    }
}

/// `LongBinomialDistribution(n, p).logProbability(x)`.
fn log_prob(n: i64, p: f64, x: i64) -> f64 {
    if n == 0 {
        return if x == 0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if x < 0 || x > n {
        return f64::NEG_INFINITY;
    }
    log_binomial(x, n, p, 1.0 - p)
}

/// Chi-squared (1 degree of freedom) survival function.
fn chi2_1_survival(x: f64) -> f64 {
    if x <= 0.0 {
        1.0
    } else {
        libm::erfc((x / 2.0).sqrt())
    }
}

fn eps(v: f64) -> f64 {
    (0.05 * v + 0.5).max(1.0)
}

/// `PValueScore.getScore`. `normalize_above` 0 means no normalization.
pub fn score(sub_f: u64, sub_n: u64, sup_f: u64, sup_n: u64, bg_superset: bool, normalize_above: u64) -> f64 {
    let (sub_f, sub_n, sup_f, sup_n) = (sub_f as f64, sub_n as f64, sup_f as f64, sup_n as f64);
    let (n11, n_1, n10, n_0) = if bg_superset { (sub_f, sub_n, sup_f - sub_f, sup_n - sub_n) } else { (sub_f, sub_n, sup_f, sup_n) };
    let (mut in_c, mut all_in, mut not_in_c, mut all_not) = (n11, n_1, n10, n_0);
    if in_c * all_not <= all_in * not_in_c {
        return 0.0;
    }
    if all_not == 0.0 || all_in == 0.0 {
        return 0.0;
    }
    if normalize_above > 0 {
        let na = normalize_above as f64;
        if all_in > na {
            let f = na / all_in;
            all_in = ((all_in * f) as i64) as f64;
            in_c = ((in_c * f) as i64) as f64;
        }
        if all_not > na {
            let f = na / all_not;
            all_not = ((all_not * f) as i64) as f64;
            not_in_c = ((not_in_c * f) as i64) as f64;
        }
    }
    let e_in = (eps(all_in) as i64) as f64;
    let e_not = (eps(all_not) as i64) as f64;
    in_c += e_in;
    not_in_c += e_not;
    all_in += e_in;
    all_not += e_not;
    not_in_c = (((1.05 * not_in_c).min(in_c / all_in * all_not) + 0.5) as i64) as f64;
    let v1 = log_prob(all_in as i64, in_c / all_in, in_c as i64);
    let v2 = log_prob(all_not as i64, not_in_c / all_not, not_in_c as i64);
    let p2 = (in_c + not_in_c) / (all_in + all_not);
    let v3 = log_prob(all_in as i64, p2, in_c as i64);
    let v4 = log_prob(all_not as i64, p2, not_in_c as i64);
    let llr = v1 + v2 - v3 - v4;
    let p = chi2_1_survival(2.0 * llr);
    (-(p.max(f64::MIN_POSITIVE)).ln()).max(0.0)
}
