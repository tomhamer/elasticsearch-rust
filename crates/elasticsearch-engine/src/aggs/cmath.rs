//! Faithful ports of the Apache commons-math3 3.6.1 special functions ES
//! uses for statistics aggregations (`t_test`, `bucket_correlation`'s
//! F-test, ...). The operation order mirrors the Java code so results match
//! ES to the last bit wherever the platform `ln`/`exp` agree with
//! `FastMath`.

const HALF_LOG_TWO_PI: f64 = 0.9189385332046727;
const DELTA: [f64; 15] = [
    0.833333333333333333333333333333E-01,
    -0.277777777777777777777777752282E-04,
    0.793650793650793650791732130419E-07,
    -0.595238095238095232389839236182E-09,
    0.841750841750832853294451671990E-11,
    -0.191752691751854612334149171243E-12,
    0.641025640510325475730918472625E-14,
    -0.295506514125338232839867823991E-15,
    0.179643716359402238723287696452E-16,
    -0.139228964661627791231203060395E-17,
    0.133802855014020915603275339093E-18,
    -0.154246009867966094273710216533E-19,
    0.197701992980957427278370133333E-20,
    -0.234065664793997056856992426667E-21,
    0.171348014966398575409015466667E-22,
];

const LANCZOS_G: f64 = 607.0 / 128.0;
const LANCZOS: [f64; 15] = [
    0.99999999999999709182,
    57.156235665862923517,
    -59.597960355475491248,
    14.136097974741747174,
    -0.49191381609762019978,
    0.33994649984811888699e-4,
    0.46523628927048575665e-4,
    -0.98374475304879564677e-4,
    0.15808870322491248884e-3,
    -0.21026444172410488319e-3,
    0.21743961811521264320e-3,
    -0.16431810653676389022e-3,
    0.84418223983852743293e-4,
    -0.26190838401581408670e-4,
    0.36899182659531622704e-5,
];
const SQRT_TWO_PI: f64 = 2.506628274631000502;

const A0: f64 = 0.611609510448141581788E-08;
const A1: f64 = 0.624730830116465516210E-08;
const B: [f64; 8] = [
    0.203610414066806987300E+00,
    0.266205348428949217746E-01,
    0.493944979382446875238E-03,
    -0.851419432440314906588E-05,
    -0.643045481779353022248E-05,
    0.992641840672773722196E-06,
    -0.607761895722825260739E-07,
    0.195755836614639731882E-09,
];
const P: [f64; 7] = [
    0.6116095104481415817861E-08,
    0.6871674113067198736152E-08,
    0.6820161668496170657918E-09,
    0.4686843322948848031080E-10,
    0.1572833027710446286995E-11,
    -0.1249441572276366213222E-12,
    0.4343529937408594255178E-14,
];
const Q: [f64; 4] = [0.3056961078365221025009E+00, 0.5464213086042296536016E-01, 0.4956830093825887312020E-02, 0.2692369466186361192876E-03];
const C_NEG: f64 = -0.422784335098467139393487909917598E+00;
const C: [f64; 14] = [
    0.577215664901532860606512090082402E+00,
    -0.655878071520253881077019515145390E+00,
    -0.420026350340952355290039348754298E-01,
    0.166538611382291489501700795102105E+00,
    -0.421977345555443367482083012891874E-01,
    -0.962197152787697356211492167234820E-02,
    0.721894324666309954239501034044657E-02,
    -0.116516759185906511211397108401839E-02,
    -0.215241674114950972815729963053648E-03,
    0.128050282388116186153198626328164E-03,
    -0.201348547807882386556893914210218E-04,
    -0.125049348214267065734535947383309E-05,
    0.113302723198169588237412962033074E-05,
    -0.205633841697760710345015413002057E-06,
];

fn lanczos(x: f64) -> f64 {
    let mut sum = 0.0;
    for i in (1..LANCZOS.len()).rev() {
        sum += LANCZOS[i] / (x + i as f64);
    }
    sum + LANCZOS[0]
}

fn inv_gamma1pm1(x: f64) -> f64 {
    let t = if x <= 0.5 { x } else { (x - 0.5) - 0.5 };
    if t < 0.0 {
        let a = A0 + t * A1;
        let mut b = B[7];
        for i in (0..7).rev() {
            b = B[i] + t * b;
        }
        b = 1.0 + t * b;
        let mut c = C[13] + t * (a / b);
        for i in (1..13).rev() {
            c = C[i] + t * c;
        }
        c = C_NEG + t * c;
        if x > 0.5 { t * c / x } else { x * ((c + 0.5) + 0.5) }
    } else {
        let mut p = P[6];
        for i in (0..6).rev() {
            p = P[i] + t * p;
        }
        let mut q = Q[3];
        for i in (0..3).rev() {
            q = Q[i] + t * q;
        }
        q = 1.0 + t * q;
        let mut c = C[13] + (p / q) * t;
        for i in (0..13).rev() {
            c = C[i] + t * c;
        }
        if x > 0.5 { (t / x) * ((c - 0.5) - 0.5) } else { x * c }
    }
}

fn log_gamma1p(x: f64) -> f64 {
    -inv_gamma1pm1(x).ln_1p()
}

/// `Gamma.logGamma`.
pub(crate) fn log_gamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        f64::NAN
    } else if x < 0.5 {
        log_gamma1p(x) - x.ln()
    } else if x <= 2.5 {
        log_gamma1p((x - 0.5) - 0.5)
    } else if x <= 8.0 {
        let n = (x - 1.5).floor() as i32;
        let mut prod = 1.0;
        for i in 1..=n {
            prod *= x - i as f64;
        }
        log_gamma1p(x - (n + 1) as f64) + prod.ln()
    } else {
        let sum = lanczos(x);
        let tmp = x + LANCZOS_G + 0.5;
        let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        ((x + 0.5) * tmp.ln()) - tmp + half_log_2pi + (sum / x).ln()
    }
}

/// `Gamma.gamma`.
pub(crate) fn gamma(x: f64) -> f64 {
    if x == x.round_ties_even() && x <= 0.0 {
        return f64::NAN;
    }
    let abs_x = x.abs();
    if abs_x <= 20.0 {
        if x >= 1.0 {
            let mut prod = 1.0;
            let mut t = x;
            while t > 2.5 {
                t -= 1.0;
                prod *= t;
            }
            prod / (1.0 + inv_gamma1pm1(t - 1.0))
        } else {
            let mut prod = x;
            let mut t = x;
            while t < -0.5 {
                t += 1.0;
                prod *= t;
            }
            1.0 / (prod * (1.0 + inv_gamma1pm1(t)))
        }
    } else {
        let y = abs_x + LANCZOS_G + 0.5;
        let gamma_abs = SQRT_TWO_PI / abs_x * y.powf(abs_x + 0.5) * (-y).exp() * lanczos(abs_x);
        if x > 0.0 { gamma_abs } else { -std::f64::consts::PI / (x * (std::f64::consts::PI * x).sin() * gamma_abs) }
    }
}

fn log_gamma_sum(a: f64, b: f64) -> f64 {
    let x = (a - 1.0) + (b - 1.0);
    if x <= 0.5 {
        log_gamma1p(1.0 + x)
    } else if x <= 1.5 {
        log_gamma1p(x) + x.ln_1p()
    } else {
        log_gamma1p(x - 1.0) + (x * (1.0 + x)).ln()
    }
}

fn log_gamma_minus_log_gamma_sum(a: f64, b: f64) -> f64 {
    let (d, w) = if a <= b { (b + (a - 0.5), delta_minus_delta_sum(a, b)) } else { (a + (b - 0.5), delta_minus_delta_sum(b, a)) };
    let u = d * (a / b).ln_1p();
    let v = a * (b.ln() - 1.0);
    if u <= v { (w - u) - v } else { (w - v) - u }
}

fn delta_minus_delta_sum(a: f64, b: f64) -> f64 {
    let h = a / b;
    let p = h / (1.0 + h);
    let q = 1.0 / (1.0 + h);
    let q2 = q * q;
    let mut s = [0.0; 15];
    s[0] = 1.0;
    for i in 1..s.len() {
        s[i] = 1.0 + (q + q2 * s[i - 1]);
    }
    let sqrt_t = 10.0 / b;
    let t = sqrt_t * sqrt_t;
    let mut w = DELTA[14] * s[14];
    for i in (0..14).rev() {
        w = t * w + DELTA[i] * s[i];
    }
    w * p / b
}

fn sum_delta_minus_delta_sum(p: f64, q: f64) -> f64 {
    let a = p.min(q);
    let b = p.max(q);
    let sqrt_t = 10.0 / a;
    let t = sqrt_t * sqrt_t;
    let mut z = DELTA[14];
    for i in (0..14).rev() {
        z = t * z + DELTA[i];
    }
    z / a + delta_minus_delta_sum(a, b)
}

/// `Beta.logBeta`.
pub(crate) fn log_beta(p: f64, q: f64) -> f64 {
    if p.is_nan() || q.is_nan() || p <= 0.0 || q <= 0.0 {
        return f64::NAN;
    }
    let a = p.min(q);
    let b = p.max(q);
    if a >= 10.0 {
        let w = sum_delta_minus_delta_sum(a, b);
        let h = a / b;
        let c = h / (1.0 + h);
        let u = -(a - 0.5) * c.ln();
        let v = b * h.ln_1p();
        if u <= v { (((-0.5 * b.ln() + HALF_LOG_TWO_PI) + w) - u) - v } else { (((-0.5 * b.ln() + HALF_LOG_TWO_PI) + w) - v) - u }
    } else if a > 2.0 {
        if b > 1000.0 {
            let n = (a - 1.0).floor() as i32;
            let mut prod = 1.0;
            let mut ared = a;
            for _ in 0..n {
                ared -= 1.0;
                prod *= ared / (1.0 + ared / b);
            }
            (prod.ln() - n as f64 * b.ln()) + (log_gamma(ared) + log_gamma_minus_log_gamma_sum(ared, b))
        } else {
            let mut prod1 = 1.0;
            let mut ared = a;
            while ared > 2.0 {
                ared -= 1.0;
                let h = ared / b;
                prod1 *= h / (1.0 + h);
            }
            if b < 10.0 {
                let mut prod2 = 1.0;
                let mut bred = b;
                while bred > 2.0 {
                    bred -= 1.0;
                    prod2 *= bred / (ared + bred);
                }
                prod1.ln() + prod2.ln() + (log_gamma(ared) + (log_gamma(bred) - log_gamma_sum(ared, bred)))
            } else {
                prod1.ln() + log_gamma(ared) + log_gamma_minus_log_gamma_sum(ared, b)
            }
        }
    } else if a >= 1.0 {
        if b > 2.0 {
            if b < 10.0 {
                let mut prod = 1.0;
                let mut bred = b;
                while bred > 2.0 {
                    bred -= 1.0;
                    prod *= bred / (a + bred);
                }
                prod.ln() + (log_gamma(a) + (log_gamma(bred) - log_gamma_sum(a, bred)))
            } else {
                log_gamma(a) + log_gamma_minus_log_gamma_sum(a, b)
            }
        } else {
            log_gamma(a) + log_gamma(b) - log_gamma_sum(a, b)
        }
    } else if b >= 10.0 {
        log_gamma(a) + log_gamma_minus_log_gamma_sum(a, b)
    } else {
        (gamma(a) * gamma(b) / gamma(a + b)).ln()
    }
}

/// `Beta.regularizedBeta(x, a, b)` (epsilon 1e-14, unbounded iterations).
pub(crate) fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() || !(0.0..=1.0).contains(&x) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x > (a + 1.0) / (2.0 + b + a) && 1.0 - x <= (b + 1.0) / (2.0 + b + a) {
        return 1.0 - regularized_beta(1.0 - x, b, a);
    }
    let get_b = |n: u64| -> f64 {
        if n % 2 == 0 {
            let m = n as f64 / 2.0;
            (m * (b - m) * x) / ((a + (2.0 * m) - 1.0) * (a + (2.0 * m)))
        } else {
            let m = (n as f64 - 1.0) / 2.0;
            -((a + m) * (a + b + m) * x) / ((a + (2.0 * m)) * (a + (2.0 * m) + 1.0))
        }
    };
    (a * x.ln() + b * (-x).ln_1p() - a.ln() - log_beta(a, b)).exp() * 1.0 / continued_fraction(get_b, 1e-14)
}

/// `ContinuedFraction.evaluate` with `getA(n) == 1`.
fn continued_fraction(get_b: impl Fn(u64) -> f64, epsilon: f64) -> f64 {
    const SMALL: f64 = 1e-50;
    let near_zero = |v: f64| v.abs() <= SMALL;
    let mut h_prev = 1.0;
    let mut d_prev = 0.0;
    let mut c_prev = h_prev;
    let mut h_n = h_prev;
    let mut n = 1u64;
    while n < i32::MAX as u64 {
        let a = 1.0;
        let b = get_b(n);
        let mut d_n = a + b * d_prev;
        if near_zero(d_n) {
            d_n = SMALL;
        }
        let mut c_n = a + b / c_prev;
        if near_zero(c_n) {
            c_n = SMALL;
        }
        d_n = 1.0 / d_n;
        let delta_n = c_n * d_n;
        h_n = h_prev * delta_n;
        if !h_n.is_finite() {
            return f64::NAN;
        }
        if (delta_n - 1.0).abs() < epsilon {
            break;
        }
        d_prev = d_n;
        c_prev = c_n;
        h_prev = h_n;
        n += 1;
    }
    h_n
}

/// `new TDistribution(df).cumulativeProbability(x)`.
pub(crate) fn t_cdf(x: f64, df: f64) -> f64 {
    if x == 0.0 {
        return 0.5;
    }
    let t = regularized_beta(df / (df + (x * x)), 0.5 * df, 0.5);
    if x < 0.0 { 0.5 * t } else { 1.0 - 0.5 * t }
}
