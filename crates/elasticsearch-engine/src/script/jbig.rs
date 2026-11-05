//! `java.math`: arbitrary-precision `BigInteger` / `BigDecimal` with Java
//! semantics (scale rules, `MathContext`, `RoundingMode`, `toString`
//! exponent notation).
//!
//! `BigDecimal` is implemented here as `(unscaled BigInt, scale)` following
//! the OpenJDK algorithms (`divide`, `doRound`, `preAlign`, `layoutChars`,
//! ...) so that results, scales and error messages are byte-identical to
//! what Elasticsearch returns.

use super::ast::Ty;
use super::error::{class_cast, illegal_arg, npe, SResult, ScriptErr};
use super::runtime::Rt;
use super::value::{java_double_string, Val};
use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::rc::Rc;

// JDK stack-frame strings (see the "JDK stack frames" section below).
macro_rules! bd {
    ($m:literal, $l:literal) => {
        concat!("java.base/java.math.BigDecimal.", $m, "(BigDecimal.java:", $l, ")")
    };
}
macro_rules! bi {
    ($m:literal, $l:literal) => {
        concat!("java.base/java.math.BigInteger.", $m, "(BigInteger.java:", $l, ")")
    };
}
macro_rules! mbi {
    ($m:literal, $l:literal) => {
        concat!("java.base/java.math.MutableBigInteger.", $m, "(MutableBigInteger.java:", $l, ")")
    };
}

/// Static classes handled here.
pub const CLASSES: &[&str] = &["BigDecimal", "BigInteger", "MathContext", "RoundingMode"];

// ---------------------------------------------------------------------------
// RoundingMode / MathContext

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rm {
    Up = 0,
    Down = 1,
    Ceiling = 2,
    Floor = 3,
    HalfUp = 4,
    HalfDown = 5,
    HalfEven = 6,
    Unnecessary = 7,
}

const RMS: [Rm; 8] = [Rm::Up, Rm::Down, Rm::Ceiling, Rm::Floor, Rm::HalfUp, Rm::HalfDown, Rm::HalfEven, Rm::Unnecessary];
const RM_NAMES: [&str; 8] = ["UP", "DOWN", "CEILING", "FLOOR", "HALF_UP", "HALF_DOWN", "HALF_EVEN", "UNNECESSARY"];
/// `Enum.hashCode()` is an identity hash; these are the values of the
/// reference JVM instance (they are not reproducible across JVM restarts).
const RM_HASHES: [i32; 8] = [2106556996, 1446465740, 221714027, 996956029, 1263007563, 988839926, 1453382340, 1454238036];

impl Rm {
    fn name(self) -> &'static str {
        RM_NAMES[self as usize]
    }
    fn from_name(s: &str) -> Option<Rm> {
        RM_NAMES.iter().position(|n| *n == s).map(|i| RMS[i])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mc {
    pub precision: u32,
    pub rounding: Rm,
}

const DECIMAL32: Mc = Mc { precision: 7, rounding: Rm::HalfEven };
const DECIMAL64: Mc = Mc { precision: 16, rounding: Rm::HalfEven };
const DECIMAL128: Mc = Mc { precision: 34, rounding: Rm::HalfEven };
const UNLIMITED: Mc = Mc { precision: 0, rounding: Rm::HalfUp };

// ---------------------------------------------------------------------------
// Errors

fn arith(msg: &str) -> Box<ScriptErr> {
    ScriptErr::new("arithmetic_exception", msg)
}
fn nfe(msg: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("number_format_exception", msg)
}
fn wrong_type(msg: String) -> Box<ScriptErr> {
    ScriptErr::new("wrong_method_type_exception", msg)
}

/// Size guard (decimal digits / bits) for intermediate values. Java's own
/// limit is 2^31 bits; we fail earlier, with Java's overflow message, rather
/// than let one script allocate gigabytes.
const MAX_TEN_POW: i64 = 10_000_000;
const MAX_BITS: u64 = 1 << 27;

fn overflow() -> Box<ScriptErr> {
    arith("BigInteger would overflow supported range").frames(&["java.base/java.math.BigInteger.reportOverflow(BigInteger.java:1220)"])
}

fn ten_pow(n: i64) -> SResult<BigInt> {
    if n < 0 || n > MAX_TEN_POW {
        return Err(overflow());
    }
    Ok(BigInt::from(10u32).pow(n as u32))
}

/// Number of decimal digits of `|x|` (1 for zero).
fn digits(x: &BigInt) -> i64 {
    let m = x.magnitude();
    if let Some(v) = m.to_u64() {
        return v.checked_ilog10().map(|d| d as i64 + 1).unwrap_or(1);
    }
    // Lower bound from the bit length (1233/4096 < log10(2)), then correct.
    let bits = m.bits();
    let mut est = (((bits - 1) as u128 * 1233) >> 12) as i64;
    let mut p = num_bigint::BigUint::from(10u32).pow(est as u32 + 1);
    while *m >= p {
        est += 1;
        p *= 10u32;
    }
    est + 1
}

/// Java `checkScale` on a (possibly zero) value: saturates for zero,
/// throws otherwise.
fn check_scale(val: i64, is_zero: bool) -> SResult<i32> {
    if val > i32::MAX as i64 {
        if is_zero { Ok(i32::MAX) } else { Err(arith("Underflow")) }
    } else if val < i32::MIN as i64 {
        if is_zero { Ok(i32::MIN) } else { Err(arith("Overflow")) }
    } else {
        Ok(val as i32)
    }
}

fn saturate(val: i64) -> i32 {
    val.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Rounded quotient of `dividend / divisor` (Java `divideAndRound` without
/// the zero-stripping).
fn divide_and_round(dividend: &BigInt, divisor: &BigInt, rm: Rm) -> SResult<BigInt> {
    let (q, r) = dividend.div_rem(divisor);
    if r.is_zero() {
        return Ok(q);
    }
    let qsign = if (dividend.sign() == Sign::Minus) != (divisor.sign() == Sign::Minus) { -1 } else { 1 };
    let cmp_half = (r.abs() << 1u32).cmp(&divisor.abs());
    let inc = match rm {
        Rm::Unnecessary => return Err(arith("Rounding necessary")),
        Rm::Up => true,
        Rm::Down => false,
        Rm::Ceiling => qsign > 0,
        Rm::Floor => qsign < 0,
        _ => match cmp_half {
            Ordering::Less => false,
            Ordering::Greater => true,
            Ordering::Equal => match rm {
                Rm::HalfDown => false,
                Rm::HalfUp => true,
                _ => q.is_odd(),
            },
        },
    };
    Ok(if inc {
        if qsign > 0 { q + 1 } else { q - 1 }
    } else {
        q
    })
}

// ---------------------------------------------------------------------------
// BigDecimal

#[derive(Clone, Debug)]
pub struct Dec {
    pub u: BigInt,
    pub s: i32,
}

impl Dec {
    fn new(u: BigInt, s: i32) -> Dec {
        Dec { u, s }
    }
    fn zero(s: i32) -> Dec {
        Dec { u: BigInt::zero(), s }
    }
    fn from_i64(v: i64) -> Dec {
        Dec::new(BigInt::from(v), 0)
    }
    fn is_zero(&self) -> bool {
        self.u.is_zero()
    }
    fn signum(&self) -> i32 {
        match self.u.sign() {
            Sign::Minus => -1,
            Sign::NoSign => 0,
            Sign::Plus => 1,
        }
    }
    fn precision(&self) -> i64 {
        digits(&self.u)
    }

    /// Numeric comparison (scale-independent).
    fn cmp(&self, o: &Dec) -> SResult<Ordering> {
        let (sa, sb) = (self.signum(), o.signum());
        if sa != sb {
            return Ok(sa.cmp(&sb));
        }
        if sa == 0 {
            return Ok(Ordering::Equal);
        }
        // Quick exit on adjusted exponents.
        let ea = self.precision() - self.s as i64;
        let eb = o.precision() - o.s as i64;
        if ea != eb {
            let c = ea.cmp(&eb);
            return Ok(if sa > 0 { c } else { c.reverse() });
        }
        Ok(if self.s >= o.s {
            self.u.cmp(&(&o.u * ten_pow(self.s as i64 - o.s as i64)?))
        } else {
            (&self.u * ten_pow(o.s as i64 - self.s as i64)?).cmp(&o.u)
        })
    }
    fn cmp_magnitude(&self, o: &Dec) -> SResult<Ordering> {
        self.abs().cmp(&o.abs())
    }

    fn negate(&self) -> Dec {
        Dec::new(-&self.u, self.s)
    }
    fn abs(&self) -> Dec {
        Dec::new(self.u.abs(), self.s)
    }

    fn add(&self, o: &Dec) -> SResult<Dec> {
        let s = self.s.max(o.s);
        let a = if self.s == s { self.u.clone() } else { &self.u * ten_pow(s as i64 - self.s as i64)? };
        let b = if o.s == s { o.u.clone() } else { &o.u * ten_pow(s as i64 - o.s as i64)? };
        Ok(Dec::new(a + b, s))
    }
    fn subtract(&self, o: &Dec) -> SResult<Dec> {
        self.add(&o.negate())
    }
    fn multiply(&self, o: &Dec) -> SResult<Dec> {
        let s = check_scale(self.s as i64 + o.s as i64, self.is_zero() || o.is_zero())?;
        Ok(Dec::new(&self.u * &o.u, s))
    }

    fn set_scale(&self, new_scale: i32, rm: Rm) -> SResult<Dec> {
        if new_scale == self.s {
            return Ok(self.clone());
        }
        if self.is_zero() {
            return Ok(Dec::zero(new_scale));
        }
        if new_scale > self.s {
            let raise = check_scale(new_scale as i64 - self.s as i64, false)?;
            Ok(Dec::new(&self.u * ten_pow(raise as i64)?, new_scale))
        } else {
            let drop = check_scale(self.s as i64 - new_scale as i64, false)?;
            Ok(Dec::new(divide_and_round(&self.u, &ten_pow(drop as i64)?, rm)?, new_scale))
        }
    }

    /// Java `createAndStripZerosToMatchScale`.
    fn strip_zeros_to(mut self, preferred: i64) -> Dec {
        let ten = BigInt::from(10u32);
        while self.u.abs() >= ten && (self.s as i64) > preferred {
            if self.u.is_odd() {
                break;
            }
            let (q, r) = self.u.div_rem(&ten);
            if !r.is_zero() {
                break;
            }
            self.u = q;
            self.s -= 1;
        }
        self
    }

    /// Java `doRound`: round to `mc.precision` significant digits.
    fn do_round(self, mc: &Mc) -> SResult<Dec> {
        let mcp = mc.precision as i64;
        if mcp == 0 {
            return Ok(self);
        }
        let mut u = self.u;
        let mut s = self.s as i64;
        let mut drop = digits(&u) - mcp;
        while drop > 0 {
            s = check_scale(s - drop, false)? as i64;
            u = divide_and_round(&u, &ten_pow(drop)?, mc.rounding)?;
            drop = digits(&u) - mcp;
        }
        Ok(Dec::new(u, s as i32))
    }

    fn strip_trailing_zeros(&self) -> Dec {
        if self.is_zero() {
            return Dec::zero(0);
        }
        self.clone().strip_zeros_to(i32::MIN as i64)
    }

    fn add_mc(&self, o: &Dec, mc: &Mc) -> SResult<Dec> {
        let mcp = mc.precision as i64;
        if mcp == 0 {
            return self.add(o);
        }
        let (lz, rz) = (self.is_zero(), o.is_zero());
        if lz || rz {
            let preferred = self.s.max(o.s);
            if lz && rz {
                return Ok(Dec::zero(preferred));
            }
            let result = if lz { o.clone() } else { self.clone() }.do_round(mc)?;
            return Ok(if result.s == preferred {
                result
            } else if result.s > preferred {
                result.strip_zeros_to(preferred as i64)
            } else {
                let precision_diff = mcp - result.precision();
                let scale_diff = preferred as i64 - result.s as i64;
                if precision_diff >= scale_diff {
                    result.set_scale(preferred, Rm::Unnecessary)?
                } else {
                    let ns = result.s as i64 + precision_diff;
                    result.set_scale(ns as i32, Rm::Unnecessary)?
                }
            });
        }
        let padding = self.s as i64 - o.s as i64;
        let sum = if padding != 0 {
            // Java `preAlign`.
            let (big, small) = if padding < 0 { (self, o) } else { (o, self) };
            let est_result_ulp_scale = big.s as i64 - big.precision() + mcp;
            let small_high_digit_pos = small.s as i64 - small.precision() + 1;
            let small = if small_high_digit_pos > big.s as i64 + 2 && small_high_digit_pos > est_result_ulp_scale + 2 {
                Dec::new(BigInt::from(small.signum()), check_scale((big.s as i64).max(est_result_ulp_scale) + 3, false)?)
            } else {
                small.clone()
            };
            big.add(&small)?
        } else {
            self.add(o)?
        };
        sum.do_round(mc)
    }

    fn subtract_mc(&self, o: &Dec, mc: &Mc) -> SResult<Dec> {
        if mc.precision == 0 {
            return self.subtract(o);
        }
        self.add_mc(&o.negate(), mc)
    }

    fn multiply_mc(&self, o: &Dec, mc: &Mc) -> SResult<Dec> {
        self.multiply(o)?.do_round(mc)
    }

    fn check_div(&self, o: &Dec) -> SResult<()> {
        if o.is_zero() {
            return Err(arith(if self.is_zero() { "Division undefined" } else { "Division by zero" }));
        }
        Ok(())
    }

    /// Java `divide(BigDecimal, MathContext)`.
    fn divide_mc(&self, o: &Dec, mc: &Mc) -> SResult<Dec> {
        let mcp = mc.precision as i64;
        if mcp == 0 {
            return self.divide(o);
        }
        let preferred = self.s as i64 - o.s as i64;
        self.check_div(o)?;
        if self.is_zero() {
            return Ok(Dec::zero(saturate(preferred)));
        }
        let xp = self.precision();
        let mut yp = o.precision();
        // compareMagnitudeNormalized
        let sdiff = xp - yp;
        let cmp = if sdiff < 0 {
            (self.u.abs() * ten_pow(-sdiff)?).cmp(&o.u.abs())
        } else {
            self.u.abs().cmp(&(o.u.abs() * ten_pow(sdiff)?))
        };
        if cmp == Ordering::Greater {
            yp -= 1;
        }
        let scl = check_scale(preferred + yp - xp + mcp, false)?;
        let raise = check_scale(mcp + yp - xp, false)? as i64;
        let pref = check_scale(preferred, false)?;
        let q = if raise > 0 {
            divide_and_round_strip(&(&self.u * ten_pow(raise)?), &o.u, scl, mc.rounding, pref)?
        } else {
            let new_scale = check_scale(xp - mcp, false)? as i64;
            let raise = check_scale(new_scale - yp, false)? as i64;
            divide_and_round_strip(&self.u, &(&o.u * ten_pow(raise)?), scl, mc.rounding, pref)?
        };
        q.do_round(mc)
    }

    /// Java `divide(BigDecimal)`: exact quotient or
    /// "Non-terminating decimal expansion".
    fn divide(&self, o: &Dec) -> SResult<Dec> {
        self.check_div(o)?;
        let preferred = saturate(self.s as i64 - o.s as i64);
        if self.is_zero() {
            return Ok(Dec::zero(preferred));
        }
        let mcp = (self.precision() + (10 * o.precision() + 2) / 3).min(i32::MAX as i64) as u32;
        let q = match self.divide_mc(o, &Mc { precision: mcp, rounding: Rm::Unnecessary }) {
            Ok(q) => q,
            Err(e) if e.kind == "arithmetic_exception" => return Err(arith("Non-terminating decimal expansion; no exact representable decimal result.")),
            Err(e) => return Err(e),
        };
        if preferred > q.s {
            return q.set_scale(preferred, Rm::Unnecessary);
        }
        Ok(q)
    }

    fn divide_to_integral(&self, o: &Dec) -> SResult<Dec> {
        let preferred = saturate(self.s as i64 - o.s as i64);
        if self.cmp_magnitude(o)? == Ordering::Less {
            return Ok(Dec::zero(preferred));
        }
        if self.is_zero() && !o.is_zero() {
            return self.set_scale(preferred, Rm::Unnecessary);
        }
        let max_digits = (self.precision() + (10 * o.precision() + 2) / 3 + (self.s as i64 - o.s as i64).abs() + 2).min(i32::MAX as i64) as u32;
        let mut q = self.divide_mc(o, &Mc { precision: max_digits, rounding: Rm::Down })?;
        if q.s > 0 {
            q = q.set_scale(0, Rm::Down)?;
            q = q.strip_zeros_to(preferred as i64);
        }
        if q.s < preferred {
            q = q.set_scale(preferred, Rm::Unnecessary)?;
        }
        Ok(q)
    }

    fn divide_to_integral_mc(&self, o: &Dec, mc: &Mc) -> SResult<Dec> {
        if mc.precision == 0 || self.cmp_magnitude(o)? == Ordering::Less {
            return self.divide_to_integral(o);
        }
        let preferred = saturate(self.s as i64 - o.s as i64);
        let mut result = self.divide_mc(o, &Mc { precision: mc.precision, rounding: Rm::Down })?;
        if result.s < 0 {
            let product = result.multiply(o)?;
            if self.subtract(&product)?.cmp_magnitude(o)? != Ordering::Less {
                return Err(arith("Division impossible"));
            }
        } else if result.s > 0 {
            result = result.set_scale(0, Rm::Down)?;
        }
        let precision_diff = mc.precision as i64 - result.precision();
        if preferred > result.s && precision_diff > 0 {
            let ns = result.s as i64 + precision_diff.min(preferred as i64 - result.s as i64);
            result.set_scale(ns as i32, Rm::Unnecessary)
        } else {
            Ok(result.strip_zeros_to(preferred as i64))
        }
    }

    fn divide_and_remainder(&self, o: &Dec, mc: Option<&Mc>) -> SResult<(Dec, Dec)> {
        let q = match mc {
            Some(mc) => self.divide_to_integral_mc(o, mc)?,
            None => self.divide_to_integral(o)?,
        };
        let r = self.subtract(&q.multiply(o)?)?;
        Ok((q, r))
    }

    fn pow(&self, n: i32) -> SResult<Dec> {
        if !(0..=999_999_999).contains(&n) {
            return Err(arith("Invalid operation"));
        }
        let ns = check_scale(self.s as i64 * n as i64, self.is_zero())?;
        if self.precision() * n as i64 > MAX_TEN_POW {
            return Err(overflow());
        }
        Ok(Dec::new(self.u.pow(n as u32), ns))
    }

    fn pow_mc(&self, n: i32, mc: &Mc) -> SResult<Dec> {
        if mc.precision == 0 {
            return self.pow(n);
        }
        if !(-999_999_999..=999_999_999).contains(&n) {
            return Err(arith("Invalid operation"));
        }
        if n == 0 {
            return Ok(Dec::from_i64(1));
        }
        let mut mag: i32 = n.abs();
        let elength = digits(&BigInt::from(mag));
        if elength > mc.precision as i64 {
            return Err(arith("Invalid operation"));
        }
        let workmc = Mc { precision: (mc.precision as i64 + elength + 1) as u32, rounding: mc.rounding };
        let mut acc = Dec::from_i64(1);
        let mut seenbit = false;
        for i in 1.. {
            mag = mag.wrapping_add(mag);
            if mag < 0 {
                seenbit = true;
                acc = acc.multiply_mc(self, &workmc)?;
            }
            if i == 31 {
                break;
            }
            if seenbit {
                acc = acc.clone().multiply_mc(&acc, &workmc)?;
            }
        }
        if n < 0 {
            acc = Dec::from_i64(1).divide_mc(&acc, &workmc)?;
        }
        acc.do_round(mc)
    }

    fn move_point_left(&self, n: i32) -> SResult<Dec> {
        if n == 0 {
            return Ok(self.clone());
        }
        let ns = check_scale(self.s as i64 + n as i64, self.is_zero())?;
        let num = Dec::new(self.u.clone(), ns);
        if num.s < 0 { num.set_scale(0, Rm::Unnecessary) } else { Ok(num) }
    }
    fn move_point_right(&self, n: i32) -> SResult<Dec> {
        if n == 0 {
            return Ok(self.clone());
        }
        let ns = check_scale(self.s as i64 - n as i64, self.is_zero())?;
        let num = Dec::new(self.u.clone(), ns);
        if num.s < 0 { num.set_scale(0, Rm::Unnecessary) } else { Ok(num) }
    }
    fn scale_by_power_of_ten(&self, n: i32) -> SResult<Dec> {
        Ok(Dec::new(self.u.clone(), check_scale(self.s as i64 - n as i64, self.is_zero())?))
    }

    fn to_big_integer(&self) -> SResult<BigInt> {
        if self.s <= 0 {
            Ok(&self.u * ten_pow(-(self.s as i64))?)
        } else {
            Ok(&self.u / ten_pow(self.s as i64)?)
        }
    }
    fn to_big_integer_exact(&self) -> SResult<BigInt> {
        self.set_scale(0, Rm::Unnecessary)?.to_big_integer()
    }

    fn long_value_exact(&self) -> SResult<i64> {
        if self.s == 0 {
            return self.u.to_i64().ok_or_else(|| arith("Overflow"));
        }
        if self.is_zero() {
            return Ok(0);
        }
        if self.precision() - self.s as i64 <= 0 {
            return Err(arith("Rounding necessary"));
        }
        if self.precision() - self.s as i64 > 19 {
            return Err(arith("Overflow"));
        }
        let num = self.set_scale(0, Rm::Unnecessary)?;
        num.u.to_i64().ok_or_else(|| arith("Overflow"))
    }

    fn to_f64(&self) -> f64 {
        format!("{}E{}", self.u, -(self.s as i64)).parse::<f64>().unwrap_or(0.0)
    }
    fn to_f32(&self) -> f32 {
        format!("{}E{}", self.u, -(self.s as i64)).parse::<f32>().unwrap_or(0.0)
    }

    fn hash(&self) -> i32 {
        bigint_hash(&self.u).wrapping_mul(31).wrapping_add(self.s)
    }

    /// Java `layoutChars(sci)`: `toString` (sci) / `toEngineeringString`.
    fn layout(&self, sci: bool) -> String {
        if self.s == 0 {
            return self.u.to_string();
        }
        let coeff = self.u.magnitude().to_string();
        let clen = coeff.len() as i64;
        let mut buf = String::with_capacity(coeff.len() + 14);
        if self.u.sign() == Sign::Minus {
            buf.push('-');
        }
        let mut adjusted = -(self.s as i64) + (clen - 1);
        if self.s >= 0 && adjusted >= -6 {
            let pad = self.s as i64 - clen;
            if pad >= 0 {
                buf.push_str("0.");
                for _ in 0..pad {
                    buf.push('0');
                }
                buf.push_str(&coeff);
            } else {
                let (a, b) = coeff.split_at((clen - self.s as i64) as usize);
                buf.push_str(a);
                buf.push('.');
                buf.push_str(b);
            }
            return buf;
        }
        if sci {
            buf.push_str(&coeff[..1]);
            if clen > 1 {
                buf.push('.');
                buf.push_str(&coeff[1..]);
            }
        } else {
            let mut sig = adjusted % 3;
            if sig < 0 {
                sig += 3;
            }
            adjusted -= sig;
            sig += 1;
            if self.is_zero() {
                match sig {
                    1 => buf.push('0'),
                    2 => {
                        buf.push_str("0.00");
                        adjusted += 3;
                    }
                    _ => {
                        buf.push_str("0.0");
                        adjusted += 3;
                    }
                }
            } else if sig >= clen {
                buf.push_str(&coeff);
                for _ in 0..(sig - clen) {
                    buf.push('0');
                }
            } else {
                buf.push_str(&coeff[..sig as usize]);
                buf.push('.');
                buf.push_str(&coeff[sig as usize..]);
            }
        }
        if adjusted != 0 {
            buf.push('E');
            if adjusted > 0 {
                buf.push('+');
            }
            let _ = write!(buf, "{adjusted}");
        }
        buf
    }

    fn to_plain_string(&self) -> String {
        if self.s == 0 {
            return self.u.to_string();
        }
        if self.s < 0 {
            if self.is_zero() {
                return "0".into();
            }
            let mut s = self.u.to_string();
            for _ in 0..(-(self.s as i64)) {
                s.push('0');
            }
            return s;
        }
        let int_string = self.u.magnitude().to_string();
        let neg = self.u.sign() == Sign::Minus;
        let insertion = int_string.len() as i64 - self.s as i64;
        if insertion == 0 {
            format!("{}{}", if neg { "-0." } else { "0." }, int_string)
        } else if insertion > 0 {
            let (a, b) = int_string.split_at(insertion as usize);
            format!("{}{a}.{b}", if neg { "-" } else { "" })
        } else {
            let mut s = String::from(if neg { "-0." } else { "0." });
            for _ in 0..(-insertion) {
                s.push('0');
            }
            s.push_str(&int_string);
            s
        }
    }
}

/// Java `divideAndRound(BigInteger, BigInteger, scale, rm, preferredScale)`.
fn divide_and_round_strip(dividend: &BigInt, divisor: &BigInt, scale: i32, rm: Rm, preferred: i32) -> SResult<Dec> {
    let (q, r) = dividend.div_rem(divisor);
    if r.is_zero() {
        let d = Dec::new(q, scale);
        return Ok(if preferred != scale { d.strip_zeros_to(preferred as i64) } else { d });
    }
    Ok(Dec::new(divide_and_round(dividend, divisor, rm)?, scale))
}

/// Parse with Java `BigDecimal(String)` rules and error messages.
fn parse_decimal(s: &str) -> SResult<Dec> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    if chars.is_empty() {
        // Java: ArrayIndexOutOfBounds → NumberFormatException with no message.
        return Err(nfe("").frames(&[BD_INIT_721]));
    }
    let mut neg = false;
    if chars[0] == '-' {
        neg = true;
        i = 1;
    } else if chars[0] == '+' {
        i = 1;
    }
    let mut dot = false;
    let mut scl: i64 = 0;
    let mut exp: i64 = 0;
    let mut coeff = String::new();
    while i < chars.len() {
        let c = chars[i];
        if let Some(d) = java_digit(c) {
            coeff.push(char::from(b'0' + d as u8));
            if dot {
                scl += 1;
            }
        } else if c == '.' {
            if dot {
                return Err(nfe("Character array contains more than one decimal point.").frames(&[bd!("<init>", 603)]));
            }
            dot = true;
        } else if c == 'e' || c == 'E' {
            exp = parse_exp(&chars, i)?;
            break;
        } else {
            return Err(nfe(format!("Character {c} is neither a decimal digit number, decimal point, nor \"e\" notation exponential mark.")).frames(&[bd!("<init>", 626)]));
        }
        i += 1;
    }
    if coeff.is_empty() {
        return Err(nfe("No digits found.").frames(&[bd!("<init>", 632)]));
    }
    if exp != 0 {
        let adjusted = scl - exp;
        if adjusted > i32::MAX as i64 || adjusted < i32::MIN as i64 {
            return Err(nfe("Exponent overflow.").frames(&[bd!("<init>", 726)]));
        }
        scl = adjusted;
    }
    let mut u = coeff.parse::<BigInt>().map_err(|_| nfe("No digits found."))?;
    if neg {
        u = -u;
    }
    Ok(Dec::new(u, scl as i32))
}

/// Java `parseExp`: `at` is the index of the `e`/`E`.
fn parse_exp(chars: &[char], at: usize) -> SResult<i64> {
    let mut off = at + 1;
    let mut len = chars.len() as i64 - off as i64; // chars remaining after 'e'
    let mut c = *chars.get(off).ok_or_else(|| nfe("").frames(&[BD_INIT_721]))?;
    let negexp = c == '-';
    if negexp || c == '+' {
        off += 1;
        len -= 1;
        c = *chars.get(off).ok_or_else(|| nfe("").frames(&[BD_INIT_721]))?;
    }
    if len <= 0 {
        return Err(nfe("No exponent digits.").frames(&[bd!("parseExp", 751), bd!("<init>", 623)]));
    }
    while len > 10 && java_digit(c) == Some(0) {
        off += 1;
        len -= 1;
        c = chars[off];
    }
    if len > 10 {
        return Err(nfe("Too many nonzero exponent digits.").frames(&[bd!("parseExp", 757), bd!("<init>", 623)]));
    }
    let mut exp: i64 = 0;
    loop {
        let v = java_digit(c).ok_or_else(|| nfe("Not a digit.").frames(&[bd!("parseExp", 766), bd!("<init>", 623)]))?;
        exp = exp * 10 + v as i64;
        len -= 1;
        if len == 0 {
            break;
        }
        off += 1;
        c = chars[off];
    }
    Ok(if negexp { -exp } else { exp })
}

/// Zero code points of the Unicode `Nd` digit blocks (`Character.digit`).
const DIGIT_ZEROS: &[u32] = &[
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66, 0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946,
    0x19D0, 0x1A80, 0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x11066, 0x110F0,
    0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650, 0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x16A60, 0x16B50, 0x1D7CE, 0x1D7D8,
    0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E950, 0x1FBF0,
];

/// `Character.digit(c, 10)`.
fn java_digit(c: char) -> Option<u32> {
    java_digit_radix(c, 10)
}

/// `Character.digit(c, radix)`: Unicode decimal digits and (full-width)
/// Latin letters.
fn java_digit_radix(c: char, radix: u32) -> Option<u32> {
    let cp = c as u32;
    let v = if c.is_ascii_alphanumeric() {
        c.to_digit(36)?
    } else if (0xFF21..=0xFF3A).contains(&cp) {
        cp - 0xFF21 + 10
    } else if (0xFF41..=0xFF5A).contains(&cp) {
        cp - 0xFF41 + 10
    } else {
        let z = *DIGIT_ZEROS.iter().rev().find(|z| **z <= cp)?;
        if cp - z > 9 {
            return None;
        }
        cp - z
    };
    (v < radix).then_some(v)
}

// ---------------------------------------------------------------------------
// BigInteger helpers

/// Java `BigInteger.hashCode()`.
fn bigint_hash(x: &BigInt) -> i32 {
    let mut h: i32 = 0;
    let words = x.magnitude().to_u32_digits();
    for w in words.iter().rev() {
        h = h.wrapping_mul(31).wrapping_add(*w as i32);
    }
    match x.sign() {
        Sign::Minus => h.wrapping_neg(),
        _ => h,
    }
}

/// Low 64 bits, two's complement (Java `longValue()`).
fn low_i64(x: &BigInt) -> i64 {
    let low = x.magnitude().iter_u64_digits().next().unwrap_or(0);
    match x.sign() {
        Sign::Minus => low.wrapping_neg() as i64,
        _ => low as i64,
    }
}

/// Java `shiftRight` (floor semantics for negatives).
fn shr(x: &BigInt, n: u64) -> BigInt {
    if x.sign() == Sign::Minus {
        !(&!x >> n)
    } else {
        x >> n
    }
}

fn shl(x: &BigInt, n: u64) -> SResult<BigInt> {
    if x.is_zero() {
        return Ok(BigInt::zero());
    }
    if n + x.bits() > MAX_BITS {
        return Err(overflow());
    }
    Ok(x << n)
}

fn bit_length(x: &BigInt) -> i64 {
    if x.is_zero() {
        return 0;
    }
    let bits = x.bits() as i64;
    if x.sign() == Sign::Minus && x.magnitude().count_ones() == 1 { bits - 1 } else { bits }
}

fn bit_count(x: &BigInt) -> i64 {
    let bc = x.magnitude().count_ones() as i64;
    if x.sign() == Sign::Minus {
        bc + x.magnitude().trailing_zeros().unwrap_or(0) as i64 - 1
    } else {
        bc
    }
}

fn parse_big_integer(s: &str, radix: u32) -> SResult<BigInt> {
    if !(2..=36).contains(&radix) {
        return Err(nfe("Radix out of range").frames(&[BI_INIT_531]));
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return Err(nfe("Zero length BigInteger").frames(&[BI_INIT_533]));
    }
    let minus = chars.iter().rposition(|c| *c == '-');
    let plus = chars.iter().rposition(|c| *c == '+');
    let mut cursor = 0;
    let mut neg = false;
    if let Some(m) = minus {
        if plus.is_some() || m != 0 {
            return Err(nfe("Illegal embedded sign character").frames(&[BI_INIT_541]));
        }
        neg = true;
        cursor = 1;
    } else if let Some(p) = plus {
        if p != 0 {
            return Err(nfe("Illegal embedded sign character").frames(&[BI_INIT_547]));
        }
        cursor = 1;
    }
    if cursor == chars.len() {
        return Err(nfe("Zero length BigInteger").frames(&[BI_INIT_552]));
    }
    while cursor < chars.len() && java_digit_radix(chars[cursor], radix) == Some(0) {
        cursor += 1;
    }
    if cursor == chars.len() {
        return Ok(BigInt::zero());
    }
    // Java parses the digits in groups (`digitsPerInt[radix]`) with
    // `Integer.parseInt`; the error names the offending group.
    const DIGITS_PER_INT: [usize; 37] = [0, 0, 30, 19, 15, 13, 11, 11, 10, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5];
    let per = DIGITS_PER_INT[radix as usize];
    let digits: Vec<char> = chars[cursor..].to_vec();
    let n = digits.len();
    let mut first = n % per;
    if first == 0 {
        first = per;
    }
    let mut result = BigInt::zero();
    let mut pos = 0;
    let mut group_len = first;
    while pos < n {
        let group: String = digits[pos..pos + group_len].iter().collect();
        let v = group.chars().try_fold(0u64, |acc, c| java_digit_radix(c, radix).map(|d| acc * radix as u64 + d as u64)).ok_or_else(|| {
            let msg = if radix == 10 { format!("For input string: \"{group}\"") } else { format!("For input string: \"{group}\" under radix {radix}") };
            nfe(msg).frames(if pos == 0 { BI_PARSE_FIRST } else { BI_PARSE_GROUP })
        })?;
        result = result * BigInt::from(radix).pow(group_len as u32) + BigInt::from(v);
        pos += group_len;
        group_len = per;
    }
    Ok(if neg { -result } else { result })
}

fn big_to_f64(x: &BigInt) -> f64 {
    x.to_string().parse::<f64>().unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Value

#[derive(Debug)]
pub enum Big {
    Dec(Dec),
    Int(BigInt),
    MathCtx(Mc),
    Rounding(Rm),
}

fn dec(d: Dec) -> Val {
    Val::Big(Rc::new(Big::Dec(d)))
}
fn bint(i: BigInt) -> Val {
    Val::Big(Rc::new(Big::Int(i)))
}

/// A `BigInteger` holding an unsigned 64-bit value.
pub fn biginteger_u64(v: u64) -> Val {
    bint(BigInt::from(v))
}
fn mctx(m: Mc) -> Val {
    Val::Big(Rc::new(Big::MathCtx(m)))
}
fn rmode(r: Rm) -> Val {
    Val::Big(Rc::new(Big::Rounding(r)))
}

impl Big {
    pub fn type_name(&self) -> &'static str {
        match self {
            Big::Dec(_) => "java.math.BigDecimal",
            Big::Int(_) => "java.math.BigInteger",
            Big::MathCtx(_) => "java.math.MathContext",
            Big::Rounding(_) => "java.math.RoundingMode",
        }
    }
    fn simple_name(&self) -> &'static str {
        match self {
            Big::Dec(_) => "BigDecimal",
            Big::Int(_) => "BigInteger",
            Big::MathCtx(_) => "MathContext",
            Big::Rounding(_) => "RoundingMode",
        }
    }
    pub fn write_java(&self, out: &mut String) {
        match self {
            Big::Dec(d) => out.push_str(&d.layout(true)),
            Big::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Big::MathCtx(m) => {
                let _ = write!(out, "precision={} roundingMode={}", m.precision, m.rounding.name());
            }
            Big::Rounding(r) => out.push_str(r.name()),
        }
    }
    pub fn hash(&self) -> i32 {
        match self {
            Big::Dec(d) => d.hash(),
            Big::Int(i) => bigint_hash(i),
            Big::MathCtx(m) => (m.precision as i32).wrapping_add(RM_HASHES[m.rounding as usize].wrapping_mul(59)),
            Big::Rounding(r) => RM_HASHES[*r as usize],
        }
    }
    pub fn equals(&self, o: &Big) -> bool {
        match (self, o) {
            (Big::Dec(a), Big::Dec(b)) => a.s == b.s && a.u == b.u,
            (Big::Int(a), Big::Int(b)) => a == b,
            (Big::MathCtx(a), Big::MathCtx(b)) => a == b,
            (Big::Rounding(a), Big::Rounding(b)) => a == b,
            _ => false,
        }
    }
    pub fn compare(&self, o: &Big) -> SResult<Ordering> {
        match (self, o) {
            (Big::Dec(a), Big::Dec(b)) => a.cmp(b),
            (Big::Int(a), Big::Int(b)) => Ok(a.cmp(b)),
            (Big::Rounding(a), Big::Rounding(b)) => Ok((*a as u8).cmp(&(*b as u8))),
            (Big::MathCtx(_), _) => Err(class_cast(format!("Cannot cast {} to java.lang.Comparable", self.type_name()))),
            _ => Err(class_cast(format!("Cannot cast {} to {}", o.type_name(), self.type_name()))),
        }
    }
    /// `Number` view (`doubleValue()`), for numeric contexts.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Big::Dec(d) => Some(d.to_f64()),
            Big::Int(i) => Some(big_to_f64(i)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Argument conversion (Painless def-call semantics)

/// Java simple class name of a value, as printed in `MethodHandle` types.
fn simple_type(v: &Val) -> String {
    match v {
        Val::Int(_) => "int".into(),
        Val::Long(_) => "long".into(),
        Val::Double(_) => "double".into(),
        Val::Bool(_) => "boolean".into(),
        Val::Char(_) => "char".into(),
        Val::Null => "Object".into(),
        other => other.type_name().rsplit(['.', '$']).next().unwrap_or("Object").to_string(),
    }
}

fn null_value_npe() -> Box<ScriptErr> {
    npe("Cannot invoke \"Object.getClass()\" because \"value\" is null")
}

/// `def` → `int` implicit cast (constructor / static arguments).
fn def_int(v: &Val) -> SResult<i32> {
    match v {
        Val::Int(i) => Ok(*i),
        Val::Char(c) => Ok(*c as i32),
        Val::Null => Err(null_value_npe()),
        other => Err(class_cast(format!("cannot implicitly cast def [{}] to int", other.def_name()))),
    }
}
fn def_long(v: &Val) -> SResult<i64> {
    match v {
        Val::Int(i) => Ok(*i as i64),
        Val::Long(l) => Ok(*l),
        Val::Char(c) => Ok(*c as i64),
        Val::Null => Err(null_value_npe()),
        other => Err(class_cast(format!("cannot implicitly cast def [{}] to long", other.def_name()))),
    }
}
fn def_double(v: &Val) -> SResult<f64> {
    match v {
        Val::Int(i) => Ok(*i as f64),
        Val::Long(l) => Ok(*l as f64),
        Val::Double(d) => Ok(*d),
        Val::Char(c) => Ok(*c as u32 as f64),
        Val::Null => Err(null_value_npe()),
        other => Err(class_cast(format!("cannot implicitly cast def [{}] to double", other.def_name()))),
    }
}
/// `def` → `java.lang.String` implicit cast; a null gets the JDK's helpful
/// NPE (message + frames) of the method that first dereferences the string.
fn def_string<'a>(v: &'a Val, null_msg: &str, null_frames: Frames) -> SResult<&'a str> {
    match v {
        Val::Str(s) => Ok(s),
        Val::Null => Err(npe(null_msg).frames(null_frames)),
        other => Err(class_cast(format!("cannot implicitly cast def [{}] to java.lang.String", other.def_name()))),
    }
}
/// Reference-typed constructor argument: JVM `checkcast` failure message.
fn ctor_cast_msg(v: &Val, target: &str) -> Box<ScriptErr> {
    let a = v.type_name();
    class_cast(format!("class {a} cannot be cast to class {target} ({a} and {target} are in module java.base of loader 'bootstrap')"))
}

/// Parameter kinds of the whitelisted `java.math` methods.
#[derive(Clone, Copy, PartialEq, Eq)]
enum P {
    /// `int`
    I,
    /// `BigDecimal`
    D,
    /// `BigInteger`
    B,
    /// `MathContext`
    M,
    /// `RoundingMode`
    R,
    /// `Object`
    O,
}

impl P {
    fn name(self) -> &'static str {
        match self {
            P::I => "int",
            P::D => "BigDecimal",
            P::B => "BigInteger",
            P::M => "MathContext",
            P::R => "RoundingMode",
            P::O => "Object",
        }
    }
    fn java_name(self) -> &'static str {
        match self {
            P::I => "int",
            P::D => "java.math.BigDecimal",
            P::B => "java.math.BigInteger",
            P::M => "java.math.MathContext",
            P::R => "java.math.RoundingMode",
            P::O => "java.lang.Object",
        }
    }
    fn accepts(self, v: &Val) -> bool {
        match (self, v) {
            (P::I, Val::Int(_) | Val::Char(_)) => true,
            (P::D, Val::Big(b)) => matches!(**b, Big::Dec(_)),
            (P::B, Val::Big(b)) => matches!(**b, Big::Int(_)),
            (P::M, Val::Big(b)) => matches!(**b, Big::MathCtx(_)),
            (P::R, Val::Big(b)) => matches!(**b, Big::Rounding(_)),
            (P::O, _) => true,
            _ => false,
        }
    }
}

/// Signature of a whitelisted instance method: parameter kinds and the
/// return type (as printed in `MethodHandle` types).
fn sig(recv: &Big, name: &str, n: usize) -> Option<(&'static [P], &'static str)> {
    use P::*;
    let common: Option<(&'static [P], &'static str)> = match (name, n) {
        ("toString", 0) => Some((&[], "String")),
        ("hashCode", 0) => Some((&[], "int")),
        ("equals", 1) => Some((&[O], "boolean")),
        _ => None,
    };
    if common.is_some() {
        return common;
    }
    let number: Option<(&'static [P], &'static str)> = match (name, n) {
        ("byteValue", 0) => Some((&[], "byte")),
        ("shortValue", 0) => Some((&[], "short")),
        ("intValue", 0) => Some((&[], "int")),
        ("longValue", 0) => Some((&[], "long")),
        ("floatValue", 0) => Some((&[], "float")),
        ("doubleValue", 0) => Some((&[], "double")),
        _ => None,
    };
    match recv {
        Big::Dec(_) => {
            if number.is_some() {
                return number;
            }
            Some(match (name, n) {
                ("abs" | "negate" | "plus" | "stripTrailingZeros" | "ulp", 0) => (&[], "BigDecimal"),
                ("abs" | "negate" | "plus" | "round", 1) => (&[M], "BigDecimal"),
                ("add" | "subtract" | "multiply" | "divide" | "divideToIntegralValue" | "remainder" | "max" | "min", 1) => (&[D], "BigDecimal"),
                ("add" | "subtract" | "multiply" | "divide" | "divideToIntegralValue" | "remainder", 2) => (&[D, M], "BigDecimal"),
                ("divideAndRemainder", 1) => (&[D], "BigDecimal[]"),
                ("divideAndRemainder", 2) => (&[D, M], "BigDecimal[]"),
                ("compareTo", 1) => (&[D], "int"),
                ("byteValueExact", 0) => (&[], "byte"),
                ("shortValueExact", 0) => (&[], "short"),
                ("intValueExact" | "precision" | "scale" | "signum", 0) => (&[], "int"),
                ("longValueExact", 0) => (&[], "long"),
                ("movePointLeft" | "movePointRight" | "pow" | "scaleByPowerOfTen" | "setScale", 1) => (&[I], "BigDecimal"),
                ("pow", 2) => (&[I, M], "BigDecimal"),
                ("setScale", 2) => (&[I, R], "BigDecimal"),
                ("toBigInteger" | "toBigIntegerExact", 0) => (&[], "BigInteger"),
                ("toEngineeringString" | "toPlainString", 0) => (&[], "String"),
                _ => return None,
            })
        }
        Big::Int(_) => {
            if number.is_some() {
                return number;
            }
            Some(match (name, n) {
                ("abs" | "negate" | "not", 0) => (&[], "BigInteger"),
                ("add" | "subtract" | "multiply" | "divide" | "remainder" | "mod" | "modInverse" | "gcd" | "and" | "or" | "xor" | "andNot" | "max" | "min", 1) => (&[B], "BigInteger"),
                ("modPow", 2) => (&[B, B], "BigInteger"),
                ("divideAndRemainder", 1) => (&[B], "BigInteger[]"),
                ("compareTo", 1) => (&[B], "int"),
                ("bitCount" | "bitLength" | "getLowestSetBit" | "intValueExact" | "signum", 0) => (&[], "int"),
                ("byteValueExact", 0) => (&[], "byte"),
                ("shortValueExact", 0) => (&[], "short"),
                ("longValueExact", 0) => (&[], "long"),
                ("clearBit" | "flipBit" | "setBit" | "pow" | "shiftLeft" | "shiftRight", 1) => (&[I], "BigInteger"),
                ("testBit", 1) => (&[I], "boolean"),
                ("toByteArray", 0) => (&[], "byte[]"),
                ("toString", 1) => (&[I], "String"),
                _ => return None,
            })
        }
        Big::MathCtx(_) => Some(match (name, n) {
            ("getPrecision", 0) => (&[], "int"),
            ("getRoundingMode", 0) => (&[], "RoundingMode"),
            _ => return None,
        }),
        Big::Rounding(_) => Some(match (name, n) {
            ("name", 0) => (&[], "String"),
            ("ordinal", 0) => (&[], "int"),
            ("compareTo", 1) => (&[R], "int"),
            _ => return None,
        }),
    }
}

/// Java "helpful NullPointerException" message for a null reference argument.
fn null_arg_msg(recv: &Big, name: &str, n: usize, idx: usize) -> String {
    let f = |field: &str, var: &str| format!("Cannot read field \"{field}\" because \"{var}\" is null");
    let m = |meth: &str, var: &str| format!("Cannot invoke \"{meth}\" because \"{var}\" is null");
    match recv {
        Big::Dec(_) => match (name, n, idx) {
            ("add", 1, _) => f("intCompact", "augend"),
            ("add", 2, 0) => m("java.math.BigDecimal.signum()", "augend"),
            ("subtract", 1, _) => f("intCompact", "subtrahend"),
            ("subtract", 2, 0) => m("java.math.BigDecimal.negate()", "subtrahend"),
            ("multiply", _, 0) => f("scale", "multiplicand"),
            ("divide", 1, _) => m("java.math.BigDecimal.signum()", "divisor"),
            ("divide", 2, 0) => f("scale", "divisor"),
            ("divideToIntegralValue" | "remainder" | "divideAndRemainder", 1, _) => f("scale", "divisor"),
            ("divideToIntegralValue" | "remainder" | "divideAndRemainder", 2, 0) => f("intCompact", "val"),
            ("compareTo" | "max" | "min", _, _) => f("scale", "val"),
            ("setScale", _, _) => f("oldMode", "roundingMode"),
            (_, _, _) => f("precision", "mc"),
        },
        Big::Int(_) => match (name, idx) {
            ("divide" | "remainder" | "divideAndRemainder", _) => f("mag", "val"),
            ("mod" | "modInverse", _) => f("signum", "m"),
            ("modPow", 0) => f("signum", "exponent"),
            ("modPow", 1) => f("signum", "m"),
            ("and" | "or" | "xor" | "andNot", _) => m("java.math.BigInteger.intLength()", "val"),
            _ => f("signum", "val"),
        },
        Big::Rounding(_) => m("Object.getClass()", "other"),
        Big::MathCtx(_) => m("Object.getClass()", "value"),
    }
}

/// Validate arguments against the method signature with Painless def-call
/// semantics (primitive mismatch → `wrong_method_type_exception`, reference
/// mismatch → `class_cast_exception`, null → NPE).
fn check_args(recv: &Big, name: &str, ps: &[P], ret: &str, args: &[Val]) -> SResult<()> {
    // Primitive parameters are converted by `MethodHandle.asType`: any
    // mismatch there fails before the call.
    for (p, a) in ps.iter().zip(args) {
        let primitive = matches!(a, Val::Int(_) | Val::Long(_) | Val::Double(_) | Val::Bool(_) | Val::Char(_));
        if (*p == P::I && !p.accepts(a)) || (*p != P::I && *p != P::O && primitive) {
            if a.is_null() {
                return Err(npe("Cannot invoke \"java.lang.Number.intValue()\" because the return value of \"sun.invoke.util.ValueConversions.primitiveConversion(sun.invoke.util.Wrapper, Object, boolean)\" is null"));
            }
            let params: Vec<&str> = ps.iter().map(|p| p.name()).collect();
            let actual: Vec<String> = args.iter().map(simple_type).collect();
            return Err(wrong_type(format!(
                "cannot convert MethodHandle({},{}){} to (Object,{})Object",
                recv.simple_name(),
                params.join(","),
                ret,
                actual.join(",")
            )));
        }
    }
    let _ = name;
    for i in (0..ps.len().min(args.len())).rev() {
        let (p, a) = (ps[i], &args[i]);
        if p == P::I || p == P::O || p.accepts(a) {
            continue;
        }
        if a.is_null() {
            return Err(npe(null_arg_msg(recv, name, args.len(), i)).frames(npe_frames(recv, name, args.len(), i)));
        }
        let target = if matches!(recv, Big::Rounding(_)) { "java.lang.Enum" } else { p.java_name() };
        return Err(class_cast(format!("Cannot cast {} to {}", a.type_name(), target)).frames(CLASS_CAST));
    }
    Ok(())
}

fn arg_int(args: &[Val], i: usize) -> i32 {
    match &args[i] {
        Val::Int(v) => *v,
        Val::Char(c) => *c as i32,
        _ => 0,
    }
}
fn arg_dec(args: &[Val], i: usize) -> Dec {
    match &args[i] {
        Val::Big(b) => match &**b {
            Big::Dec(d) => d.clone(),
            _ => Dec::zero(0),
        },
        _ => Dec::zero(0),
    }
}
fn arg_bint(args: &[Val], i: usize) -> BigInt {
    match &args[i] {
        Val::Big(b) => match &**b {
            Big::Int(x) => x.clone(),
            _ => BigInt::zero(),
        },
        _ => BigInt::zero(),
    }
}
fn arg_mc(args: &[Val], i: usize) -> Mc {
    match &args[i] {
        Val::Big(b) => match &**b {
            Big::MathCtx(m) => *m,
            _ => UNLIMITED,
        },
        _ => UNLIMITED,
    }
}
fn arg_rm(args: &[Val], i: usize) -> Rm {
    match &args[i] {
        Val::Big(b) => match &**b {
            Big::Rounding(r) => *r,
            _ => Rm::HalfUp,
        },
        _ => Rm::HalfUp,
    }
}

/// A `float` result: the engine has no float value, so carry the float's
/// shortest decimal rendering (what `Float.toString` prints) as a double.
fn float_val(f: f32) -> Val {
    Val::Double(format!("{f}").parse::<f64>().unwrap_or(f as f64))
}

// ---------------------------------------------------------------------------
// JDK stack frames (ES prints the `java.base/...` frames of the throwing
// JDK method above the script frame; line numbers are those of the JDK
// bundled with the reference ES 8.19.22).


type Frames = &'static [&'static str];

const BD_INIT_515: &str = bd!("<init>", 515);
const BD_INIT_721: &str = bd!("<init>", 721);
const BD_INIT_921: &str = bd!("<init>", 921);
const BD_INIT_937: &str = bd!("<init>", 937);
const BI_INIT_528: &str = bi!("<init>", 528);
const BI_INIT_531: &str = bi!("<init>", 531);
const BI_INIT_533: &str = bi!("<init>", 533);
const BI_INIT_541: &str = bi!("<init>", 541);
const BI_INIT_547: &str = bi!("<init>", 547);
const BI_INIT_552: &str = bi!("<init>", 552);
const BI_INIT_721: &str = bi!("<init>", 721);
const BI_PARSE_FIRST: Frames = &["java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:67)", "java.base/java.lang.Integer.parseInt(Integer.java:529)", bi!("<init>", 583)];
const BI_PARSE_GROUP: Frames = &["java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:67)", "java.base/java.lang.Integer.parseInt(Integer.java:529)", bi!("<init>", 592)];
const CLASS_CAST: Frames = &["java.base/java.lang.Class.cast(Class.java:3495)"];

/// `commonNeedIncrement` prefixes: compact `divideAndRound(long, long)`,
/// `divideAndRound(BigInteger, long)`, `divideAndRound(BigInteger, BigInteger)`,
/// and the `doRound` power-of-ten variants.
const RN_C: Frames = &[bd!("commonNeedIncrement", 4963), bd!("needIncrement", 5013), bd!("divideAndRound", 4927)];
const RN_B: Frames = &[bd!("commonNeedIncrement", 4963), bd!("needIncrement", 5093), bd!("divideAndRound", 5061)];
const RN_BB: Frames = &[bd!("commonNeedIncrement", 4963), bd!("needIncrement", 5164), bd!("divideAndRound", 5139)];
const RN_T: Frames = &[bd!("commonNeedIncrement", 4963), bd!("needIncrement", 5013), bd!("divideAndRound", 4949)];
const RN_TB: Frames = &[bd!("commonNeedIncrement", 4963), bd!("needIncrement", 5164), bd!("divideAndRound", 5111), bd!("divideAndRoundByTenPow", 4904)];

/// Concatenate frame lists into an interned `'static` slice (each distinct
/// combination is allocated once).
fn join(parts: &[Frames]) -> Frames {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<Vec<&'static str>, Frames>>> = OnceLock::new();
    let v: Vec<&'static str> = parts.iter().flat_map(|p| p.iter().copied()).collect();
    if parts.len() == 1 {
        return parts[0];
    }
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(f) = cache.get(&v) {
        return f;
    }
    let leaked: Frames = Box::leak(v.clone().into_boxed_slice());
    cache.insert(v, leaked);
    leaked
}

/// Set frames on an error that has none yet.
fn with_frames(mut e: Box<ScriptErr>, frames: Frames) -> Box<ScriptErr> {
    if e.frames.is_empty() {
        e.frames = frames;
    }
    e
}

/// Append caller frames to an error thrown deeper in the JDK.
fn with_tail(mut e: Box<ScriptErr>, tail: Frames) -> Box<ScriptErr> {
    if !e.frames.is_empty() {
        e.frames = join(&[e.frames, tail]);
    }
    e
}

/// Java `intCompact != INFLATED`.
fn is_compact(x: &BigInt) -> bool {
    x.to_i64().is_some_and(|v| v != i64::MIN)
}

/// Frames of the helpful-NPE thrown for a null reference argument.
fn npe_frames(recv: &Big, name: &str, n: usize, idx: usize) -> Frames {
    match recv {
        Big::Dec(_) => match (name, n, idx) {
            ("add", 1, _) => &[bd!("add", 1414)],
            ("add", 2, 1) => &[bd!("add", 1441)],
            ("add", 2, _) => &[bd!("add", 1449)],
            ("subtract", 1, _) => &[bd!("subtract", 1559)],
            ("subtract", 2, 1) => &[bd!("subtract", 1589)],
            ("subtract", 2, _) => &[bd!("subtract", 1592)],
            ("multiply", 1, _) => &[bd!("multiply", 1604)],
            ("multiply", 2, 1) => &[bd!("multiply", 1630)],
            ("multiply", 2, _) => &[bd!("multiply", 1632)],
            ("divide", 1, _) => &[bd!("divide", 1783)],
            ("divide", 2, 1) => &[bd!("divide", 1842)],
            ("divide", 2, _) => &[bd!("divide", 1847)],
            ("divideToIntegralValue", 1, _) => &[bd!("divideToIntegralValue", 1897)],
            ("divideToIntegralValue", 2, 1) => &[bd!("divideToIntegralValue", 1948)],
            ("divideToIntegralValue", 2, _) => &[bd!("compareMagnitude", 3185), bd!("divideToIntegralValue", 1949)],
            ("divideAndRemainder", 1, _) => &[bd!("divideToIntegralValue", 1897), bd!("divideAndRemainder", 2068)],
            ("divideAndRemainder", 2, 1) => &[bd!("divideAndRemainder", 2100)],
            ("divideAndRemainder", 2, _) => &[bd!("compareMagnitude", 3185), bd!("divideToIntegralValue", 1949), bd!("divideAndRemainder", 2106)],
            ("remainder", 1, _) => &[bd!("divideToIntegralValue", 1897), bd!("divideAndRemainder", 2068), bd!("remainder", 2010)],
            ("remainder", 2, 1) => &[bd!("divideAndRemainder", 2100), bd!("remainder", 2040)],
            ("remainder", 2, _) => &[bd!("compareMagnitude", 3185), bd!("divideToIntegralValue", 1949), bd!("divideAndRemainder", 2106), bd!("remainder", 2040)],
            ("compareTo", _, _) => &[bd!("compareTo", 3164)],
            ("max", _, _) => &[bd!("compareTo", 3164), bd!("max", 3305)],
            ("min", _, _) => &[bd!("compareTo", 3164), bd!("min", 3291)],
            ("setScale", _, _) => &[bd!("setScale", 2914)],
            ("round", _, _) => &[bd!("plus", 2700), bd!("round", 2881)],
            ("plus", _, _) => &[bd!("plus", 2700)],
            ("abs", _, _) => &[bd!("plus", 2700), bd!("abs", 2641)],
            ("negate", _, _) => &[bd!("plus", 2700), bd!("negate", 2667)],
            ("pow", _, _) => &[bd!("pow", 2582)],
            _ => &[],
        },
        Big::Int(_) => match (name, idx) {
            ("add", _) => &[bi!("add", 1385)],
            ("subtract", _) => &[bi!("subtract", 1587)],
            ("multiply", _) => &[bi!("multiply", 1682), bi!("multiply", 1645)],
            ("divide", _) => &[bi!("divide", 2471)],
            ("remainder", _) => &[bi!("remainder", 2537)],
            ("divideAndRemainder", _) => &[bi!("divideAndRemainder", 2508)],
            ("mod", _) => &[bi!("mod", 2959)],
            ("modPow", 0) => &[bi!("modPow", 2984)],
            ("modPow", _) => &[bi!("modPow", 2980)],
            ("modInverse", _) => &[bi!("modInverse", 3541)],
            ("gcd", _) => &[bi!("gcd", 2833)],
            ("and", _) => &[bi!("and", 3740)],
            ("or", _) => &[bi!("or", 3757)],
            ("xor", _) => &[bi!("xor", 3774)],
            ("andNot", _) => &[bi!("andNot", 3808)],
            ("compareTo", _) => &[bi!("compareTo", 4026)],
            ("max", _) => &[bi!("compareTo", 4026), bi!("max", 4146)],
            ("min", _) => &[bi!("compareTo", 4026), bi!("min", 4135)],
            _ => &[],
        },
        Big::Rounding(_) => &["java.base/java.lang.Enum.compareTo(Enum.java:218)"],
        Big::MathCtx(_) => &[],
    }
}

impl Dec {
    /// Frames of "Rounding necessary" thrown inside `divide(BigDecimal, mc)`:
    /// which `divide`/`divideAndRound` overloads the JDK dispatched to.
    fn divide_rn_frames(&self, o: &Dec, mc: &Mc) -> Frames {
        let (xc, yc) = (is_compact(&self.u), is_compact(&o.u));
        let xp = self.precision();
        let yp0 = o.precision();
        let mcp = mc.precision as i64;
        if xc && yc && xp <= yp0 && yp0 < 18 && mcp < 18 {
            return join(&[RN_C, &[bd!("divideSmallFastPath", 5493), bd!("divide", 5508), bd!("divide", 1871)]]);
        }
        match (xc, yc) {
            (true, true) => {
                let sdiff = xp - yp0;
                let gt = if sdiff < 0 {
                    ten_pow(-sdiff).map(|p| self.u.abs() * p > o.u.abs()).unwrap_or(false)
                } else {
                    ten_pow(sdiff).map(|p| self.u.abs() > o.u.abs() * p).unwrap_or(false)
                };
                let yp = if gt { yp0 - 1 } else { yp0 };
                let raise = mcp + yp - xp;
                let fits = raise <= 0 || ten_pow(raise).map(|p| is_compact(&(self.u.abs() * p))).unwrap_or(false);
                if fits {
                    join(&[RN_C, &[bd!("divide", 5526), bd!("divide", 1871)]])
                } else {
                    join(&[RN_B, &[bd!("divide", 5524), bd!("divide", 1871)]])
                }
            }
            (true, false) => join(&[RN_BB, &[bd!("divide", 5610), bd!("divide", 1873)]]),
            (false, true) => join(&[RN_B, &[bd!("divide", 5582), bd!("divide", 1877)]]),
            (false, false) => join(&[RN_BB, &[bd!("divide", 5641), bd!("divide", 1879)]]),
        }
    }
}

/// Frames for an exception thrown by a `BigDecimal` method.
fn dec_frames(d: &Dec, name: &str, args: &[Val], e: &ScriptErr) -> Frames {
    let n = args.len();
    let msg = e.msg.as_str();
    let compact = is_compact(&d.u);
    let set_scale_prefix = || if compact { join(&[RN_C, &[bd!("setScale", 2981)]]) } else { join(&[RN_B, &[bd!("setScale", 2996)]]) };
    let do_round_prefix = || if compact { join(&[RN_T, &[bd!("doRound", 4825)]]) } else { join(&[RN_TB, &[bd!("doRound", 4810)]]) };
    let mcp = |i: usize| arg_mc(args, i).precision;
    const RN: &str = "Rounding necessary";
    match (name, n, msg) {
        ("setScale", 1, RN) => join(&[set_scale_prefix(), &[bd!("setScale", 3041)]]),
        ("setScale", 2, RN) => join(&[set_scale_prefix(), &[bd!("setScale", 2914)]]),
        ("toBigIntegerExact", _, RN) => join(&[set_scale_prefix(), &[bd!("toBigIntegerExact", 3613)]]),
        ("longValueExact" | "intValueExact" | "shortValueExact" | "byteValueExact", _, _) => {
            let tail: Frames = match name {
                "intValueExact" => &[bd!("intValueExact", 3759)],
                "shortValueExact" => &[bd!("shortValueExact", 3779)],
                "byteValueExact" => &[bd!("byteValueExact", 3799)],
                _ => &[],
            };
            if msg == RN {
                if d.signum() != 0 && d.precision() - d.s as i64 <= 0 {
                    join(&[&[bd!("longValueExact", 3686)], tail])
                } else {
                    join(&[set_scale_prefix(), &[bd!("longValueExact", 3700)], tail])
                }
            } else if d.long_value_exact().is_ok() {
                match name {
                    "intValueExact" => &[bd!("intValueExact", 3761)],
                    "shortValueExact" => &[bd!("shortValueExact", 3781)],
                    _ => &[bd!("byteValueExact", 3801)],
                }
            } else if d.precision() - d.s as i64 > 19 {
                join(&[&[bd!("longValueExact", 3697)], tail])
            } else {
                join(&[&["java.base/java.math.BigDecimal$LongOverflow.check(BigDecimal.java:3717)", bd!("longValueExact", 3702)], tail])
            }
        }
        ("divide", 1, "Division by zero") => &[bd!("divide", 1786)],
        ("divide", 1, "Division undefined") => &[bd!("divide", 1785)],
        ("divide", 1, _) => &[bd!("divide", 1811)],
        ("divide", 2, _) if mcp(1) == 0 => match msg {
            "Division by zero" => &[bd!("divide", 1786), bd!("divide", 1844)],
            "Division undefined" => &[bd!("divide", 1785), bd!("divide", 1844)],
            _ => &[bd!("divide", 1811), bd!("divide", 1844)],
        },
        ("divide", 2, "Division by zero") => &[bd!("divide", 1863)],
        ("divide", 2, "Division undefined") => &[bd!("divide", 1862)],
        ("divide", 2, RN) => d.divide_rn_frames(&arg_dec(args, 0), &arg_mc(args, 1)),
        ("divideToIntegralValue" | "remainder" | "divideAndRemainder", _, _) => {
            let tail: Frames = match (name, n) {
                ("divideAndRemainder", 1) => &[bd!("divideAndRemainder", 2068)],
                ("divideAndRemainder", 2) => &[bd!("divideAndRemainder", 2106)],
                ("remainder", 1) => &[bd!("divideAndRemainder", 2068), bd!("remainder", 2010)],
                ("remainder", 2) => &[bd!("divideAndRemainder", 2106), bd!("remainder", 2040)],
                _ => &[],
            };
            let o = arg_dec(args, 0);
            let mc_path = n == 2 && mcp(1) > 0 && d.cmp_magnitude(&o).map(|c| c != Ordering::Less).unwrap_or(true);
            let dtiv: Frames = if mc_path { &[bd!("divideToIntegralValue", 1962)] } else { &[bd!("divideToIntegralValue", 1913)] };
            let div_zero: Frames = &[bd!("divide", 1863)];
            let div_undef: Frames = &[bd!("divide", 1862)];
            let div_imposs: Frames = &[bd!("divideToIntegralValue", 1974)];
            match msg {
                "Division impossible" => join(&[div_imposs, tail]),
                "Division by zero" => join(&[div_zero, dtiv, tail]),
                "Division undefined" => join(&[div_undef, dtiv, tail]),
                _ => &[],
            }
        }
        ("pow", 1, "Invalid operation") => &[bd!("pow", 2523)],
        ("pow", 2, "Invalid operation") => if mcp(1) == 0 { &[bd!("pow", 2523), bd!("pow", 2583)] } else { &[bd!("pow", 2594)] },
        ("pow", 2, RN) => join(&[do_round_prefix(), &[bd!("pow", 2617)]]),
        ("round", 1, RN) => join(&[do_round_prefix(), &[bd!("plus", 2702), bd!("round", 2881)]]),
        ("plus", 1, RN) => join(&[do_round_prefix(), &[bd!("plus", 2702)]]),
        ("abs", 1, RN) => join(&[do_round_prefix(), &[bd!("plus", 2702), bd!("abs", 2641)]]),
        ("negate", 1, RN) => join(&[do_round_prefix(), &[bd!("plus", 2702), bd!("negate", 2667)]]),
        ("add", 2, RN) => join(&[RN_T, &[bd!("doRound", 4887), bd!("add", 1482)]]),
        ("subtract", 2, RN) => join(&[RN_T, &[bd!("doRound", 4887), bd!("add", 1482), bd!("subtract", 1592)]]),
        ("multiply", 2, RN) => join(&[RN_T, &[bd!("doRound", 4848), bd!("multiplyAndRound", 5899), bd!("multiply", 1635)]]),
        ("movePointLeft", _, "Underflow" | "Overflow") => &[bd!("checkScale", 4656), bd!("movePointLeft", 3065)],
        ("movePointRight", _, "Underflow" | "Overflow") => &[bd!("checkScale", 4656), bd!("movePointRight", 3089)],
        ("scaleByPowerOfTen", _, "Underflow" | "Overflow") => &[bd!("checkScale", 4656), bd!("scaleByPowerOfTen", 3109)],
        ("multiply", 1, "Underflow" | "Overflow") => &[bd!("checkScale", 4656), bd!("multiply", 1604)],
        ("pow", 1, "Underflow" | "Overflow") => &[bd!("checkScale", 4656), bd!("pow", 2526)],
        _ => &[],
    }
}

/// Frames for an exception thrown by a `BigInteger` method.
fn int_frames(name: &str, e: &ScriptErr) -> Frames {
    match (name, e.msg.as_str()) {
        ("divide", "BigInteger divide by zero") => &[mbi!("divideKnuth", 1271), bi!("divideKnuth", 2492), bi!("divide", 2473)],
        ("remainder", "BigInteger divide by zero") => &[mbi!("divideKnuth", 1271), mbi!("divideKnuth", 1255), bi!("remainderKnuth", 2551), bi!("remainder", 2539)],
        ("divideAndRemainder", "BigInteger divide by zero") => &[mbi!("divideKnuth", 1271), mbi!("divideKnuth", 1255), bi!("divideAndRemainderKnuth", 2522), bi!("divideAndRemainder", 2510)],
        ("mod", "BigInteger: modulus not positive") => &[bi!("mod", 2960)],
        ("modPow", "BigInteger: modulus not positive") => &[bi!("modPow", 2981)],
        ("modInverse", "BigInteger: modulus not positive") => &[bi!("modInverse", 3542)],
        ("modInverse", "BigInteger not invertible.") => &[mbi!("modInverse", 2512), mbi!("mutableModInverse", 2389), bi!("modInverse", 3558)],
        ("modPow", "BigInteger not invertible.") => &[mbi!("modInverse", 2512), mbi!("mutableModInverse", 2389), bi!("modInverse", 3558), bi!("modPow", 3046)],
        ("testBit", "Negative bit address") => &[bi!("testBit", 3829)],
        ("setBit", "Negative bit address") => &[bi!("setBit", 3844)],
        ("clearBit", "Negative bit address") => &[bi!("clearBit", 3868)],
        ("flipBit", "Negative bit address") => &[bi!("flipBit", 3892)],
        ("pow", "Negative exponent") => &[bi!("pow", 2597)],
        ("longValueExact", _) => &[bi!("longValueExact", 5131)],
        ("intValueExact", _) => &[bi!("intValueExact", 5154)],
        ("shortValueExact", _) => &[bi!("shortValueExact", 5173)],
        ("byteValueExact", _) => &[bi!("byteValueExact", 5192)],
        _ => &[],
    }
}

// ---------------------------------------------------------------------------
// Hooks

pub fn call_static(_rt: &mut Rt, class: &str, method: &str, args: &[Val]) -> SResult<Option<Val>> {
    Ok(Some(match (class, method, args.len()) {
        ("BigDecimal", "valueOf", 1) => {
            let d = def_double(&args[0])?;
            if d.is_nan() || d.is_infinite() {
                return Err(nfe("Infinite or NaN").frames(&[bd!("valueOf", 1390)]));
            }
            dec(parse_decimal(&java_double_string(d))?)
        }
        ("BigInteger", "valueOf", 1) => bint(BigInt::from(def_long(&args[0])?)),
        ("RoundingMode", "valueOf", 1) => {
            let s = def_string(&args[0], "Name is null", &["java.base/java.lang.Enum.valueOf(Enum.java:291)", "java.base/java.math.RoundingMode.valueOf(RoundingMode.java:120)"])?;
            match Rm::from_name(s) {
                Some(r) => rmode(r),
                None => return Err(illegal_arg(format!("No enum constant java.math.RoundingMode.{s}")).frames(&["java.base/java.lang.Enum.valueOf(Enum.java:293)", "java.base/java.math.RoundingMode.valueOf(RoundingMode.java:120)"])),
            }
        }
        ("RoundingMode", "values", 0) => Val::new_array(Ty::Ref("RoundingMode".into()), RMS.iter().map(|r| rmode(*r)).collect()),
        _ => return Ok(None),
    }))
}

pub fn static_field(class: &str, name: &str) -> Option<Val> {
    Some(match (class, name) {
        ("BigDecimal", "ZERO") => dec(Dec::zero(0)),
        ("BigDecimal", "ONE") => dec(Dec::from_i64(1)),
        ("BigDecimal", "TEN") => dec(Dec::from_i64(10)),
        ("BigInteger", "ZERO") => bint(BigInt::zero()),
        ("BigInteger", "ONE") => bint(BigInt::one()),
        ("BigInteger", "TEN") => bint(BigInt::from(10)),
        ("MathContext", "DECIMAL32") => mctx(DECIMAL32),
        ("MathContext", "DECIMAL64") => mctx(DECIMAL64),
        ("MathContext", "DECIMAL128") => mctx(DECIMAL128),
        ("MathContext", "UNLIMITED") => mctx(UNLIMITED),
        ("RoundingMode", n) => rmode(Rm::from_name(n)?),
        _ => return None,
    })
}

/// `new BigDecimal(..)` / `new BigInteger(..)` / `new MathContext(..)`.
pub fn construct(class: &str, args: &[Val]) -> Option<SResult<Val>> {
    Some((|| {
        let guard=super::allocation::Allocation::new();
        if guard.enabled() {
            super::allocation::Scope::collect_dead();
            let mut bytes=4096u64;
            for arg in args {bytes=bytes.saturating_add(super::allocation::serialization_bound(arg)?.saturating_mul(4));}
            guard.reserve(bytes)?;
        }
        let result=construct_inner(class,args)?;
        super::allocation::Scope::adopt(&result)?;
        Ok(result)
    })())
}

fn construct_inner(class: &str, args: &[Val]) -> SResult<Val> {
    let n = args.len();
    Ok(match (class, n) {
        ("BigDecimal", 1) => {
            let s = def_string(&args[0], "Cannot invoke \"String.toCharArray()\" because \"val\" is null", &[BD_INIT_921])?;
            dec(parse_decimal(s).map_err(|e| with_tail(e, &[BD_INIT_515, BD_INIT_921]))?)
        }
        ("BigDecimal", 2) => {
            let s = def_string(&args[0], "Cannot invoke \"String.toCharArray()\" because \"val\" is null", &[BD_INIT_937])?;
            let mc = match &args[1] {
                Val::Big(b) => match &**b {
                    Big::MathCtx(m) => *m,
                    _ => return Err(ctor_cast_msg(&args[1], "java.math.MathContext")),
                },
                Val::Null => return Err(npe("Cannot read field \"precision\" because \"mc\" is null").frames(&[bd!("<init>", 634), BD_INIT_937])),
                other => return Err(ctor_cast_msg(other, "java.math.MathContext")),
            };
            let d = parse_decimal(s).map_err(|e| with_tail(e, &[BD_INIT_937]))?;
            dec(d.do_round(&mc).map_err(|e| with_tail(e, &[bd!("<init>", 640), BD_INIT_937]))?)
        }
        ("BigInteger", 1) => {
            let s = def_string(&args[0], "Cannot invoke \"String.length()\" because \"val\" is null", &[BI_INIT_528, BI_INIT_721])?;
            bint(parse_big_integer(s, 10).map_err(|e| with_tail(e, &[BI_INIT_721]))?)
        }
        ("BigInteger", 2) => {
            let s = def_string(&args[0], "Cannot invoke \"String.length()\" because \"val\" is null", &[BI_INIT_528])?;
            let radix = def_int(&args[1])?;
            bint(parse_big_integer(s, radix as u32)?)
        }
        ("MathContext", 1) => {
            let p = def_int(&args[0])?;
            if p < 0 {
                return Err(illegal_arg("Digits < 0").frames(&["java.base/java.math.MathContext.<init>(MathContext.java:158)", "java.base/java.math.MathContext.<init>(MathContext.java:142)"]));
            }
            mctx(Mc { precision: p as u32, rounding: Rm::HalfUp })
        }
        ("MathContext", 2) => {
            let p = def_int(&args[0])?;
            let rm = match &args[1] {
                Val::Big(b) => match &**b {
                    Big::Rounding(r) => *r,
                    _ => return Err(ctor_cast_msg(&args[1], "java.math.RoundingMode")),
                },
                Val::Null => return Err(npe("null RoundingMode").frames(&["java.base/java.math.MathContext.<init>(MathContext.java:160)"])),
                other => return Err(ctor_cast_msg(other, "java.math.RoundingMode")),
            };
            if p < 0 {
                return Err(illegal_arg("Digits < 0").frames(&["java.base/java.math.MathContext.<init>(MathContext.java:158)"]));
            }
            mctx(Mc { precision: p as u32, rounding: rm })
        }
        _ => return Err(illegal_arg(format!("constructor [{class}, <init>/{n}] not found"))),
    })
}

pub fn instance_of(v: &Val, class: &str) -> Option<bool> {
    let Val::Big(b) = v else { return None };
    Some(match (&**b, class) {
        (_, "Object") => true,
        (Big::Dec(_), "BigDecimal" | "Number" | "Comparable") => true,
        (Big::Int(_), "BigInteger" | "Number" | "Comparable") => true,
        (Big::MathCtx(_), "MathContext") => true,
        (Big::Rounding(_), "RoundingMode" | "Enum" | "Comparable") => true,
        _ => false,
    })
}

/// Instance methods of `Val::Big`.
/// Bound bigint coefficients, decimal scale alignment, and scratch copies before
/// dispatching to num-bigint. Reservations follow the returned coefficient owner.
fn arithmetic_guard(recv:&Val,name:&str,args:&[Val])->SResult<super::allocation::Allocation> {
    let guard=super::allocation::Allocation::new();
    if !guard.enabled(){return Ok(guard)}
    super::allocation::Scope::collect_dead();
    let mut bits=64u64;
    let mut low_scale=i64::MAX;
    let mut high_scale=i64::MIN;
    for value in std::iter::once(recv).chain(args.iter()) {
        if let Val::Big(value)=value {match &**value {
            Big::Int(v)=>bits=bits.saturating_add(v.bits()),
            Big::Dec(v)=>{bits=bits.saturating_add(v.u.bits());low_scale=low_scale.min(v.s as i64);high_scale=high_scale.max(v.s as i64);},
            Big::MathCtx(v)=>bits=bits.saturating_add((v.precision as u64).saturating_mul(4)),
            _=>{},
        }}
    }
    if name=="pow" {
        let coefficient_bits=match recv {Val::Big(v)=>match &**v {Big::Int(v)=>v.bits(),Big::Dec(v)=>v.u.bits(),_=>0},_=>0};
        let bounded_precision=args.iter().any(|v|matches!(v,Val::Big(v) if matches!(&**v,Big::MathCtx(mc) if mc.precision>0)));
        if coefficient_bits>1&&!bounded_precision {bits=bits.saturating_mul(args.first().and_then(Val::as_i64).unwrap_or(0).unsigned_abs().max(1));}
    }
    if matches!(name,"shiftLeft"|"shiftRight"|"setBit"|"clearBit"|"flipBit") {bits=bits.saturating_add(args.first().and_then(Val::as_i64).unwrap_or(0).unsigned_abs());}
    if matches!(recv,Val::Big(v) if matches!(&**v,Big::Dec(_))) {
        if let Val::Big(value)=recv {if let Big::Dec(d)=&**value {
            let arg=args.first().and_then(Val::as_i64).unwrap_or(0);
            let expanded=match name {
                "setScale" if !d.is_zero()=>arg.saturating_sub(d.s as i64).unsigned_abs(),
                "movePointLeft" if arg!=0&&!d.is_zero()=>(d.s as i64).saturating_add(arg).min(0).unsigned_abs(),
                "movePointRight" if arg!=0&&!d.is_zero()=>(d.s as i64).saturating_sub(arg).min(0).unsigned_abs(),
                _=>0,
            };
            bits=bits.saturating_add(expanded.saturating_mul(4));
        }}
        if matches!(name,"add"|"subtract"|"compareTo"|"max"|"min"|"divide"|"divideToIntegralValue"|"remainder"|"divideAndRemainder") && low_scale<=high_scale {
            bits=bits.saturating_add(high_scale.saturating_sub(low_scale).unsigned_abs().saturating_mul(4));
        }
        if matches!(name,"toBigInteger"|"toBigIntegerExact"|"toPlainString"|"longValue"|"intValue"|"shortValue"|"byteValue") && low_scale<=high_scale {
            bits=bits.saturating_add(low_scale.unsigned_abs().max(high_scale.unsigned_abs()).saturating_mul(4));
        }
    }
    guard.reserve(bits.saturating_add(7).saturating_div(8).saturating_mul(32).saturating_add(4096))?;
    Ok(guard)
}
pub fn call_method(rt: &mut Rt, recv: &Val, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let _guard=arithmetic_guard(recv,name,args)?;
    let result=call_method_inner(rt,recv,name,args)?;
    if let Some(value)=&result {super::allocation::Scope::adopt(value)?;}
    Ok(result)
}
fn call_method_inner(_rt: &mut Rt, recv: &Val, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let Val::Big(b) = recv else { return Ok(None) };
    let n = args.len();
    let Some((ps, ret)) = sig(b, name, n) else { return Ok(None) };
    check_args(b, name, ps, ret, args)?;
    // Object / Number methods shared by all four classes.
    match (name, n) {
        ("toString", 0) => return Ok(Some(Val::string(recv.to_java_string()))),
        ("hashCode", 0) => return Ok(Some(Val::Int(b.hash()))),
        ("equals", 1) => return Ok(Some(Val::Bool(recv.equals(&args[0])))),
        _ => {}
    }
    Ok(Some(match &**b {
        Big::Dec(d) => dec_method(d, name, args).map_err(|e| {
            let f = dec_frames(d, name, args, &e);
            with_frames(e, f)
        })?,
        Big::Int(x) => int_method(x, name, args).map_err(|e| {
            let f = int_frames(name, &e);
            with_frames(e, f)
        })?,
        Big::MathCtx(m) => match name {
            "getPrecision" => Val::Int(m.precision as i32),
            _ => rmode(m.rounding),
        },
        Big::Rounding(r) => match name {
            "name" => Val::str(r.name()),
            "ordinal" => Val::Int(*r as i32),
            _ => Val::Int(*r as i32 - arg_rm(args, 0) as i32),
        },
    }))
}

fn dec_method(d: &Dec, name: &str, args: &[Val]) -> SResult<Val> {
    let n = args.len();
    Ok(match (name, n) {
        ("abs", 0) => dec(d.abs()),
        ("abs", 1) => dec(if d.signum() < 0 { d.negate().do_round(&arg_mc(args, 0))? } else { d.clone().do_round(&arg_mc(args, 0))? }),
        ("negate", 0) => dec(d.negate()),
        ("negate", 1) => dec(d.negate().do_round(&arg_mc(args, 0))?),
        ("plus", 0) => dec(d.clone()),
        ("plus" | "round", 1) => dec(d.clone().do_round(&arg_mc(args, 0))?),
        ("stripTrailingZeros", 0) => dec(d.strip_trailing_zeros()),
        ("ulp", 0) => dec(Dec::new(BigInt::one(), d.s)),
        ("add", 1) => dec(d.add(&arg_dec(args, 0))?),
        ("add", 2) => dec(d.add_mc(&arg_dec(args, 0), &arg_mc(args, 1))?),
        ("subtract", 1) => dec(d.subtract(&arg_dec(args, 0))?),
        ("subtract", 2) => dec(d.subtract_mc(&arg_dec(args, 0), &arg_mc(args, 1))?),
        ("multiply", 1) => dec(d.multiply(&arg_dec(args, 0))?),
        ("multiply", 2) => dec(d.multiply_mc(&arg_dec(args, 0), &arg_mc(args, 1))?),
        ("divide", 1) => dec(d.divide(&arg_dec(args, 0))?),
        ("divide", 2) => dec(d.divide_mc(&arg_dec(args, 0), &arg_mc(args, 1))?),
        ("divideToIntegralValue", 1) => dec(d.divide_to_integral(&arg_dec(args, 0))?),
        ("divideToIntegralValue", 2) => dec(d.divide_to_integral_mc(&arg_dec(args, 0), &arg_mc(args, 1))?),
        ("remainder", 1) => dec(d.divide_and_remainder(&arg_dec(args, 0), None)?.1),
        ("remainder", 2) => dec(d.divide_and_remainder(&arg_dec(args, 0), Some(&arg_mc(args, 1)))?.1),
        ("divideAndRemainder", _) => {
            let mc = if n == 2 { Some(arg_mc(args, 1)) } else { None };
            let (q, r) = d.divide_and_remainder(&arg_dec(args, 0), mc.as_ref())?;
            Val::new_array(Ty::Ref("BigDecimal".into()), vec![dec(q), dec(r)])
        }
        ("max", 1) => {
            let o = arg_dec(args, 0);
            dec(if d.cmp(&o)? != Ordering::Less { d.clone() } else { o })
        }
        ("min", 1) => {
            let o = arg_dec(args, 0);
            dec(if d.cmp(&o)? != Ordering::Greater { d.clone() } else { o })
        }
        ("compareTo", 1) => Val::Int(match d.cmp(&arg_dec(args, 0))? {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }),
        ("precision", 0) => Val::Int(d.precision() as i32),
        ("scale", 0) => Val::Int(d.s),
        ("signum", 0) => Val::Int(d.signum()),
        ("movePointLeft", 1) => dec(d.move_point_left(arg_int(args, 0))?),
        ("movePointRight", 1) => dec(d.move_point_right(arg_int(args, 0))?),
        ("scaleByPowerOfTen", 1) => dec(d.scale_by_power_of_ten(arg_int(args, 0))?),
        ("pow", 1) => dec(d.pow(arg_int(args, 0))?),
        ("pow", 2) => dec(d.pow_mc(arg_int(args, 0), &arg_mc(args, 1))?),
        ("setScale", 1) => dec(d.set_scale(arg_int(args, 0), Rm::Unnecessary)?),
        ("setScale", 2) => dec(d.set_scale(arg_int(args, 0), arg_rm(args, 1))?),
        ("toBigInteger", 0) => bint(d.to_big_integer()?),
        ("toBigIntegerExact", 0) => bint(d.to_big_integer_exact()?),
        ("toEngineeringString", 0) => Val::string(d.layout(false)),
        ("toPlainString", 0) => Val::string(d.to_plain_string()),
        ("longValueExact", 0) => Val::Long(d.long_value_exact()?),
        ("intValueExact", 0) => {
            let v = d.long_value_exact()?;
            if v as i32 as i64 != v {
                return Err(arith("Overflow"));
            }
            Val::Int(v as i32)
        }
        ("shortValueExact", 0) => {
            let v = d.long_value_exact()?;
            if v as i16 as i64 != v {
                return Err(arith("Overflow"));
            }
            Val::Int(v as i32)
        }
        ("byteValueExact", 0) => {
            let v = d.long_value_exact()?;
            if v as i8 as i64 != v {
                return Err(arith("Overflow"));
            }
            Val::Int(v as i32)
        }
        ("longValue", 0) => Val::Long(low_i64(&d.to_big_integer()?)),
        ("intValue", 0) => Val::Int(low_i64(&d.to_big_integer()?) as i32),
        ("shortValue", 0) => Val::Int(low_i64(&d.to_big_integer()?) as i16 as i32),
        ("byteValue", 0) => Val::Int(low_i64(&d.to_big_integer()?) as i8 as i32),
        ("doubleValue", 0) => Val::Double(d.to_f64()),
        ("floatValue", 0) => float_val(d.to_f32()),
        _ => return Err(illegal_arg(format!("dynamic method [java.math.BigDecimal, {name}/{n}] not found"))),
    })
}

fn int_method(x: &BigInt, name: &str, args: &[Val]) -> SResult<Val> {
    let n = args.len();
    let bit_index = |i: usize| -> SResult<u64> {
        let v = arg_int(args, i);
        if v < 0 {
            return Err(arith("Negative bit address"));
        }
        Ok(v as u64)
    };
    let one_shl = |n: u64| -> SResult<BigInt> { shl(&BigInt::one(), n) };
    Ok(match (name, n) {
        ("abs", 0) => bint(x.abs()),
        ("negate", 0) => bint(-x),
        ("not", 0) => bint(!x),
        ("signum", 0) => Val::Int(match x.sign() {
            Sign::Minus => -1,
            Sign::NoSign => 0,
            Sign::Plus => 1,
        }),
        ("add", 1) => bint(x + arg_bint(args, 0)),
        ("subtract", 1) => bint(x - arg_bint(args, 0)),
        ("multiply", 1) => bint(x * arg_bint(args, 0)),
        ("divide" | "remainder" | "divideAndRemainder", 1) => {
            let y = arg_bint(args, 0);
            if y.is_zero() {
                return Err(arith("BigInteger divide by zero"));
            }
            let (q, r) = x.div_rem(&y);
            match name {
                "divide" => bint(q),
                "remainder" => bint(r),
                _ => Val::new_array(Ty::Ref("BigInteger".into()), vec![bint(q), bint(r)]),
            }
        }
        ("mod", 1) => {
            let m = arg_bint(args, 0);
            if m.sign() != Sign::Plus {
                return Err(arith("BigInteger: modulus not positive"));
            }
            bint(x.mod_floor(&m))
        }
        ("modPow", 2) => {
            let e = arg_bint(args, 0);
            let m = arg_bint(args, 1);
            if m.sign() != Sign::Plus {
                return Err(arith("BigInteger: modulus not positive"));
            }
            if m.is_one() {
                return Ok(bint(BigInt::zero()));
            }
            let base = if e.sign() == Sign::Minus { mod_inverse(x, &m)? } else { x.mod_floor(&m) };
            bint(base.modpow(&e.abs(), &m))
        }
        ("modInverse", 1) => {
            let m = arg_bint(args, 0);
            if m.sign() != Sign::Plus {
                return Err(arith("BigInteger: modulus not positive"));
            }
            if m.is_one() {
                return Ok(bint(BigInt::zero()));
            }
            bint(mod_inverse(x, &m)?)
        }
        ("gcd", 1) => bint(x.gcd(&arg_bint(args, 0))),
        ("and", 1) => bint(x & arg_bint(args, 0)),
        ("or", 1) => bint(x | arg_bint(args, 0)),
        ("xor", 1) => bint(x ^ arg_bint(args, 0)),
        ("andNot", 1) => bint(x & !arg_bint(args, 0)),
        ("max", 1) => {
            let o = arg_bint(args, 0);
            bint(if *x >= o { x.clone() } else { o })
        }
        ("min", 1) => {
            let o = arg_bint(args, 0);
            bint(if *x <= o { x.clone() } else { o })
        }
        ("compareTo", 1) => Val::Int(match x.cmp(&arg_bint(args, 0)) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }),
        ("bitCount", 0) => Val::Int(bit_count(x) as i32),
        ("bitLength", 0) => Val::Int(bit_length(x) as i32),
        ("getLowestSetBit", 0) => Val::Int(if x.is_zero() { -1 } else { x.magnitude().trailing_zeros().unwrap_or(0) as i32 }),
        ("testBit", 1) => Val::Bool((shr(x, bit_index(0)?) & BigInt::one()).is_one()),
        ("setBit", 1) => {
            let n = bit_index(0)?;
            if x.sign() == Sign::Minus && n >= x.bits() { bint(x.clone()) } else { bint(x | one_shl(n)?) }
        }
        ("clearBit", 1) => {
            let n = bit_index(0)?;
            if x.sign() != Sign::Minus && n >= x.bits() { bint(x.clone()) } else { bint(x & !one_shl(n)?) }
        }
        ("flipBit", 1) => bint(x ^ one_shl(bit_index(0)?)?),
        ("pow", 1) => {
            let e = arg_int(args, 0);
            if e < 0 {
                return Err(arith("Negative exponent"));
            }
            if x.bits() * e as u64 > MAX_BITS {
                return Err(overflow());
            }
            bint(x.pow(e as u32))
        }
        ("shiftLeft", 1) => {
            let s = arg_int(args, 0);
            bint(if s >= 0 { shl(x, s as u64)? } else { shr(x, (-(s as i64)) as u64) })
        }
        ("shiftRight", 1) => {
            let s = arg_int(args, 0);
            bint(if s >= 0 { shr(x, s as u64) } else { shl(x, (-(s as i64)) as u64)? })
        }
        ("toByteArray", 0) => Val::new_array(Ty::Byte, x.to_signed_bytes_be().into_iter().map(|b| Val::Int(b as i8 as i32)).collect()),
        ("toString", 1) => {
            let r = arg_int(args, 0);
            let r = if (2..=36).contains(&r) { r as u32 } else { 10 };
            Val::string(x.to_str_radix(r))
        }
        ("longValueExact", 0) => Val::Long(x.to_i64().ok_or_else(|| arith("BigInteger out of long range"))?),
        ("intValueExact", 0) => Val::Int(x.to_i32().ok_or_else(|| arith("BigInteger out of int range"))?),
        ("shortValueExact", 0) => Val::Int(x.to_i16().ok_or_else(|| arith("BigInteger out of short range"))? as i32),
        ("byteValueExact", 0) => Val::Int(x.to_i8().ok_or_else(|| arith("BigInteger out of byte range"))? as i32),
        ("longValue", 0) => Val::Long(low_i64(x)),
        ("intValue", 0) => Val::Int(low_i64(x) as i32),
        ("shortValue", 0) => Val::Int(low_i64(x) as i16 as i32),
        ("byteValue", 0) => Val::Int(low_i64(x) as i8 as i32),
        ("doubleValue", 0) => Val::Double(big_to_f64(x)),
        ("floatValue", 0) => float_val(x.to_string().parse::<f32>().unwrap_or(0.0)),
        _ => return Err(illegal_arg(format!("dynamic method [java.math.BigInteger, {name}/{n}] not found"))),
    })
}

fn mod_inverse(x: &BigInt, m: &BigInt) -> SResult<BigInt> {
    let a = x.mod_floor(m);
    let g = a.extended_gcd(m);
    if !g.gcd.is_one() {
        return Err(arith("BigInteger not invertible."));
    }
    Ok(g.x.mod_floor(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Dec {
        parse_decimal(s).unwrap()
    }
    fn s(x: &Dec) -> String {
        x.layout(true)
    }

    #[test]
    fn layout_matches_java() {
        assert_eq!(s(&d("0.0000001234")), "1.234E-7");
        assert_eq!(s(&d("0.000001234")), "0.000001234");
        assert_eq!(s(&d("100").strip_trailing_zeros()), "1E+2");
        assert_eq!(d("100").strip_trailing_zeros().to_plain_string(), "100");
        assert_eq!(d("1E+3").to_plain_string(), "1000");
        assert_eq!(s(&d("0E+2")), "0E+2");
        assert_eq!(d("0E+5").layout(false), "0.0E+6");
        assert_eq!(d("0E-8").layout(false), "0.00E-6");
        assert_eq!(d("1E+4").layout(false), "10E+3");
        assert_eq!(d("1.2E-7").layout(false), "120E-9");
        assert_eq!(d("-1.23E-10").to_plain_string(), "-0.000000000123");
    }

    #[test]
    fn arithmetic_matches_java() {
        assert_eq!(s(&d("1.2").multiply(&d("3.45")).unwrap()), "4.140");
        assert_eq!(s(&d("10").divide(&d("4")).unwrap()), "2.5");
        assert_eq!(d("10").divide(&d("3")).unwrap_err().msg, "Non-terminating decimal expansion; no exact representable decimal result.");
        assert_eq!(s(&d("10").divide_mc(&d("3"), &DECIMAL64).unwrap()), "3.333333333333333");
        assert_eq!(s(&d("100").divide_mc(&d("8"), &Mc { precision: 1, rounding: Rm::HalfUp }).unwrap()), "1E+1");
        assert_eq!(s(&d("6").divide(&d("2E+1")).unwrap()), "0.3");
        assert_eq!(s(&d("2.55").set_scale(1, Rm::HalfEven).unwrap()), "2.6");
        assert_eq!(s(&d("-2.5").set_scale(0, Rm::HalfUp).unwrap()), "-3");
        assert_eq!(s(&d("12345.6789").do_round(&Mc { precision: 3, rounding: Rm::HalfUp }).unwrap()), "1.23E+4");
        assert_eq!(s(&d("999").do_round(&Mc { precision: 2, rounding: Rm::HalfUp }).unwrap()), "1.0E+3");
        assert_eq!(s(&d("10.5").divide_to_integral(&d("3")).unwrap()), "3.0");
        assert_eq!(s(&d("1E+2").divide_and_remainder(&d("30"), None).unwrap().1), "10");
        assert_eq!(s(&d("100").add_mc(&d("0.000001"), &Mc { precision: 5, rounding: Rm::HalfUp }).unwrap()), "100.00");
        assert_eq!(s(&d("2").pow_mc(10, &Mc { precision: 2, rounding: Rm::HalfUp }).unwrap()), "1.0E+3");
        assert_eq!(s(&d("3").pow_mc(-2, &Mc { precision: 5, rounding: Rm::HalfUp }).unwrap()), "0.11111");
        assert_eq!(d("1E+20").divide_to_integral_mc(&d("3"), &Mc { precision: 5, rounding: Rm::HalfUp }).unwrap_err().msg, "Division impossible");
    }

    #[test]
    fn hashes_match_java() {
        assert_eq!(d("1.5").hash(), 466);
        assert_eq!(d("1E+3").hash(), 28);
        assert_eq!(d("123456789012345678901234567890.5").hash(), 1113278083);
        assert_eq!(bigint_hash(&BigInt::from(i64::MAX)), 2147483616);
        assert_eq!(bigint_hash(&"123456789012345678901234567890".parse::<BigInt>().unwrap()), 1915528825);
    }

    #[test]
    fn bigint_parse_and_bits() {
        assert_eq!(parse_big_integer("ff", 16).unwrap(), BigInt::from(255));
        assert_eq!(parse_big_integer("abc", 10).unwrap_err().msg, "For input string: \"abc\"");
        assert_eq!(parse_big_integer("", 10).unwrap_err().msg, "Zero length BigInteger");
        assert_eq!(bit_length(&BigInt::from(-8)), 3);
        assert_eq!(bit_count(&BigInt::from(-5)), 1);
        assert_eq!(shr(&BigInt::from(-17), 2), BigInt::from(-5));
        assert_eq!(low_i64(&(BigInt::from(i64::MAX) + 1)), i64::MIN);
    }
}
