//! `sin`, `cos` and `asin` as Java's `StrictMath` computes them (fdlibm
//! 5.3, Java's `FdLibm`), bit for bit. Lucene's `SloppyMath` builds its
//! look-up tables with them, so ES distances need these exact values (the
//! platform libm differs in the last bit for about half of the entries).
//! Only finite arguments of moderate size are needed (`|x| <= 2^19 * pi/2`
//! for `sin` / `cos`); larger ones fall back to the libm value.

#[inline]
fn hi(x: f64) -> i32 {
    (x.to_bits() >> 32) as i32
}

#[inline]
fn with_lo_zero(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0xFFFF_FFFF_0000_0000)
}

#[inline]
fn from_words(hi: u32, lo: u32) -> f64 {
    f64::from_bits(((hi as u64) << 32) | lo as u64)
}

fn kernel_sin(x: f64, y: f64, iy: i32) -> f64 {
    const S1: f64 = -1.666_666_666_666_663_243_48e-01;
    const S2: f64 = 8.333_333_333_322_489_461_24e-03;
    const S3: f64 = -1.984_126_982_985_794_931_34e-04;
    const S4: f64 = 2.755_731_370_707_006_767_89e-06;
    const S5: f64 = -2.505_076_025_340_686_341_95e-08;
    const S6: f64 = 1.589_690_995_211_550_102_21e-10;
    let ix = hi(x) & 0x7fff_ffff;
    if ix < 0x3e40_0000 && x as i32 == 0 {
        return x;
    }
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if iy == 0 {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

fn kernel_cos(x: f64, y: f64) -> f64 {
    const C1: f64 = 4.166_666_666_666_660_190_37e-02;
    const C2: f64 = -1.388_888_888_887_410_957_49e-03;
    const C3: f64 = 2.480_158_728_947_672_941_78e-05;
    const C4: f64 = -2.755_731_435_139_066_330_35e-07;
    const C5: f64 = 2.087_572_321_298_174_827_90e-09;
    const C6: f64 = -1.135_964_755_778_819_482_65e-11;
    let ix = hi(x) & 0x7fff_ffff;
    if ix < 0x3e40_0000 && x as i32 == 0 {
        return 1.0;
    }
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if ix < 0x3FD3_3333 {
        1.0 - (0.5 * z - (z * r - x * y))
    } else {
        let qx = if ix > 0x3fe9_0000 { 0.28125 } else { from_words((ix - 0x0020_0000) as u32, 0) };
        let hz = 0.5 * z - qx;
        let a = 1.0 - qx;
        a - (hz - (z * r - x * y))
    }
}

/// `__ieee754_rem_pio2` for `|x| <= 2^19 * pi/2`: `(n, y0, y1)` with
/// `x - n*pi/2 = y0 + y1`; `None` beyond that range.
fn rem_pio2(x: f64) -> Option<(i32, f64, f64)> {
    const NPIO2_HW: [i32; 32] = [
        0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB, 0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB,
        0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
    ];
    const INVPIO2: f64 = 6.366_197_723_675_813_824_33e-01;
    let pio2_1 = from_words(0x3FF921FB, 0x54400000);
    let pio2_1t = from_words(0x3DD0B461, 0x1A626331);
    let pio2_2 = from_words(0x3DD0B461, 0x1A600000);
    let pio2_2t = from_words(0x3BA3198A, 0x2E037073);
    let pio2_3 = from_words(0x3BA3198A, 0x2E000000);
    let pio2_3t = from_words(0x397B839A, 0x252049C1);
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return Some((0, x, 0.0));
    }
    if ix < 0x4002_d97c {
        // |x| < 3pi/4: n = +-1
        return Some(if hx > 0 {
            let mut z = x - pio2_1;
            if ix != 0x3ff9_21fb {
                let y0 = z - pio2_1t;
                (1, y0, (z - y0) - pio2_1t)
            } else {
                z -= pio2_2;
                let y0 = z - pio2_2t;
                (1, y0, (z - y0) - pio2_2t)
            }
        } else {
            let mut z = x + pio2_1;
            if ix != 0x3ff9_21fb {
                let y0 = z + pio2_1t;
                (-1, y0, (z - y0) + pio2_1t)
            } else {
                z += pio2_2;
                let y0 = z + pio2_2t;
                (-1, y0, (z - y0) + pio2_2t)
            }
        });
    }
    if ix > 0x4139_21fb {
        return None;
    }
    let t = x.abs();
    let n = (t * INVPIO2 + 0.5) as i32;
    let f = n as f64;
    let mut r = t - f * pio2_1;
    let mut w = f * pio2_1t;
    let mut y0;
    if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
        y0 = r - w;
    } else {
        let j = ix >> 20;
        y0 = r - w;
        let i = j - ((hi(y0) >> 20) & 0x7ff);
        if i > 16 {
            let t2 = r;
            w = f * pio2_2;
            r = t2 - w;
            w = f * pio2_2t - ((t2 - r) - w);
            y0 = r - w;
            let i = j - ((hi(y0) >> 20) & 0x7ff);
            if i > 49 {
                let t3 = r;
                w = f * pio2_3;
                r = t3 - w;
                w = f * pio2_3t - ((t3 - r) - w);
                y0 = r - w;
            }
        }
    }
    let y1 = (r - y0) - w;
    Some(if hx < 0 { (-n, -y0, -y1) } else { (n, y0, y1) })
}

/// `StrictMath.sin`.
pub fn sin(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return kernel_sin(x, 0.0, 0);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    match rem_pio2(x) {
        Some((n, y0, y1)) => match n & 3 {
            0 => kernel_sin(y0, y1, 1),
            1 => kernel_cos(y0, y1),
            2 => -kernel_sin(y0, y1, 1),
            _ => -kernel_cos(y0, y1),
        },
        None => x.sin(),
    }
}

/// `StrictMath.cos`.
pub fn cos(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return kernel_cos(x, 0.0);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    match rem_pio2(x) {
        Some((n, y0, y1)) => match n & 3 {
            0 => kernel_cos(y0, y1),
            1 => -kernel_sin(y0, y1, 1),
            2 => -kernel_cos(y0, y1),
            _ => kernel_sin(y0, y1, 1),
        },
        None => x.cos(),
    }
}

/// `StrictMath.asin` (`Math.asin` delegates to it).
pub fn asin(x: f64) -> f64 {
    let pio2_hi = from_words(0x3FF921FB, 0x54442D18);
    let pio2_lo = from_words(0x3C91A626, 0x33145C07);
    let pio4_hi = from_words(0x3FE921FB, 0x54442D18);
    let ps0 = from_words(0x3FC55555, 0x55555555);
    let ps1 = from_words(0xBFD4D612, 0x03EB6F7D);
    let ps2 = from_words(0x3FC9C155, 0x0E884455);
    let ps3 = from_words(0xBFA48228, 0xB5688F3B);
    let ps4 = from_words(0x3F49EFE0, 0x7501B288);
    let ps5 = from_words(0x3F023DE1, 0x0DFDF709);
    let qs1 = from_words(0xC0033A27, 0x1C8A2D4B);
    let qs2 = from_words(0x40002AE5, 0x9C598AC8);
    let qs3 = from_words(0xBFE6066C, 0x1B8D0159);
    let qs4 = from_words(0x3FB3B8C5, 0xB12E9282);
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        let lo = x.to_bits() as u32;
        if ((ix - 0x3ff0_0000) as u32 | lo) == 0 {
            return x * pio2_hi + x * pio2_lo;
        }
        return (x - x) / (x - x);
    }
    if ix < 0x3fe0_0000 {
        if ix < 0x3e40_0000 && 1.0e300 + x > 1.0 {
            return x;
        }
        let t = x * x;
        let p = t * (ps0 + t * (ps1 + t * (ps2 + t * (ps3 + t * (ps4 + t * ps5)))));
        let q = 1.0 + t * (qs1 + t * (qs2 + t * (qs3 + t * qs4)));
        let w = p / q;
        return x + x * w;
    }
    let w = 1.0 - x.abs();
    let t = w * 0.5;
    let p = t * (ps0 + t * (ps1 + t * (ps2 + t * (ps3 + t * (ps4 + t * ps5)))));
    let q = 1.0 + t * (qs1 + t * (qs2 + t * (qs3 + t * qs4)));
    let s = t.sqrt();
    let r = if ix >= 0x3FEF_3333 {
        let w = p / q;
        pio2_hi - (2.0 * (s + s * w) - pio2_lo)
    } else {
        let w = with_lo_zero(s);
        let c = (t - w * w) / (s + w);
        let r = p / q;
        let p = 2.0 * s * r - (pio2_lo - 2.0 * c);
        let q = pio4_hi - 2.0 * w;
        pio4_hi - (p - q)
    };
    if hx > 0 {
        r
    } else {
        -r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Java's `h = h * 31 + doubleToRawLongBits(v)` over Lucene's table
    /// inputs (JDK 27, ES 9.5.5's bundled JVM).
    #[test]
    fn tables_match_java_strictmath() {
        let n = (1usize << 11) + 1;
        let (dh, dl) = (4.0 * from_words(0x3FF921FB, 0x54400000) / (n - 1) as f64, 4.0 * from_words(0x3DD0B461, 0x1A626331) / (n - 1) as f64);
        let mut h: i64 = 0;
        for i in 0..n {
            let a = i as f64 * dh + i as f64 * dl;
            h = h.wrapping_mul(31).wrapping_add(sin(a).to_bits() as i64);
            h = h.wrapping_mul(31).wrapping_add(cos(a).to_bits() as i64);
        }
        assert_eq!(h, -6266737676328334204);
        let amax = sin(73.0f64.to_radians());
        assert_eq!(amax.to_bits(), 4606788847030491935);
        let an = (1usize << 13) + 1;
        let ad = amax / (an - 1) as f64;
        let mut h: i64 = 0;
        for i in 0..an {
            h = h.wrapping_mul(31).wrapping_add(asin(i as f64 * ad).to_bits() as i64);
        }
        assert_eq!(h, 581087780319860948);
    }
}
