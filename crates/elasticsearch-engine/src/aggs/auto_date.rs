//! `auto_date_histogram` interval choice (ES `AutoDateHistogramAggregationBuilder`
//! rounding ladder), shared by the shard aggregation and the cluster
//! coordinator's cross-shard reduce (`InternalAutoDateHistogram.reduce`).

use crate::date::{add_units, round_down, CalUnit, Tz};
use crate::error::{EsError, Result};

/// The rounding ladder: (unit, multiple, interval suffix), finest first.
pub const LADDER: &[(CalUnit, i64, &str)] = &[
    (CalUnit::Second, 1, "s"),
    (CalUnit::Second, 5, "s"),
    (CalUnit::Second, 10, "s"),
    (CalUnit::Second, 30, "s"),
    (CalUnit::Minute, 1, "m"),
    (CalUnit::Minute, 5, "m"),
    (CalUnit::Minute, 10, "m"),
    (CalUnit::Minute, 30, "m"),
    (CalUnit::Hour, 1, "h"),
    (CalUnit::Hour, 3, "h"),
    (CalUnit::Hour, 12, "h"),
    (CalUnit::Day, 1, "d"),
    (CalUnit::Day, 7, "d"),
    (CalUnit::Month, 1, "M"),
    (CalUnit::Month, 3, "M"),
    (CalUnit::Year, 1, "y"),
    (CalUnit::Year, 5, "y"),
    (CalUnit::Year, 10, "y"),
    (CalUnit::Year, 20, "y"),
    (CalUnit::Year, 50, "y"),
    (CalUnit::Year, 100, "y"),
];

fn rank(u: CalUnit) -> u8 {
    match u {
        CalUnit::Second => 0,
        CalUnit::Minute => 1,
        CalUnit::Hour => 2,
        CalUnit::Day | CalUnit::Week => 3,
        CalUnit::Month | CalUnit::Quarter => 4,
        CalUnit::Year => 5,
    }
}

/// `minimum_interval` as a unit.
pub fn minimum_unit(v: Option<&str>) -> Result<CalUnit> {
    Ok(match v {
        None | Some("second") => CalUnit::Second,
        Some("minute") => CalUnit::Minute,
        Some("hour") => CalUnit::Hour,
        Some("day") => CalUnit::Day,
        Some("month") => CalUnit::Month,
        Some("year") => CalUnit::Year,
        Some(other) => return Err(EsError::illegal_argument(format!("minimum_interval must be one of [second, minute, hour, day, month, year], found [{other}]"))),
    })
}

/// Ladder position of an `interval` string such as `7d` (`None` if unknown).
pub fn ladder_index(interval: &str) -> Option<usize> {
    LADDER.iter().position(|(_, m, s)| format!("{m}{s}") == interval)
}

fn ym(ms: i64, tz: Tz) -> (i64, i64) {
    use chrono::Datelike;
    let l = tz.to_local(ms);
    (l.year() as i64, l.month0() as i64)
}

/// A chosen rounding: bucket `b` starts at `key_of(b)`.
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub unit: CalUnit,
    pub mult: i64,
    pub suffix: &'static str,
    pub first: i64,
    pub tz: Tz,
    /// Number of buckets from `lo` to `hi`.
    pub buckets: u64,
}

impl Choice {
    fn idx(&self, k: i64) -> i64 {
        let r = round_down(k, self.unit, self.tz);
        match self.unit {
            CalUnit::Second | CalUnit::Minute | CalUnit::Hour => (r - self.first) / self.unit.approx_millis(),
            CalUnit::Day | CalUnit::Week => ((r - self.first) as f64 / 86_400_000.0).round() as i64,
            CalUnit::Month | CalUnit::Quarter | CalUnit::Year => {
                let (y0, m0) = ym(self.first, self.tz);
                let (y1, m1) = ym(r, self.tz);
                let months = (y1 - y0) * 12 + (m1 - m0);
                if self.unit == CalUnit::Year { months / 12 } else { months }
            }
        }
    }

    /// Bucket of a timestamp.
    pub fn bucket_of(&self, k: i64) -> i64 {
        self.idx(k) / self.mult
    }

    /// Start of bucket `b`.
    pub fn key_of(&self, b: i64) -> i64 {
        match self.unit {
            CalUnit::Second | CalUnit::Minute | CalUnit::Hour => self.first + b * self.mult * self.unit.approx_millis(),
            _ => add_units(self.first, self.unit, b * self.mult, self.tz),
        }
    }

    /// The response's `interval`, e.g. `7d`.
    pub fn interval(&self) -> String {
        format!("{}{}", self.mult, self.suffix)
    }
}

/// The finest rounding at or above `min_unit` and ladder position `floor`
/// that covers `lo..=hi` in at most `target` buckets (the coarsest if none).
pub fn choose(lo: i64, hi: i64, target: u64, min_unit: CalUnit, floor: usize, tz: Tz) -> Choice {
    let candidates: Vec<(usize, &(CalUnit, i64, &str))> = LADDER.iter().enumerate().filter(|(i, (u, _, _))| *i >= floor && rank(*u) >= rank(min_unit)).collect();
    let make = |c: &(CalUnit, i64, &'static str)| {
        let mut ch = Choice { unit: c.0, mult: c.1, suffix: c.2, first: round_down(lo, c.0, tz), tz, buckets: 0 };
        ch.buckets = (ch.bucket_of(hi) + 1).max(1) as u64;
        ch
    };
    let last = candidates.last().map(|(_, c)| *c).unwrap_or(&LADDER[LADDER.len() - 1]);
    for (_, c) in &candidates {
        let ch = make(c);
        if ch.buckets <= target {
            return ch;
        }
    }
    make(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400_000;

    #[test]
    fn finest_rounding_that_fits_and_respects_the_floor() {
        // 2023-01-01 .. 2023-01-10: ten days fit in 10 daily buckets.
        let lo = 1_672_531_200_000;
        let c = choose(lo, lo + 9 * DAY, 10, CalUnit::Second, 0, Tz::UTC);
        assert_eq!(c.interval(), "1d");
        assert_eq!(c.buckets, 10);
        assert_eq!(c.key_of(c.bucket_of(lo + 3 * DAY + 5)), lo + 3 * DAY);
        // Fewer buckets allowed: weekly.
        assert_eq!(choose(lo, lo + 9 * DAY, 2, CalUnit::Second, 0, Tz::UTC).interval(), "7d");
        // A shard already at 7d keeps the merged result at least that coarse.
        let floor = ladder_index("7d").unwrap();
        assert_eq!(choose(lo, lo + 9 * DAY, 10, CalUnit::Second, floor, Tz::UTC).interval(), "7d");
        assert_eq!(ladder_index("1M"), Some(13));
    }
}
