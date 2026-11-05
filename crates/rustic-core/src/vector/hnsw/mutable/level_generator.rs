//! Pinned-Vespa level generator used by the mutable graph.
//!
//! Reference revision: 0521b00640ef934b359d2fc356f97d84499dd17d
//! https://github.com/vespa-engine/vespa/blob/0521b00640ef934b359d2fc356f97d84499dd17d/searchlib/src/vespa/searchlib/tensor/inv_log_level_generator.h
//! https://github.com/vespa-engine/vespa/blob/0521b00640ef934b359d2fc356f97d84499dd17d/searchlib/src/vespa/searchlib/tensor/hnsw_index.cpp#L710
//!
//! One shared mutex serializes the MT19937_64 draw; the inverse-log calculation
//! happens after releasing it, as in Vespa. A diagnostic ticket records draw
//! order, not insertion order. Parallel scheduling may assign different levels
//! to source keys across graph builds. No keyed/per-node RNG is substituted.
//!
//! The conversion is the observed GCC 13.3/libstdc++ EC2 oracle conversion:
//! `(word as f64) * 2^-64`, with rounded 1.0 replaced by the previous double.
//! It is NOT a high-53-bits truncation. MT words and uniform bits are exact;
//! `ln` and therefore boundary rounding still depend on the platform's libm.
//! The included oracle tests make those differences visible rather than hiding
//! them with tolerances. No cross-platform libm identity is claimed.
//!
//! Defensive POC exception: m=1 produces level 0 but consumes its normal draw.
//! Vespa's formula divides by log(1), so its floating-to-integer result is not a
//! defined contract to port. Production m=16 is covered by the oracle; m=0 errors.
//! The cap is applied here for convenience; pinned Vespa applies it in HnswIndex.

use parking_lot::Mutex;
use std::fmt;

const SEED: u64 = 0x1234_dead_beef_5678;
const STATE_WORDS: usize = 312;
const SHIFT_WORDS: usize = 156;
const MATRIX_A: u64 = 0xb502_6f5a_a966_19e9;
const LOWER_MASK: u64 = 0x0000_0000_7fff_ffff;
const UPPER_MASK: u64 = !LOWER_MASK;
pub(super) const MAX_LEVEL: u32 = 29;

/// Standard 64-bit Mersenne Twister parameters, matching std::mt19937_64.
/// In-place twist order is deliberate: the second half reads already-twisted
/// words, and the final splice uses the new state's first word.
#[derive(Clone)]
struct Mt19937_64 {
    state: [u64; STATE_WORDS],
    index: usize,
}

impl Mt19937_64 {
    fn new(seed: u64) -> Self {
        let mut state = [0; STATE_WORDS];
        state[0] = seed;
        for i in 1..STATE_WORDS {
            let previous = state[i - 1];
            state[i] = 6_364_136_223_846_793_005u64
                .wrapping_mul(previous ^ (previous >> 62))
                .wrapping_add(i as u64);
        }
        Self {
            state,
            index: STATE_WORDS,
        }
    }

    fn next(&mut self) -> u64 {
        if self.index == STATE_WORDS {
            for i in 0..STATE_WORDS {
                let joined =
                    (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % STATE_WORDS] & LOWER_MASK);
                self.state[i] = self.state[(i + SHIFT_WORDS) % STATE_WORDS]
                    ^ (joined >> 1)
                    ^ if joined & 1 != 0 { MATRIX_A } else { 0 };
            }
            self.index = 0;
        }
        let mut word = self.state[self.index];
        self.index += 1;
        word ^= (word >> 29) & 0x5555_5555_5555_5555;
        word ^= (word << 17) & 0x71d6_7fff_eda6_0000;
        word ^= (word << 37) & 0xfff7_eee0_0000_0000;
        word ^= word >> 43;
        word
    }
}

fn uniform_from_word(word: u64) -> f64 {
    // 2^64 is exactly representable. The integer conversion rounds first, as
    // the recorded C++ conversion does; discarding low bits first is different.
    let uniform = (word as f64) * (1.0 / 18_446_744_073_709_551_616.0);
    if uniform >= 1.0 {
        f64::from_bits(1.0f64.to_bits() - 1)
    } else {
        uniform
    }
}

fn capped(raw: u32) -> u32 {
    raw.min(MAX_LEVEL)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LevelGeneratorError {
    InvalidFanout,
    InvalidUniform,
    InvalidLevel,
    DrawCounterExhausted,
}
impl fmt::Display for LevelGeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vespa level generator: {self:?}")
    }
}
impl std::error::Error for LevelGeneratorError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LevelDraw {
    pub ticket: u64,
    pub engine_word: u64,
    pub uniform: f64,
    pub raw_level: u32,
    pub level: u32,
}

struct Stream {
    rng: Mt19937_64,
    next_ticket: u64,
}

pub(super) struct VespaLevelGenerator {
    // None means the explicit m=1 defensive exception, never m=0.
    multiplier: Option<f64>,
    stream: Mutex<Stream>,
}

impl VespaLevelGenerator {
    pub(super) fn new(m: u32) -> Result<Self, LevelGeneratorError> {
        if m == 0 {
            return Err(LevelGeneratorError::InvalidFanout);
        }
        Ok(Self {
            multiplier: (m > 1).then(|| 1.0 / f64::from(m).ln()),
            stream: Mutex::new(Stream {
                rng: Mt19937_64::new(SEED),
                next_ticket: 0,
            }),
        })
    }

    fn calculate(&self, uniform: f64) -> Result<(f64, u32), LevelGeneratorError> {
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err(LevelGeneratorError::InvalidUniform);
        }
        let Some(multiplier) = self.multiplier else {
            return Ok((0.0, 0));
        };
        // Keep pinned multiply-by-inverse order, not division by ln(m).
        let scaled = -(1.0 - uniform).ln() * multiplier;
        if !scaled.is_finite() || scaled < 0.0 || scaled > f64::from(u32::MAX) {
            return Err(LevelGeneratorError::InvalidLevel);
        }
        Ok((scaled, scaled as u32))
    }

    pub(super) fn next_level(&self) -> Result<u32, LevelGeneratorError> {
        self.next_sample().map(|draw| draw.level)
    }

    /// Serialized raw draw with provenance. No external callback runs while the
    /// stream mutex is held. Tickets cannot wrap and fail before consuming RNG.
    pub(super) fn next_sample(&self) -> Result<LevelDraw, LevelGeneratorError> {
        let (ticket, engine_word) = {
            let mut stream = self.stream.lock();
            if stream.next_ticket == u64::MAX {
                return Err(LevelGeneratorError::DrawCounterExhausted);
            }
            let ticket = stream.next_ticket;
            let engine_word = stream.rng.next();
            stream.next_ticket += 1;
            (ticket, engine_word)
        };
        let uniform = uniform_from_word(engine_word);
        let (_, raw_level) = self.calculate(uniform)?;
        Ok(LevelDraw {
            ticket,
            engine_word,
            uniform,
            raw_level,
            level: capped(raw_level),
        })
    }

    pub(super) fn draws(&self) -> u64 {
        self.stream.lock().next_ticket
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Derived ONLY by parsing the saved EC2 C++ oracle JSONL, not by executing
    // this Rust implementation. GCC 13.3, libstdc++ 20240904, no fast math.
    // Source: the Vespa level oracle's draws.jsonl
    // Source SHA256: 575ea702353963dd111278c4ff1bdc12b5bb28423b3a0df4c32fa294400c3fd2
    // Fixture SHA256: 0aef9dfbf43bdcb5d7bbaa8a66c8b6d88c1d07c14b01c11d496c78ae1217c6b2
    // Format, little endian: magic[8], source sha256[32], seed:u64,
    // counts(draws,boundaries,caps,fanouts):4*u32, fanouts:3*u32;
    // draw: raw:u64, uniform_bits:u64, 3*(raw_level:u8,capped:u8);
    // boundary: m:u32, uniform_bits:u64,multiplier_bits:u64,scaled_bits:u64,
    // raw:u32,capped:u32; cap: raw:u32,capped:u32.
    const FIXTURE: &[u8] = include_bytes!("fixtures/vespa_level_gcc13_v1.bin");
    const FANOUTS: [u32; 3] = [2, 16, 32];

    struct Cursor<'a> {
        rest: &'a [u8],
    }
    impl Cursor<'_> {
        fn take<const N: usize>(&mut self) -> [u8; N] {
            let value = self.rest[..N].try_into().unwrap();
            self.rest = &self.rest[N..];
            value
        }
        fn u8(&mut self) -> u8 {
            self.take::<1>()[0]
        }
        fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take())
        }
        fn u64(&mut self) -> u64 {
            u64::from_le_bytes(self.take())
        }
    }

    struct OracleDraw {
        word: u64,
        uniform_bits: u64,
        raw: [u32; 3],
        capped: [u32; 3],
    }
    struct Boundary {
        m: u32,
        uniform_bits: u64,
        multiplier_bits: u64,
        scaled_bits: u64,
        raw: u32,
        capped: u32,
    }
    struct Oracle {
        draws: Vec<OracleDraw>,
        boundaries: Vec<Boundary>,
        caps: Vec<(u32, u32)>,
    }
    fn oracle() -> Oracle {
        let mut input = Cursor { rest: FIXTURE };
        assert_eq!(&input.take::<8>(), b"VLVL0001");
        let digest = input.take::<32>();
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(
            hex,
            "575ea702353963dd111278c4ff1bdc12b5bb28423b3a0df4c32fa294400c3fd2"
        );
        assert_eq!(input.u64(), SEED);
        let count = input.u32();
        let boundary_count = input.u32();
        let cap_count = input.u32();
        assert_eq!(
            (count, boundary_count, cap_count, input.u32()),
            (4096, 25, 5, 3)
        );
        for m in FANOUTS {
            assert_eq!(input.u32(), m);
        }
        let draws = (0..count)
            .map(|_| {
                let word = input.u64();
                let uniform_bits = input.u64();
                let mut raw = [0; 3];
                let mut capped = [0; 3];
                for i in 0..3 {
                    raw[i] = u32::from(input.u8());
                    capped[i] = u32::from(input.u8());
                }
                OracleDraw {
                    word,
                    uniform_bits,
                    raw,
                    capped,
                }
            })
            .collect();
        let boundaries = (0..boundary_count)
            .map(|_| Boundary {
                m: input.u32(),
                uniform_bits: input.u64(),
                multiplier_bits: input.u64(),
                scaled_bits: input.u64(),
                raw: input.u32(),
                capped: input.u32(),
            })
            .collect();
        let caps = (0..cap_count).map(|_| (input.u32(), input.u32())).collect();
        assert!(input.rest.is_empty(), "unparsed oracle fixture bytes");
        Oracle {
            draws,
            boundaries,
            caps,
        }
    }

    #[test]
    fn all_4096_mt_words_and_uniform_bits_match_ec2_oracle() {
        let oracle = oracle();
        let mut engine = Mt19937_64::new(SEED);
        for (index, expected) in oracle.draws.iter().enumerate() {
            let word = engine.next();
            assert_eq!(word, expected.word, "raw word {index}");
            assert_eq!(
                uniform_from_word(word).to_bits(),
                expected.uniform_bits,
                "uniform {index}"
            );
        }
        // The full test crosses thirteen 312-word states, including both sides
        // of the first and second twists, not only a short seed prefix.
        assert!(oracle.draws.len() > 13 * STATE_WORDS);
    }

    #[test]
    fn generator_levels_and_tickets_match_every_oracle_draw_for_each_fanout() {
        let oracle = oracle();
        for (fanout, m) in FANOUTS.into_iter().enumerate() {
            let generator = VespaLevelGenerator::new(m).unwrap();
            for (index, expected) in oracle.draws.iter().enumerate() {
                let actual = generator.next_sample().unwrap();
                assert_eq!(actual.ticket, index as u64);
                assert_eq!(actual.engine_word, expected.word, "m={m}, draw={index}");
                assert_eq!(
                    actual.uniform.to_bits(),
                    expected.uniform_bits,
                    "m={m}, draw={index}"
                );
                assert_eq!(
                    actual.raw_level, expected.raw[fanout],
                    "m={m}, draw={index}"
                );
                assert_eq!(actual.level, expected.capped[fanout], "m={m}, draw={index}");
            }
            assert_eq!(generator.draws(), 4096);
        }
    }

    #[test]
    fn explicit_state_transition_words_around_312_and_624_match_oracle() {
        let oracle = oracle();
        let generator = VespaLevelGenerator::new(16).unwrap();
        let mut checked = Vec::new();
        for index in 0..=625 {
            let draw = generator.next_sample().unwrap();
            if [310, 311, 312, 313, 622, 623, 624, 625].contains(&index) {
                assert_eq!(draw.engine_word, oracle.draws[index].word);
                assert_eq!(draw.uniform.to_bits(), oracle.draws[index].uniform_bits);
                checked.push(index);
            }
        }
        assert_eq!(checked.len(), 8);
    }

    #[test]
    fn boundary_rounding_and_cap_match_observed_cpp_results() {
        let oracle = oracle();
        for expected in oracle.boundaries {
            let generator = VespaLevelGenerator::new(expected.m).unwrap();
            assert_eq!(
                generator.multiplier.unwrap().to_bits(),
                expected.multiplier_bits
            );
            let (scaled, raw) = generator
                .calculate(f64::from_bits(expected.uniform_bits))
                .unwrap();
            assert_eq!(
                scaled.to_bits(),
                expected.scaled_bits,
                "m={}, uniform={:016x}",
                expected.m,
                expected.uniform_bits
            );
            assert_eq!(raw, expected.raw);
            assert_eq!(capped(raw), expected.capped);
        }
        for (raw, expected) in oracle.caps {
            assert_eq!(capped(raw), expected);
        }
        // The recorded m=32 exact mathematical threshold rounds *below* 1.
        // Do not replace observed FP behavior with an idealized floor identity.
        let generator = VespaLevelGenerator::new(32).unwrap();
        assert_eq!(generator.calculate(1.0 - 1.0 / 32.0).unwrap().1, 0);
    }

    #[test]
    fn rounded_one_guard_and_low_bits_match_ieee_conversion_contract() {
        let largest = f64::from_bits(1.0f64.to_bits() - 1);
        assert_eq!(uniform_from_word(0).to_bits(), 0);
        assert_eq!(uniform_from_word(1), 1.0 / 18_446_744_073_709_551_616.0);
        assert_eq!(uniform_from_word(1 << 63), 0.5);
        assert_eq!(uniform_from_word(u64::MAX), largest);
        assert_eq!(uniform_from_word(u64::MAX - 1023), largest);
        assert_eq!(uniform_from_word(u64::MAX - 1024), largest);
        assert_ne!(
            uniform_from_word(1),
            ((1u64 >> 11) as f64) / 9_007_199_254_740_992.0
        );
        for m in FANOUTS {
            assert!(
                VespaLevelGenerator::new(m)
                    .unwrap()
                    .calculate(largest)
                    .is_ok()
            );
        }
    }

    #[test]
    fn concurrent_draw_tickets_form_one_exact_serialized_stream() {
        let oracle = oracle();
        let generator = Arc::new(VespaLevelGenerator::new(16).unwrap());
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let generator = generator.clone();
                std::thread::spawn(move || {
                    (0..512)
                        .map(|_| generator.next_sample().unwrap())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        // No barrier or channel wait: a failed worker cannot strand another.
        let mut joined = Vec::new();
        let mut failed = false;
        for worker in workers {
            match worker.join() {
                Ok(draws) => joined.extend(draws),
                Err(_) => failed = true,
            }
        }
        assert!(!failed, "level-generator worker panicked");
        joined.sort_unstable_by_key(|draw| draw.ticket);
        assert_eq!(joined.len(), oracle.draws.len());
        for (index, (actual, expected)) in joined.iter().zip(&oracle.draws).enumerate() {
            assert_eq!(actual.ticket, index as u64);
            assert_eq!(actual.engine_word, expected.word);
            assert_eq!(actual.uniform.to_bits(), expected.uniform_bits);
            assert_eq!(actual.raw_level, expected.raw[1]);
            assert_eq!(actual.level, expected.capped[1]);
        }
        assert_eq!(generator.draws(), 4096);
    }

    #[test]
    fn invalid_inputs_defensive_m_one_and_counter_exhaustion_are_explicit() {
        assert!(matches!(
            VespaLevelGenerator::new(0),
            Err(LevelGeneratorError::InvalidFanout)
        ));
        let defensive = VespaLevelGenerator::new(1).unwrap();
        let ordinary = VespaLevelGenerator::new(16).unwrap();
        for _ in 0..16 {
            let actual = defensive.next_sample().unwrap();
            let comparison = ordinary.next_sample().unwrap();
            assert_eq!(actual.ticket, comparison.ticket);
            assert_eq!(actual.engine_word, comparison.engine_word);
            assert_eq!((actual.raw_level, actual.level), (0, 0));
        }
        for uniform in [-1.0, 1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                ordinary.calculate(uniform),
                Err(LevelGeneratorError::InvalidUniform)
            );
        }
        let expected_next = {
            let mut stream = ordinary.stream.lock();
            stream.next_ticket = u64::MAX;
            stream.rng.clone().next()
        };
        assert_eq!(
            ordinary.next_level(),
            Err(LevelGeneratorError::DrawCounterExhausted)
        );
        let mut stream = ordinary.stream.lock();
        assert_eq!(stream.next_ticket, u64::MAX);
        assert_eq!(
            stream.rng.next(),
            expected_next,
            "failure consumed RNG word"
        );
    }
}
