//! Token-bucket admission for cache misses (ES ScriptCache). Ingest is exempt.
use super::ScriptContext;
use crate::error::{EsError, Result};
use parking_lot::Mutex;
use std::{collections::BTreeMap, sync::OnceLock, time::{Duration, Instant}};

/// A compilation rate. `text` is how ES prints it (`CompilationRate.toString`:
/// the configured value as written); equality ignores it, as ES's does.
#[derive(Clone, Debug)]
pub(crate) enum Rate { Unlimited, Limited { count: u32, window: Duration, text: std::sync::Arc<str> } }
impl PartialEq for Rate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Rate::Unlimited, Rate::Unlimited) => true,
            (Rate::Limited { count: a, window: x, .. }, Rate::Limited { count: b, window: y, .. }) => a == b && x == y,
            _ => false,
        }
    }
}
/// `script.context.*.max_compilations_rate`: a rate or `unlimited`.
pub(crate) fn parse(value: &str) -> Result<Rate> {
    if value == "unlimited" { return Ok(Rate::Unlimited); }
    parse_rate(value)
}
/// ES `ScriptCache.CompilationRate(String)` with its messages (the general
/// `script.max_compilations_rate` has no `unlimited`).
pub(crate) fn parse_rate(value: &str) -> Result<Rate> {
    let invalid = |reason: String| EsError::illegal_argument(reason);
    if !value.contains('/') || value.starts_with('/') || value.ends_with('/') {
        return Err(invalid(format!("parameter must contain a positive integer and a timevalue, i.e. 10/1m, but was [{value}]")));
    }
    let (count, time) = value.split_once('/').expect("contains a slash");
    let rate = count.parse::<i32>().map_err(|_| invalid(format!("could not parse [{count}] as integer in value [{value}]")))?;
    if rate < 0 { return Err(invalid(format!("rate [{rate}] must be positive"))); }
    let nanos = time_value_nanos(time, "script.max_compilations_rate")?;
    if nanos <= 0 { return Err(invalid(format!("time value [{time}] must be positive"))); }
    if nanos / 1_000_000_000 < 60 { return Err(invalid(format!("time value [{time}] must be at least on a one minute resolution"))); }
    let window = Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX));
    Ok(Rate::Limited { count: rate as u32, window, text: value.into() })
}
/// ES `TimeValue.parseTimeValue` in nanoseconds (`-1` and `0` are magic).
fn time_value_nanos(value: &str, setting: &str) -> Result<i128> {
    let normalized = value.to_lowercase();
    let normalized = normalized.trim();
    let unit = |suffix: &str, nanos: i128| -> Result<i128> {
        let number = normalized[..normalized.len() - suffix.len()].trim();
        match number.parse::<i64>() {
            Ok(n) if n < -1 => Err(EsError::illegal_argument(format!("failed to parse setting [{setting}] with value [{value}] as a time value: negative durations are not supported"))),
            Ok(n) => Ok(n as i128 * nanos),
            Err(_) if number.parse::<f64>().is_ok() => Err(EsError::illegal_argument(format!("failed to parse [{value}], fractional time values are not supported"))),
            Err(_) => Err(EsError::illegal_argument(format!("failed to parse [{value}]"))),
        }
    };
    if normalized.ends_with("nanos") { unit("nanos", 1) }
    else if normalized.ends_with("micros") { unit("micros", 1_000) }
    else if normalized.ends_with("ms") { unit("ms", 1_000_000) }
    else if normalized.ends_with('s') { unit("s", 1_000_000_000) }
    // Case-sensitive: `M` is not minutes.
    else if value.ends_with('m') { unit("m", 60_000_000_000) }
    else if normalized.ends_with('h') { unit("h", 3_600_000_000_000) }
    else if normalized.ends_with('d') { unit("d", 86_400_000_000_000) }
    else if normalized.strip_prefix('-').is_some_and(|r| !r.is_empty() && r.trim_start_matches('0') == "1") { Ok(-1_000_000) }
    else if !normalized.is_empty() && normalized.chars().all(|c| c == '0') { Ok(0) }
    else { Err(EsError::illegal_argument(format!("failed to parse setting [{setting}] with value [{value}] as a time value: unit is missing or unrecognized"))) }
}
struct Bucket { rate: Rate, available: f64, at: Instant }
impl Bucket {
    fn new(rate: Rate, now: Instant) -> Self {
        let available = match &rate { Rate::Unlimited => 0.0, Rate::Limited { count, .. } => *count as f64 };
        Self { rate, available, at: now }
    }
    fn take(&mut self, now: Instant) -> bool {
        let Rate::Limited { count, window, .. } = self.rate else { return true };
        self.available = (self.available + now.saturating_duration_since(self.at).as_secs_f64() * count as f64 / window.as_secs_f64()).min(count as f64);
        self.at = now;
        if self.available < 1.0 { false } else { self.available -= 1.0; true }
    }
}
struct Admission { policy: super::cache_policy::Policy, buckets: BTreeMap<String, Bucket> }
impl Default for Admission {
    fn default() -> Self { Self { policy: super::cache_policy::policy(&BTreeMap::new()).expect("default policy"), buckets: BTreeMap::new() } }
}
impl Admission {
    fn check(&mut self, context: ScriptContext) -> Result<()> {
        let (name, rate) = match &self.policy {
            super::cache_policy::Policy::General(config) => ("", &config.rate),
            super::cache_policy::Policy::Contexts(configs) => (context.name(), &configs[context.name()].rate),
        };
        let bucket = self.buckets.entry(name.into()).or_insert_with(||Bucket::new(rate.clone(), Instant::now()));
        check_bucket(bucket).map_err(|mut e| {
            if !name.is_empty() { e.reason=e.reason.replace("[script.max_compilations_rate]", &format!("[script.context.{name}.max_compilations_rate]")); }
            e
        })
    }
    fn configure(&mut self, next: super::cache_policy::Policy) {
        if self.policy == next { return; }
        match (&self.policy, &next) {
            (super::cache_policy::Policy::Contexts(old),super::cache_policy::Policy::Contexts(new)) => self.buckets.retain(|name,_|old.get(name)==new.get(name)),
            _ => self.buckets.clear(),
        }
        self.policy=next;
    }
}
fn admission() -> &'static Mutex<Admission> {
    static ADMISSION: OnceLock<Mutex<Admission>> = OnceLock::new();
    ADMISSION.get_or_init(||Mutex::new(Admission::default()))
}
fn default_rate() -> Rate { parse_rate("150/5m").expect("default rate") }
pub(crate) fn check(context: ScriptContext) -> Result<()> {
    if !context.compilation_rate_limited() { return Ok(()); }
    #[cfg(test)]
    if let Some(result) = FIXTURE_BUCKET.with(|slot| slot.borrow_mut().as_mut().map(check_bucket)) {
        return result;
    }
    let result = admission().lock().check(context);
    if result.is_err() { super::telemetry::rejected(context); }
    result
}
fn check_bucket(bucket: &mut Bucket) -> Result<()> {
    if bucket.take(Instant::now()) { return Ok(()); }
    let Rate::Limited { text, .. } = &bucket.rate else { unreachable!() };
    Err(EsError::new(429, "circuit_breaking_exception", format!("[script] Too many dynamic script compilations within, max: [{text}]; please use indexed, or scripts with parameters instead; this limit can be changed by the [script.max_compilations_rate] setting"))
        .with("bytes_wanted", 0).with("bytes_limit", 0).with("durability", "TRANSIENT"))
}
/// Cluster settings commit before calling this, transient over persistent over
/// startup settings. Reapplying identical metadata never replenishes tokens.
pub fn configure(node: &crate::node::Node) -> Result<()> {
    let cs = node.cluster_settings.read();
    let ns = node.node_settings.read();
    let mut effective: BTreeMap<String, serde_json::Value> = ns.iter().map(|(k,v)|(k.clone(),v.clone())).collect();
    effective.extend(cs.0.iter().map(|(k,v)|(k.clone(),v.clone())));
    effective.extend(cs.1.iter().map(|(k,v)|(k.clone(),v.clone())));
    let next = super::cache_policy::policy(&effective)?;
    // Same lock order as compilation: cache -> admission. This prevents
    // deadlock and installs cache mode and its budget as one compiler transition.
    let mut caches = super::cache().lock();
    admission().lock().configure(next.clone());
    caches.configure(next);
    Ok(())
}
/// Validate proposed merged settings before the caller commits metadata.
pub fn validate_settings(settings: &BTreeMap<String, serde_json::Value>) -> Result<()> {
    super::cache_policy::policy(settings).map(|_|())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compilation_bucket_refills_without_sleep_or_reset_on_denial() {
        let now = Instant::now();
        let mut b = Bucket::new(parse("2/1m").unwrap(), now);
        assert!(b.take(now)); assert!(b.take(now)); assert!(!b.take(now));
        assert!(!b.take(now + Duration::from_secs(29)));
        assert!(b.take(now + Duration::from_secs(30)));
        assert!(!b.take(now + Duration::from_secs(30)));
        assert!(b.take(now + Duration::from_secs(300)));
        assert!(b.take(now + Duration::from_secs(300)));
        assert!(!b.take(now + Duration::from_secs(300)));
        let mut zero = Bucket::new(parse("0/1m").unwrap(), now);
        assert!(!zero.take(now + Duration::from_secs(1000)));
    }
    #[test]
    fn default_compilation_budget_is_enforced() {
        let now = Instant::now();
        let mut b = Bucket::new(default_rate(), now);
        for _ in 0..150 { assert!(b.take(now)); }
        assert!(!b.take(now));
        assert!(b.take(now + Duration::from_secs(2)));
    }
    #[test]
    fn compilation_rate_validation() {
        for bad in ["-1/1m", "1/0s", "1/59s", "1", "x/1m", "1/1ms", "1/1m/1m", "1/1.5m"] { assert!(parse(bad).is_err(), "{bad}"); }
        assert!(parse("150/5m").is_ok()); assert!(parse("unlimited").is_ok());
        assert!(parse_rate("unlimited").is_err());
        let reason = |value: &str| parse_rate(value).unwrap_err().reason;
        assert_eq!(reason("-1/1m"), "rate [-1] must be positive");
        assert_eq!(reason("x/1m"), "could not parse [x] as integer in value [x/1m]");
        assert_eq!(reason("1"), "parameter must contain a positive integer and a timevalue, i.e. 10/1m, but was [1]");
        assert_eq!(reason("1/0"), "time value [0] must be positive");
        assert_eq!(reason("1/59s"), "time value [59s] must be at least on a one minute resolution");
        assert_eq!(reason("1/1.5m"), "failed to parse [1.5m], fractional time values are not supported");
        assert_eq!(reason("1/1m/1m"), "failed to parse [1m/1m]");
        assert_eq!(reason("1/5"), "failed to parse setting [script.max_compilations_rate] with value [5] as a time value: unit is missing or unrecognized");
        assert_eq!(parse("10/90s").unwrap(), parse("10/90000ms").unwrap());
    }
}

/// Apply startup settings once, but an unsuccessful first open does not consume
/// the opportunity. Serialize concurrent opens until the first successful apply.
pub fn configure_at_open(node: &crate::node::Node) -> Result<()> {
    static CONFIGURED: Mutex<bool> = Mutex::new(false);
    configure_first(&CONFIGURED, || configure(node))
}
fn configure_first(configured: &Mutex<bool>, apply: impl FnOnce() -> Result<()>) -> Result<()> {
    let mut configured = configured.lock();
    if !*configured { apply()?; *configured = true; }
    Ok(())
}
#[cfg(test)]
mod initialization_tests {
    use super::*;
    #[test]
    fn invalid_first_open_does_not_suppress_later_configuration() {
        let configured = Mutex::new(false);
        assert!(configure_first(&configured, || Err(EsError::illegal_argument("invalid"))).is_err());
        assert!(!*configured.lock());
        configure_first(&configured, || Ok(())).unwrap();
        assert!(*configured.lock());
        configure_first(&configured, || panic!("already configured")).unwrap();
    }
}

// Explicit opt-in only: ordinary tests and all production requests keep the
// production budget. Semantic compiler fixtures need independent budgets because
// they compile hundreds of unrelated examples in one process within seconds.
#[cfg(test)]
thread_local! {
    static FIXTURE_BUCKET: std::cell::RefCell<Option<Bucket>> = const { std::cell::RefCell::new(None) };
}
#[cfg(test)]
pub(crate) struct FixtureBudget {
    previous: Option<Bucket>,
    // Restoration must occur on the same thread that installed the fixture.
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}
#[cfg(test)]
impl Drop for FixtureBudget {
    fn drop(&mut self) {
        FIXTURE_BUCKET.with(|slot| *slot.borrow_mut() = self.previous.take());
    }
}
#[cfg(test)]
pub(crate) fn fixture_budget(rate: &str) -> FixtureBudget {
    let next = Bucket::new(parse(rate).expect("explicit fixture compilation rate"), Instant::now());
    let previous = FIXTURE_BUCKET.with(|slot| slot.borrow_mut().replace(next));
    FixtureBudget { previous, _not_send: std::marker::PhantomData }
}
#[cfg(test)]
mod fixture_tests {
    use super::*;
    #[test]
    fn explicit_fixture_budget_still_enforces_cache_misses() {
        let params = serde_json::json!({});
        let cached = "return 19384217 + params.fixtureBudgetValue";
        {
            let _warm = fixture_budget("1/1m");
            super::super::compile(cached, &params, ScriptContext::PainlessTest).unwrap();
        }
        let _closed = fixture_budget("0/1m");
        assert!(super::super::compile(cached, &params, ScriptContext::PainlessTest).is_ok());
        let error = super::super::compile("return 19384218 + params.fixtureBudgetValue", &params, ScriptContext::PainlessTest).err().unwrap();
        assert_eq!(error.status, 429);
        assert!(error.reason.contains("[0/1m]"));
    }
    #[test]
    fn fixture_budget_restores_nested_limits_and_does_not_escape_thread() {
        assert!(FIXTURE_BUCKET.with(|slot| slot.borrow().is_none()));
        {
            let _default = fixture_budget("150/5m");
            for _ in 0..150 { check(ScriptContext::PainlessTest).unwrap(); }
            assert_eq!(check(ScriptContext::PainlessTest).unwrap_err().status, 429);
            {
                let _semantic = fixture_budget("10000/1m");
                check(ScriptContext::PainlessTest).unwrap();
                std::thread::spawn(|| assert!(FIXTURE_BUCKET.with(|slot| slot.borrow().is_none()))).join().unwrap();
            }
            assert_eq!(check(ScriptContext::PainlessTest).unwrap_err().status, 429);
        }
        assert!(FIXTURE_BUCKET.with(|slot| slot.borrow().is_none()));
    }
}

#[cfg(test)]
mod context_admission_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn context_budgets_are_independent_and_unchanged_updates_do_not_refill() {
        let mut settings:BTreeMap<String,serde_json::Value> = [
            ("script.context.score.max_compilations_rate".into(),json!("1/1m")),
            ("script.context.filter.max_compilations_rate".into(),json!("1/1m")),
        ].into_iter().collect();
        let mut admission=Admission::default();
        admission.configure(super::super::cache_policy::policy(&settings).unwrap());
        admission.check(ScriptContext::Score).unwrap();
        admission.check(ScriptContext::Filter).unwrap();
        let denied=admission.check(ScriptContext::Score).unwrap_err();
        assert!(denied.reason.contains("script.context.score.max_compilations_rate"));
        assert!(denied.reason.contains("max: [1/1m]"), "{}", denied.reason);
        assert_eq!(denied.extra.iter().find(|(k, _)| k == "bytes_wanted").map(|(_, v)| v.clone()), Some(serde_json::json!(0)));
        admission.configure(super::super::cache_policy::policy(&settings).unwrap());
        assert!(admission.check(ScriptContext::Score).is_err());
        settings.insert("script.context.filter.max_compilations_rate".into(),json!("2/1m"));
        admission.configure(super::super::cache_policy::policy(&settings).unwrap());
        admission.check(ScriptContext::Filter).unwrap();
        admission.check(ScriptContext::Filter).unwrap();
        assert!(admission.check(ScriptContext::Filter).is_err());
        assert!(admission.check(ScriptContext::Score).is_err());
    }
    #[test]
    fn system_contexts_remain_exempt_even_when_user_budget_is_zero() {
        let _budget=fixture_budget("0/1m");
        assert!(check(ScriptContext::Score).is_err());
        for context in [ScriptContext::Ingest,ScriptContext::IngestConditional,ScriptContext::RuntimeLong,ScriptContext::RuntimeKeyword] {
            check(context).unwrap();
        }
    }
}
