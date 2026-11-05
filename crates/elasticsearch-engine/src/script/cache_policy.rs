//! ScriptService cache modes. All mutation happens under the compiler cache lock;
//! admission configuration is installed while that lock is held as well.
use super::{rate_limit::{self, Rate}, CacheKey, Program, ScriptContext};
use crate::error::{EsError, Result};
use serde_json::Value;
use std::{collections::{BTreeMap, HashMap}, num::NonZeroUsize, sync::Arc, time::{Duration, Instant}};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CacheConfig { pub size: usize, pub expire: Duration, pub rate: Rate }
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Policy { General(CacheConfig), Contexts(BTreeMap<String, CacheConfig>) }

pub fn context_setting(key: &str) -> Option<(&str, &str)> {
    let rest = key.strip_prefix("script.context.")?;
    let (context, suffix) = rest.rsplit_once('.')?;
    matches!(suffix, "max_compilations_rate" | "cache_max_size" | "cache_expire").then_some((context, suffix))
}
fn number(value: &Value, key: &str) -> Result<usize> {
    value.as_u64().or_else(||value.as_str()?.parse().ok()).filter(|n| *n <= i32::MAX as u64).map(|n|n as usize)
        .ok_or_else(|| EsError::illegal_argument(format!("Failed to parse value [{value}] for setting [{key}] must be a non-negative integer")))
}
fn duration(value: &Value, key: &str) -> Result<Duration> {
    let text = value.as_str().map(str::to_owned).unwrap_or_else(||value.to_string());
    crate::date::parse_time_value(&text).filter(|n| *n >= 0).map(|n|Duration::from_millis(n as u64))
        .ok_or_else(||EsError::illegal_argument(format!("Failed to parse value [{value}] for setting [{key}] as a non-negative time value")))
}
fn text(value: &Value) -> String { value.as_str().map(str::to_owned).unwrap_or_else(|| value.to_string()) }
pub fn validate_value(key: &str, value: &Value) -> Result<()> {
    if value.is_null() { return Ok(()); }
    let suffix = if let Some((context, suffix)) = context_setting(key) {
        if ScriptContext::from_name(context).is_none() { return Err(EsError::illegal_argument(format!("Context [{context}] doesn't exist for setting [{key}]"))); }
        suffix
    } else { key };
    match suffix {
        "script.max_compilations_rate" if value.as_str() == Some("use-context") => {},
        "script.max_compilations_rate" => { rate_limit::parse_rate(&text(value))?; },
        "max_compilations_rate" => { rate_limit::parse(&text(value))?; },
        "script.cache.max_size" | "cache_max_size" => { number(value,key)?; },
        "script.cache.expire" | "cache_expire" => { duration(value,key)?; },
        _ => {},
    }
    Ok(())
}
pub(super) fn policy(settings: &BTreeMap<String, Value>) -> Result<Policy> {
    let mut contexts = Vec::new();
    for (key,value) in settings { validate_value(key,value)?; if context_setting(key).is_some() { contexts.push(key.clone()); } }
    let general = settings.get("script.max_compilations_rate");
    let explicit_context = general.and_then(Value::as_str) == Some("use-context");
    // ES `ContextSettings`: keys grouped by kind, each sorted by context.
    let keys_of = |suffix: &str| {
        let mut keys: Vec<(&str, &String)> = contexts.iter().filter_map(|k| context_setting(k).filter(|(_, s)| *s == suffix).map(|(c, _)| (c, k))).collect();
        keys.sort();
        keys.into_iter().map(|(_, k)| k.clone()).collect::<Vec<_>>()
    };
    let rate_keys = keys_of("max_compilations_rate");
    if general.is_some() && !explicit_context && !contexts.is_empty() {
        let mut keys = rate_keys.clone();
        keys.extend(keys_of("cache_max_size"));
        keys.extend(keys_of("cache_expire"));
        return Err(EsError::illegal_argument(format!("Context cache settings [{}] are incompatible with [script.max_compilations_rate] set to non-default value [{}]. Either remove the incompatible settings (recommended) or set [script.max_compilations_rate] to [use-context] to use per-context settings", keys.join(","), text(general.unwrap()))));
    }
    let disabled = match settings.get("script.disable_max_compilations_rate") {
        None => false,
        Some(v) if v.as_bool()==Some(true) || v.as_str()==Some("true") => true,
        Some(v) if v.as_bool()==Some(false) || v.as_str()==Some("false") => false,
        Some(v) => return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] for setting [script.disable_max_compilations_rate] as a boolean"))),
    };
    if disabled && !rate_keys.is_empty() {
        return Err(EsError::illegal_argument(format!("Cannot set custom context compilation rates [{}] if compile rates disabled via [script.disable_max_compilations_rate]", rate_keys.join(", "))));
    }
    if disabled && general.is_some() && !explicit_context {
        return Err(EsError::illegal_argument(format!("Cannot set custom general compilation rates [script.max_compilations_rate] to [{}] if compile rates disabled via [script.disable_max_compilations_rate]", text(general.unwrap()))));
    }
    let size = settings.get("script.cache.max_size").map(|v|number(v,"script.cache.max_size")).transpose()?;
    let expire = settings.get("script.cache.expire").map(|v|duration(v,"script.cache.expire")).transpose()?.unwrap_or_default();
    if !explicit_context && contexts.is_empty() {
        let rate = if disabled { Rate::Unlimited } else { rate_limit::parse_rate(&general.map(text).unwrap_or_else(|| "150/5m".into()))? };
        return Ok(Policy::General(CacheConfig { size:size.unwrap_or(3000),expire,rate }));
    }
    let mut configs = BTreeMap::new();
    for (name,_) in super::CONTEXT_NAMES {
        let key = |suffix|format!("script.context.{name}.{suffix}");
        let context = ScriptContext::from_name(name).expect("registered context");
        let rate = if disabled || !context.compilation_rate_limited() { Rate::Unlimited } else { rate_limit::parse(&settings.get(&key("max_compilations_rate")).map(text).unwrap_or_else(|| "150/5m".into()))? };
        configs.insert(name.to_string(), CacheConfig {
            size:settings.get(&key("cache_max_size")).map(|v|number(v,&key("cache_max_size"))).transpose()?.or(size).unwrap_or(context.default_cache_size()),
            expire:settings.get(&key("cache_expire")).map(|v|duration(v,&key("cache_expire"))).transpose()?.unwrap_or(expire), rate,
        });
    }
    Ok(Policy::Contexts(configs))
}

pub(super) struct CachedPrograms {
    config: CacheConfig,
    entries: lru::LruCache<CacheKey,(Arc<Program>,Instant)>,
}
impl CachedPrograms {
    fn new(config:CacheConfig) -> Self { Self { entries:lru::LruCache::new(NonZeroUsize::new(config.size.max(1)).unwrap()),config } }
    pub fn get(&mut self,key:&CacheKey)->Option<&Arc<Program>> {
        if self.entries.peek(key).is_some_and(|(_,at)| !self.config.expire.is_zero() && at.elapsed() >= self.config.expire) {
            self.entries.pop(key); super::telemetry::evicted(key.1);
            super::EVICTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.entries.get_mut(key).map(|(program,at)| { *at=Instant::now(); &*program })
    }
    pub fn len(&self)->usize { self.entries.len() }
    pub fn cap(&self)->NonZeroUsize { self.entries.cap() }
    pub fn peek_lru(&self)->Option<(&CacheKey,&Arc<Program>)> { self.entries.peek_lru().map(|(key,(program,_))|(key,program)) }
    pub fn put(&mut self,key:CacheKey,program:Arc<Program>) {
        if self.config.size > 0 { self.entries.put(key,(program,Instant::now())); }
    }
}
pub(super) struct Caches { policy:Policy, caches:HashMap<String,CachedPrograms>, pub(super) inflight:HashMap<CacheKey,Arc<super::Pending>> }
impl Default for Caches {
    fn default()->Self { Self { policy:policy(&BTreeMap::new()).expect("default script policy"), caches:HashMap::new(), inflight:HashMap::new() } }
}
impl Caches {
    pub(super) fn per_context(&self) -> bool { matches!(self.policy, Policy::Contexts(_)) }
    pub fn context(&mut self,context:ScriptContext)->&mut CachedPrograms {
        let (name,config)=match &self.policy { Policy::General(config)=>("",config), Policy::Contexts(configs)=>(context.name(),&configs[context.name()]) };
        self.caches.entry(name.into()).or_insert_with(||CachedPrograms::new(config.clone()))
    }
    pub fn configure(&mut self,next:Policy) {
        if self.policy == next { return; }
        // Retain unrelated context caches on a per-context update.
        if let (Policy::Contexts(old),Policy::Contexts(new))=(&self.policy,&next) {
            self.caches.retain(|name,_|old.get(name)==new.get(name));
        } else { self.caches.clear(); }
        self.policy=next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn settings(value:Value)->BTreeMap<String,Value> { value.as_object().unwrap().iter().map(|(k,v)|(k.clone(),v.clone())).collect() }
    #[test]
    fn context_policy_rejects_conflicts_and_unknown_names() {
        assert!(matches!(policy(&settings(json!({}))).unwrap(),Policy::General(_)));
        assert!(matches!(policy(&settings(json!({"script.max_compilations_rate":"use-context"}))).unwrap(),Policy::Contexts(_)));
        assert!(matches!(policy(&settings(json!({"script.context.score.cache_max_size":2}))).unwrap(),Policy::Contexts(_)));
        for invalid in [
            json!({"script.max_compilations_rate":"2/1m","script.context.score.cache_max_size":2}),
            json!({"script.context.no_such_context.max_compilations_rate":"1/1m"}),
            json!({"script.context.score.cache_expire":"-1s"}),
            json!({"script.context.score.cache_max_size":-1}),
            json!({"script.disable_max_compilations_rate":true,"script.context.score.max_compilations_rate":"1/1m"}),
        ] { assert!(policy(&settings(invalid.clone())).is_err(),"{invalid}"); }
        let reason=|value:Value| policy(&settings(value)).unwrap_err().reason;
        assert_eq!(reason(json!({"script.max_compilations_rate":"2/1m","script.context.score.cache_max_size":2,"script.context.filter.max_compilations_rate":"1/1m"})),
            "Context cache settings [script.context.filter.max_compilations_rate,script.context.score.cache_max_size] are incompatible with [script.max_compilations_rate] set to non-default value [2/1m]. Either remove the incompatible settings (recommended) or set [script.max_compilations_rate] to [use-context] to use per-context settings");
        assert_eq!(reason(json!({"script.disable_max_compilations_rate":true,"script.context.score.max_compilations_rate":"1/1m","script.context.filter.max_compilations_rate":"1/1m"})),
            "Cannot set custom context compilation rates [script.context.filter.max_compilations_rate, script.context.score.max_compilations_rate] if compile rates disabled via [script.disable_max_compilations_rate]");
        assert_eq!(reason(json!({"script.disable_max_compilations_rate":true,"script.max_compilations_rate":"10/90s"})),
            "Cannot set custom general compilation rates [script.max_compilations_rate] to [10/90s] if compile rates disabled via [script.disable_max_compilations_rate]");
        assert!(policy(&settings(json!({"script.max_compilations_rate":"unlimited"}))).is_err());
        // Like ES, a context setting alone selects the (deprecated) context cache.
        assert!(matches!(policy(&settings(json!({"script.context.score.max_compilations_rate":"1/1m"}))).unwrap(),Policy::Contexts(_)));
        let Policy::Contexts(configs)=policy(&settings(json!({"script.max_compilations_rate":"use-context"}))).unwrap() else { unreachable!() };
        assert_eq!(configs["score"].rate,rate_limit::parse("150/5m").unwrap());
        assert_eq!(configs["ingest"].size,200);
        assert_eq!(configs["long_field"].rate,Rate::Unlimited);
        assert_eq!(configs["processor_conditional"].rate,Rate::Unlimited);
    }
    #[test]
    fn context_caches_do_not_evict_each_other_and_updates_retain_other_contexts() {
        let mut caches=Caches::default();
        let mut config=settings(json!({"script.max_compilations_rate":"use-context","script.context.score.cache_max_size":1,"script.context.filter.cache_max_size":1}));
        caches.configure(policy(&config).unwrap());
        let program=super::super::compile::compile_arc("return 1",ScriptContext::Score).unwrap();
        let score=("one".into(),ScriptContext::Score);
        let filter=("one".into(),ScriptContext::Filter);
        caches.context(ScriptContext::Score).put(score.clone(),program.clone());
        caches.context(ScriptContext::Filter).put(filter.clone(),program.clone());
        caches.context(ScriptContext::Score).put(("two".into(),ScriptContext::Score),program);
        assert!(caches.context(ScriptContext::Score).get(&score).is_none());
        assert!(caches.context(ScriptContext::Filter).get(&filter).is_some());
        config.insert("script.context.score.cache_max_size".into(),json!(2));
        caches.configure(policy(&config).unwrap());
        assert!(caches.context(ScriptContext::Filter).get(&filter).is_some());
        assert_eq!(caches.context(ScriptContext::Score).len(),0);
    }
    #[test]
    fn zero_cache_capacity_and_expiration_recompile_instead_of_returning_stale_programs() {
        let program=super::super::compile::compile_arc("return 1",ScriptContext::Score).unwrap();
        let key=("one".into(),ScriptContext::Score);
        let mut cache=CachedPrograms::new(CacheConfig {size:0,expire:Duration::ZERO,rate:Rate::Unlimited});
        cache.put(key.clone(),program.clone());
        assert!(cache.get(&key).is_none());
        let mut cache=CachedPrograms::new(CacheConfig {size:1,expire:Duration::from_secs(1),rate:Rate::Unlimited});
        cache.put(key.clone(),program);
        cache.entries.peek_mut(&key).unwrap().1=Instant::now()-Duration::from_secs(2);
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.len(),0);
    }
}
