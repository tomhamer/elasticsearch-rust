//! `scripted_metric` (`ScriptedMetricAggregator`): per shard (each index
//! stands for one) `init_script` builds a `state` map, `map_script` runs per
//! matching doc (`doc`, `_score`), `combine_script` turns the state into the
//! shard result and `reduce_script` folds the shard results (`states`) into
//! `value`.

use super::*;
use crate::script::{compile_json, CompiledScript, ScriptContext, Val};

// Conservative live-storage bounds for JSON -> runtime conversion. Arrays
// reserve their exact length; maps include dense entries, hash indices and
// resize overlap. Strings include UTF16 expansion and Rc bookkeeping.
fn input_bytes(value: &Value) -> u64 {
    let base = 256u64;
    match value {
        Value::Array(values) => values.iter().fold(base, |n,v| n.saturating_add(input_bytes(v))),
        Value::Object(values) => values.iter().fold(base, |n,(k,v)| n.saturating_add(256).saturating_add((k.len() as u64).saturating_mul(4)).saturating_add(input_bytes(v))),
        Value::String(value) => base.saturating_add((value.len() as u64).saturating_mul(4)),
        _ => base,
    }
}
fn source_input_bytes(value: &crate::source_json::Source) -> u64 {
    use crate::source_json::Source;
    match value {
        Source::Object(values) => values.iter().fold(256u64, |n,(k,v)| n.saturating_add(256).saturating_add((k.len() as u64).saturating_mul(4)).saturating_add(source_input_bytes(v))),
        Source::Array(values) => values.iter().fold(256u64, |n,v| n.saturating_add(source_input_bytes(v))),
        Source::Scalar(raw) => 128u64.saturating_add((raw.get().len() as u64).saturating_mul(8)),
    }
}
fn reserve_inputs(bytes: u64) -> Result<crate::memory::Reservation> {
    crate::memory::breakers().reserve_request(bytes, "<scripted_metric inputs>")
}
/// Reserve serialization trees/text before Source/JSON rendering.
fn reserve_outputs(values: &[Val]) -> Result<crate::memory::Reservation> {
    let mut bytes=512u64;
    for value in values {
        bytes=bytes.saturating_add(crate::script::allocation::serialization_bound(value).map_err(|e|{let e=*e;e.breaker.unwrap_or_else(||EsError::illegal_argument(e.msg))})?);
    }
    crate::memory::breakers().reserve_request(bytes,"<scripted_metric serialization>")
}

fn config_inputs(config: &Map<String, Value>) -> Result<crate::memory::Reservation> {
    let environment = crate::script::env();
    let bytes = config.values().fold(0u64, |n,value| {
        let raw = value.as_u64().and_then(|slot| environment.as_ref().and_then(|env| env.raw_search_scripts.get(slot as usize)));
        n.saturating_add(raw.map(source_input_bytes).unwrap_or_else(|| input_bytes(value)))
    });
    // Phase compilation owns simultaneous JSON/Source/wrapper copies before
    // executor construction. Reserve those copies before phase() touches them.
    reserve_inputs(bytes.saturating_mul(4))
}
#[cfg(test)]
thread_local! {
    static PHASE_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static OUTPUT_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn executor_inputs(cs: &CompiledScript) -> Result<crate::memory::Reservation> {
    let program = &cs.program;
    let slots = program.nslots.saturating_add(program.param_names.len()).saturating_add(program.strings.len());
    let bytes = input_bytes(&cs.params)
        .saturating_add(cs.source_params.as_deref().map(source_input_bytes).unwrap_or(0))
        .saturating_add((slots as u64).saturating_mul(std::mem::size_of::<Val>() as u64))
        .saturating_add(program.strings.iter().map(|s| (s.len() as u64).saturating_mul(4).saturating_add(64)).fold(0u64, u64::saturating_add));
    reserve_inputs(bytes)
}

/// Compile a phase script with the aggregation's `params` merged into its
/// own (ES `ScriptedMetricAggregatorFactory.mergeParams`: a name used in
/// both is an error). `Ok(None)` when the script is absent.
fn phase(o: &Map<String, Value>, key: &str, ctx: ScriptContext) -> Result<Option<CompiledScript>> {
    #[cfg(test)]
    PHASE_COPIES.with(|count| count.set(count.get() + 1));
    let v = match o.get(key) {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(s)) => json!({ "source": s }),
        Some(other) => other.clone(),
    };
    use crate::source_json::Source;
    let (metadata,raw)=crate::script::search_script_source(&v)?;
    let mut script=raw.unwrap_or_else(||Source::from_value(&metadata));
    if let Some(params)=o.get("params") {
        let params=crate::script::search_raw_slot(params).unwrap_or_else(||Source::from_value(params));
        if let Source::Object(params)=params {
            if let Source::Object(script)=&mut script {
                let target=script.entry("params".into()).or_insert_with(||Source::Object(Default::default()));
                if let Source::Object(target)=target {
                    for (key,value) in params {
                        if target.contains_key(&key) {return Err(EsError::illegal_argument(format!("Parameter name \"{key}\" used in both aggregation and script parameters")));}
                        target.insert(key,value);
                    }
                }
            }
        }
    }
    let wrapper=Source::Object(std::collections::BTreeMap::from([("script".into(),script.clone())]));
    let (metadata,_)=crate::source_json::update_request(&wrapper.text())?;
    crate::script::compile_json_source(&metadata["script"],Some(&script),ctx,None).map(Some)
}

/// A required phase script (ES `doBuild`: `[combine_script] must not be
/// null: [name]`).
fn required(o: &Map<String, Value>, key: &str, ctx: ScriptContext, name: &str) -> Result<CompiledScript> {
    phase(o, key, ctx)?.ok_or_else(|| EsError::illegal_argument(format!("[{key}] must not be null: [{name}]")))
}

/// Run the reduce phase over the combined shard states (`states`).
pub(super) fn reduce(name: &str, o: &Map<String, Value>, states: Vec<Val>) -> Result<Value> {
    let _allocations = crate::script::allocation::Scope::enter();
    let _config_inputs = config_inputs(o)?;
    let cs = required(o, "reduce_script", ScriptContext::MetricReduce, name)?;
    let _inputs = executor_inputs(&cs)?;
    let mut exec = cs.executor(None, None);
    exec.set_ctx(Val::new_list(states));
    let v = exec.run().map_err(|e| cs.error(&e))?;
    let _output = reserve_outputs(std::slice::from_ref(&v))?;
    #[cfg(test)]
    OUTPUT_COPIES.with(|count| count.set(count.get() + 1));
    let source=crate::script::source_to_json(&v)?;
    let result=match serde_json::from_str::<Value>(&source.text()) {
        Ok(value)=>json!({"value":value}),
        Err(_)=>json!({"value":null,"__p":{"utf16_metric_value":source.text()}}),
    };
    // Part of the aggregation tree a reduce (or the response) charges whole.
    _output.retain_partial_for_request();
    Ok(result)
}

/// `reduce_script` of `scripted_metric` `o` over combine `states` gathered
/// from several nodes (a cluster coordinator's reduce): `{"value": ...}`.
pub fn reduce_states(name: &str, o: &Map<String, Value>, states: &[Value]) -> Result<Value> {
    let _allocations = crate::script::allocation::Scope::enter();
    let _inputs = reserve_inputs(states.iter().map(input_bytes).fold(0u64, u64::saturating_add))?;
    reduce(name, o, states.iter().map(Val::from_json).collect())
}

/// Decode the engine-owned state envelope before coordinator reduction. User
/// state values themselves are never interpreted as transport metadata.
pub fn reduce_source_states(name:&str,o:&Map<String,Value>,values:&[&Value])->Result<Value> {
    let _allocations = crate::script::allocation::Scope::enter();
    let _inputs = reserve_inputs(values.iter().map(|v| input_bytes(v).saturating_add(v.pointer("/__p/utf16_metric_value").and_then(Value::as_str).map(|raw| (raw.len() as u64).saturating_mul(256)).unwrap_or(0))).fold(0u64, u64::saturating_add))?;
    let mut states=Vec::new();
    for value in values {
        if let Some(raw)=value.pointer("/__p/utf16_metric_value").and_then(Value::as_str){
            let source=crate::source_json::Source::parse(raw)?;
            let crate::source_json::Source::Array(values)=source else{return Err(EsError::new(500,"illegal_state_exception","invalid scripted metric combined states"));};
            states.extend(values.iter().map(crate::script::source_from_json));
        }else{states.extend(value.get("value").and_then(Value::as_array).into_iter().flatten().map(Val::from_json));}
    }
    reduce(name,o,states)
}

/// Render only aggregation result envelopes; never walk inside a user's
/// metric value or metadata map. The raw payload is validated on construction.
pub fn has_raw_value(value:&Value)->bool {
    fn result(value:&Value)->bool {
        match value {
            Value::Object(m)=>m.get("__p").and_then(|p|p.get("utf16_metric_value").or_else(||p.get("raw_json"))).is_some()
                || m.iter().filter(|(key,_)|key.as_str()!="meta"&&key.as_str()!="key"&&(key.as_str()!="value"||m.contains_key("doc_count"))).any(|(_,v)|result(v)),
            Value::Array(a)=>a.iter().any(result),
            _=>false,
        }
    }
    // The outer map contains aggregation names, including "value"/"meta".
    value.as_object().is_some_and(|m|m.values().any(result))
}

pub fn render_values(value:&Value)->crate::source_json::Source {
    use crate::source_json::Source;
    fn walk(value:&Value)->Source {
        match value {
            // A `top_hits` hit kept as raw text (`search::hit_value`).
            Value::Object(m) if m.get("__p").and_then(|p|p.get("raw_json")).is_some_and(Value::is_string) => {
                Source::parse(m["__p"]["raw_json"].as_str().unwrap_or("{}")).expect("rendered hit")
            }
            Value::Object(m)=>{
                let mut out=std::collections::BTreeMap::new();
                for (key,value) in m {
                    if key=="__p" {continue;}
                    let opaque=key=="meta"||key=="key"||(key=="value"&&!m.contains_key("doc_count"));
                    out.insert(key.clone(),if opaque{Source::from_value(value)}else{walk(value)});
                }
                if let Some(raw)=m.get("__p").and_then(|p|p.get("utf16_metric_value")).and_then(Value::as_str){
                    out.insert("value".into(),Source::parse(raw).expect("validated metric output"));
                }
                Source::Object(out)
            },
            Value::Array(a)=>Source::Array(a.iter().map(walk).collect()),
            _=>Source::from_value(value),
        }
    }
    match value.as_object(){Some(m)=>Source::Object(m.iter().map(|(name,result)|(name.clone(),walk(result))).collect()),None=>Source::from_value(value)}
}

/// Reduce of partial (per-shard) results: their `__p.states` concatenated.
pub(super) fn reduce_partials(name: &str, o: &Map<String, Value>, vals: &[&Value]) -> Result<Value> {
    let _allocations = crate::script::allocation::Scope::enter();
    let _inputs = reserve_inputs(vals.iter().map(|v| input_bytes(v).saturating_add(v.pointer("/__p/source_states").and_then(Value::as_str).map(|raw| (raw.len() as u64).saturating_mul(256)).unwrap_or(0))).fold(0u64, u64::saturating_add))?;
    let mut states=Vec::new();
    for value in vals {
        if let Some(raw)=value.pointer("/__p/source_states").and_then(Value::as_str) {
            let source=crate::source_json::Source::parse(raw)?;
            let crate::source_json::Source::Array(values)=source else{return Err(EsError::new(500,"illegal_state_exception","invalid scripted metric shard states"));};
            states.extend(values.iter().map(crate::script::source_from_json));
        } else {states.extend(value.pointer("/__p/states").and_then(Value::as_array).into_iter().flatten().map(Val::from_json));}
    }
    reduce(name, o, states)
}

impl Ctx<'_, '_> {
    pub(super) fn scripted_metric(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let _allocations = crate::script::allocation::Scope::enter();
        let _config_inputs = config_inputs(o)?;
        let map = phase(o, "map_script", ScriptContext::MetricMap)?.ok_or_else(|| EsError::illegal_argument(format!("[mapScript] must not be null: [{name}]")))?;
        let init = phase(o, "init_script", ScriptContext::MetricInit)?;
        let combine = required(o, "combine_script", ScriptContext::MetricCombine, name)?;
        if matches!(o.get("reduce_script"), None | Some(Value::Null)) {
            return Err(EsError::illegal_argument(format!("[reduce_script] must not be null: [{name}]")));
        }
        // Shards (indices) in segment order.
        let mut shards: Vec<&str> = Vec::new();
        for s in self.segs.iter() {
            if !shards.contains(&s.index) {
                shards.push(s.index);
            }
        }
        let mut states: Vec<Val> = Vec::new();
        for ix in shards {
            let state = Val::new_map(Default::default());
            let env = crate::script::index_env(Some(ix));
            let (lookup, runtime) = match &env {
                Some(e) => (Some(e.lookup.clone()), e.runtime.clone()),
                None => (self.segs.iter().find(|s| s.index == ix).map(|s| Arc::new(s.lookup.clone())), None),
            };
            if let Some(cs) = &init {
                let _inputs = executor_inputs(cs)?;
                let mut e = cs.executor(lookup.clone(), runtime.clone());
                e.set_ctx(state.clone());
                e.run().map_err(|err| cs.error(&err))?;
            }
            let _map_inputs = executor_inputs(&map)?;
            let mut e = map.executor(lookup.clone(), runtime.clone());
            e.set_ctx(state.clone());
            // The map phase is the aggregation's collection: it keeps to the
            // search's deadline (ES times it out through its collector) and
            // stops once the task is cancelled. A timed out map phase leaves
            // a partial state; combine and reduce still run, and the search
            // reports `timed_out`.
            let cancel = crate::script::env().and_then(|env| env.cancel.clone());
            e.rt.cancel = cancel.clone();
            let mut mapped = 0u32;
            'map: for (si, v) in docs.iter().enumerate() {
                if self.segs[si].index != ix || v.is_empty() {
                    continue;
                }
                // SAFETY: the executor is used only while `self.segs` holds the leaf.
                unsafe { e.bind_leaf(&self.segs[si].leaf) };
                for &(d, sc) in v {
                    // Per doc: a flag load (scripts are far costlier); the
                    // clock and the task every 256 docs.
                    if let Some(c) = &cancel {
                        if c.stopped() {
                            break 'map;
                        }
                    }
                    mapped = mapped.wrapping_add(1);
                    if mapped % 256 == 0 {
                        crate::search::check_cancelled()?;
                        if let Some(c) = cancel.as_ref().filter(|c| c.expired()) {
                            c.timed_out.store(true, std::sync::atomic::Ordering::Relaxed);
                            break 'map;
                        }
                    }
                    e.set_doc(d);
                    e.set_score(sc as f64);
                    match e.run() {
                        Ok(_) => {}
                        Err(err) if crate::script::runtime::Cancel::absorb(&cancel, &err) => break 'map,
                        Err(err) => return Err(map.error(&err)),
                    }
                }
            }
            // A task cancelled meanwhile answers its cancellation. Combine
            // runs whatever the deadline (it is not collection).
            crate::search::check_cancelled()?;
            let _combine_inputs = executor_inputs(&combine)?;
            let mut c = combine.executor(lookup, runtime);
            c.set_ctx(state.clone());
            states.push(c.run().map_err(|err| combine.error(&err))?);
        }
        if partial::active() {
            // Per-shard search of a multi-shard index: the coordinator
            // (`agg_partial`) reduces every shard's states together.
            let _output = reserve_outputs(&states)?;
            let source=crate::source_json::Source::Array(states.iter().map(crate::script::source_to_json).collect::<Result<_>>()?);
            let result=match serde_json::from_str::<Value>(&source.text()) {
                Ok(states)=>json!({"value":null,"__p":{"states":states}}),
                Err(_)=>json!({"value":null,"__p":{"source_states":source.text()}}),
            };
            // Released when the coordinator's reduce consumes it.
            _output.retain_partial_for_request();
            return Ok(result);
        }
        reduce(name, o, states)
    }
}

#[cfg(test)]
mod utf16_tests {
    use super::*;
    #[test]
    fn metric_names_are_not_treated_as_user_payload() {
        for name in ["value","meta","key","__p"] {
            let value=json!({name:{"value":null,"__p":{"utf16_metric_value":r#""\ud800""#}}});
            assert!(has_raw_value(&value));
            assert!(render_values(&value).get(name).unwrap().get("value").unwrap().equivalent(&crate::source_json::Source::parse(r#""\ud800""#).unwrap()));
        }
    }
    #[test]
    fn metric_state_transport_preserves_nested_raw_values() {
        use crate::source_json::Source;
        let user=Source::parse(r#"{"s":"\ud800","__p":{"utf16_metric_value":"user content"}}"#).unwrap();
        let state=crate::script::source_from_json(&user);
        let conf=json!({"reduce_script":"states[0]"});
        let partial=reduce("m",conf.as_object().unwrap(),vec![state.clone()]).unwrap();
        let rendered=render_values(&json!({"m":partial}));
        assert!(rendered.get("m").unwrap().get("value").unwrap().equivalent(&user));
        let shard=reduce("m",json!({"reduce_script":"states"}).as_object().unwrap(),vec![state]).unwrap();
        let merged=reduce_source_states("m",conf.as_object().unwrap(),&[&shard]).unwrap();
        assert!(render_values(&json!({"m":merged})).get("m").unwrap().get("value").unwrap().equivalent(&user));
        let partial=json!({"__p":{"source_states":Source::Array(vec![user.clone()]).text()}});
        let merged=reduce_partials("m",conf.as_object().unwrap(),&[&partial]).unwrap();
        assert!(render_values(&json!({"m":merged})).get("m").unwrap().get("value").unwrap().equivalent(&user));
    }
    #[test]
    fn shared_metric_params_preserve_units_and_duplicate_rejection() {
        use crate::source_json::{Source,search_request,search_scripts};
        let input=r#"{"aggs":{"m":{"scripted_metric":{"params":{"s":"\ud800"},"combine_script":"(int)params.s.charAt(0)"}}}}"#;
        let (metadata,raw)=search_request(input).unwrap();
        let mut environment=crate::script::ScriptEnv::new(None,None);
        environment.raw_search_scripts=search_scripts(&raw.unwrap());
        crate::script::with_env(environment,|| {
            let body=metadata["aggs"]["m"]["scripted_metric"].as_object().unwrap();
            let cs=phase(body,"combine_script",ScriptContext::MetricCombine).unwrap().unwrap();
            assert_eq!(cs.executor(None,None).run().unwrap().to_i64().unwrap(),55296);
        });
        let mut environment=crate::script::ScriptEnv::new(None,None);
        environment.raw_search_scripts=vec![Source::parse(r#"{"source":"params.s","params":{"s":"\udfff"}}"#).unwrap(),Source::parse(r#"{"s":"\ud800"}"#).unwrap()];
        crate::script::with_env(environment,|| {
            assert!(phase(json!({"combine_script":0,"params":1}).as_object().unwrap(),"combine_script",ScriptContext::MetricCombine).is_err());
        });
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    #[test]
    fn metric_big_numeric_growth_is_reserved_before_arithmetic() {
        let breakers=crate::memory::Breakers::leak_for_tests(1 << 30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("256kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),
            _=>None,
        });
        for script in [
            "return new BigInteger('2').pow(10000000).signum();",
            "return new BigInteger('1').shiftLeft(10000000).signum();",
            "return new BigDecimal('1').setScale(1000000).signum();",
            "return new BigDecimal('1E+10000000').byteValue();",
            "return new BigDecimal('1E-10000000').shortValue();",
        ] {
            let request=crate::memory::RequestScope::new("metric numeric growth",breakers);
            let out=crate::memory::scoped(Some(request),||reduce("m",json!({"reduce_script":script}).as_object().unwrap(),vec![]));
            assert_eq!(out.unwrap_err().status,429,"{script}");
            assert_eq!(breakers.request.reserved(),0);
        }
        let request=crate::memory::RequestScope::new("metric numeric temporaries",breakers);
        let out=crate::memory::scoped(Some(request),||reduce("m",json!({"reduce_script":"for(int i=0;i<1000;i++){def v=new BigInteger('1234567890').multiply(new BigInteger('1234567890'));} return 1;"}).as_object().unwrap(),vec![])).unwrap();
        assert_eq!(out["value"],1);
        assert_eq!(breakers.request.reserved(),0);
    }

    #[test]
    fn metric_compact_decimal_noop_does_not_charge_expanded_scale() {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("256kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let request=crate::memory::RequestScope::new("compact decimal",breakers);
        let out=crate::memory::scoped(Some(request),||reduce("m",json!({"reduce_script":"def v=new BigDecimal('1E+10000000'); return v.setScale(-10000000).movePointLeft(0).signum();"}).as_object().unwrap(),vec![])).unwrap();
        assert_eq!(out["value"],1);
        assert_eq!(breakers.request.reserved(),0);
    }

    #[test]
    fn metric_joiner_growth_checks_before_owned_buffer_growth() {
        let breakers=crate::memory::Breakers::leak_for_tests(1 << 30);
        for script in [
            "def j=new StringJoiner(','); for(int i=0;i<10000;i++){j.add('abcdefghij');} return 1;",
            "def j=new StringJoiner(','); j.add('abcdefghij'); for(int i=0;i<20;i++){j.merge(j);} return 1;",
        ] {
            breakers.apply_settings(&|key|match key {
                "indices.breaker.request.limit"=>Some("256kb".into()),
                "indices.breaker.total.limit"=>Some("-1".into()),
                _=>None,
            });
            let request=crate::memory::RequestScope::new("metric joiner growth",breakers);
            let out=crate::memory::scoped(Some(request),||reduce("m",json!({"reduce_script":script}).as_object().unwrap(),vec![]));
            assert_eq!(out.unwrap_err().status,429,"{script}");
            assert_eq!(breakers.request.reserved(),0);
        }
    }

    #[test]
    fn discarded_metric_strings_do_not_accumulate_reservations() {
        let breakers=crate::memory::Breakers::leak_for_tests(1 << 30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("256kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),
            _=>None,
        });
        for script in [
            "for(int i=0;i<1000;i++){def s='abcdefghij'.substring(1); } return 1;",
            "for(int i=0;i<1000;i++){def s='abcdefghij'+'abcdefghij'; } return 1;",
            "def b=new StringBuilder('abcdefghij'); for(int i=0;i<1000;i++){def s=b.toString();} return 1;",
        ] {
            let request=crate::memory::RequestScope::new("metric temporary scalars",breakers);
            let out=crate::memory::scoped(Some(request),||reduce("m",json!({"reduce_script":script}).as_object().unwrap(),vec![])).unwrap();
            assert_eq!(out["value"],1,"{script}");
            assert_eq!(breakers.request.reserved(),0);
        }
    }

    #[test]
    fn metric_shared_dag_output_trips_during_size_walk() {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("1mb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),
            _=>None,
        });
        for script in ["def a=[1]; for(int i=0;i<50;i++){a=[a,a];} return a;","def a=[1]; for(int i=0;i<200;i++){a=[a];} return a;"] {
            let before=OUTPUT_COPIES.with(std::cell::Cell::get);
            let request=crate::memory::RequestScope::new("metric output DAG",breakers);
            let out=crate::memory::scoped(Some(request),||reduce("m",json!({"reduce_script":script}).as_object().unwrap(),vec![]));
            assert_eq!(out.unwrap_err().status,429);
            assert_eq!(OUTPUT_COPIES.with(std::cell::Cell::get),before);
            assert_eq!(breakers.request.reserved(),0);
        }
    }

    #[test]
    fn metric_json_result_reservation_outlives_executor() {
        let breakers=crate::memory::Breakers::leak_for_tests(1 << 30);
        let request=crate::memory::RequestScope::new("metric output lifetime",breakers);
        let out=crate::memory::scoped(Some(request.clone()),||reduce("m",json!({"reduce_script":"return [1,2,3];"}).as_object().unwrap(),vec![])).unwrap();
        assert_eq!(out["value"],json!([1,2,3]));
        assert!(breakers.request.reserved()>0,"response stays charged after metric executor teardown");
        assert!(request.held()>0);
        drop(out);
        drop(request);
        assert_eq!(breakers.request.reserved(),0);
    }

    #[test]
    fn temporary_metric_cycles_are_valid_and_release_on_success_and_error() {
        let breakers = crate::memory::Breakers::leak_for_tests(1 << 30);
        for (script, succeeds) in [
            ("def a=[]; a.add(a); return 1;", true),
            ("def a=[]; def b=[:]; a.add(b); b.x=a; return 1;", true),
            ("def a=[]; a.add(a); throw new IllegalArgumentException('stop');", false),
            ("def a=[]; a.add(a); return a;", false),
        ] {
            let scope = crate::memory::RequestScope::new("metric cycles", breakers);
            let out = crate::memory::scoped(Some(scope), || reduce("m", json!({"reduce_script":script}).as_object().unwrap(), vec![]));
            assert_eq!(out.is_ok(), succeeds, "{script}: {out:?}");
            if succeeds { assert_eq!(out.unwrap()["value"], 1); }
            assert_eq!(breakers.request.reserved(), 0, "{script}");
        }
    }

    #[test]
    fn metric_builder_growth_is_reserved_before_mutation() {
        let breakers = crate::memory::Breakers::leak_for_tests(1 << 30);
        for script in [
            "def b=new StringBuilder(); b.setLength(100000); return b.length();",
            "def b=new StringBuilder(); b.setLength(100000); return ''.replace('',b.toString()).length();",
            "def b=new StringBuilder(); for(int i=0;i<1000;i++){b.append('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx');} return b.length();",
            "def b=new StringBuilder(); for(int i=0;i<1000;i++){b.insert(0,'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx');} return b.length();",
        ] {
            for limit in ["64kb", "64mb"] {
                breakers.apply_settings(&|key| match key {
                    "indices.breaker.request.limit" => Some(limit.into()),
                    "indices.breaker.total.limit" => Some("-1".into()),
                    _ => None,
                });
                let scope=crate::memory::RequestScope::new("metric builder",breakers);
                let out=crate::memory::scoped(Some(scope),||reduce("m",json!({"reduce_script":script}).as_object().unwrap(),vec![]));
                if limit=="64kb" { assert_eq!(out.unwrap_err().status,429,"{script}"); }
                else { assert_eq!(out.unwrap()["value"],100000,"{script}"); }
                assert_eq!(breakers.request.reserved(),0);
            }
        }
    }

    #[test]
    fn metric_scalar_concatenation_checks_before_growth() {
        let breakers=crate::memory::Breakers::leak_for_tests(1 << 30);
        for limit in ["64kb","64mb"] {
            breakers.apply_settings(&|key|match key {
                "indices.breaker.request.limit"=>Some(limit.into()),
                "indices.breaker.total.limit"=>Some("-1".into()),
                _=>None,
            });
            let scope=crate::memory::RequestScope::new("metric concatenation",breakers);
            let out=crate::memory::scoped(Some(scope),||reduce("m",json!({"reduce_script":"def s='x'; for(int i=0;i<17;i++){s=s+s;} return s.length();"}).as_object().unwrap(),vec![]));
            if limit=="64kb" { assert_eq!(out.unwrap_err().status,429); }
            else { assert_eq!(out.unwrap()["value"],131072); }
            assert_eq!(breakers.request.reserved(),0);
        }
    }

    #[test]
    fn metric_library_object_output_is_reserved_before_rendering() {
        let breakers=crate::memory::Breakers::leak_for_tests(1 << 30);
        breakers.apply_settings(&|key| match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),
            _=>None,
        });
        let value=Val::Obj(std::rc::Rc::new(crate::script::jmisc::Obj::Version("v".repeat(10000))));
        let scope=crate::memory::RequestScope::new("metric library output",breakers);
        let out=crate::memory::scoped(Some(scope),||reserve_outputs(&[value]));
        assert_eq!(out.err().unwrap().status,429);
        assert_eq!(breakers.request.reserved(),0);
    }

    #[test]
    fn metric_output_is_reserved_before_serialization() {
        let breakers = crate::memory::Breakers::leak_for_tests(1 << 30);
        let config=json!({"reduce_script":"def a=[]; for(int i=0;i<1000;i++){a.add(i);} return a;"});
        for limit in ["128kb", "16mb"] {
            breakers.apply_settings(&|key| match key {
                "indices.breaker.request.limit" => Some(limit.into()),
                "indices.breaker.total.limit" => Some("-1".into()),
                _ => None,
            });
            let before=OUTPUT_COPIES.with(std::cell::Cell::get);
            let scope=crate::memory::RequestScope::new("metric output",breakers);
            let out=crate::memory::scoped(Some(scope),||reduce("m",config.as_object().unwrap(),vec![]));
            if limit=="128kb" {
                assert_eq!(out.unwrap_err().status,429);
                assert_eq!(OUTPUT_COPIES.with(std::cell::Cell::get),before);
            } else { assert_eq!(out.unwrap()["value"].as_array().unwrap().len(),1000); }
            assert_eq!(breakers.request.reserved(),0);
        }
    }

    #[test]
    fn metric_inputs_are_reserved_before_runtime_decoding() {
        let breakers = crate::memory::Breakers::leak_for_tests(1 << 30);
        let states = vec![json!((0..10000).collect::<Vec<_>>())];
        let params = json!({"reduce_script":{"source":"params.values.size()", "params":{"values":states[0]}}});
        let shared = json!({"reduce_script":"params.values.size()", "params":{"values":states[0]}});
        for limit in ["64kb", "64mb"] {
            breakers.apply_settings(&|key| match key {
                "indices.breaker.request.limit" => Some(limit.into()),
                "indices.breaker.total.limit" => Some("-1".into()),
                _ => None,
            });
            for parameter_case in [0, 1, 2] {
                let before_copies = PHASE_COPIES.with(std::cell::Cell::get);
                let scope = crate::memory::RequestScope::new("metric input decode", breakers);
                let out = crate::memory::scoped(Some(scope), || if parameter_case != 0 {
                    reduce("m", if parameter_case == 1 { params.as_object().unwrap() } else { shared.as_object().unwrap() }, vec![])
                } else {
                    reduce_states("m", json!({"reduce_script":"states[0].size()"}).as_object().unwrap(), &states)
                });
                if limit == "64kb" {
                    assert_eq!(out.unwrap_err().status, 429);
                    assert_eq!(PHASE_COPIES.with(std::cell::Cell::get), before_copies, "reject before configuration copies");
                }
                else { assert_eq!(out.unwrap()["value"], 10000); }
                assert_eq!(breakers.request.reserved(), 0);
            }
        }
    }

    #[test]
    fn retained_metric_collection_copies_are_charged() {
        let breakers = crate::memory::Breakers::leak_for_tests(1 << 30);
        for script in [
            "def a=[]; for(int i=0;i<32;i++){a.add(i);} def copies=[]; for(int i=0;i<100;i++){copies.add(a.subList(0,32));} return copies.size();",
            "def a=[:]; for(int i=0;i<32;i++){a[i]=i;} def copies=[]; for(int i=0;i<100;i++){copies.add(a.values());} return copies.size();",
            "def a=[]; for(int i=0;i<32;i++){a.add(i);} def copies=[]; for(int i=0;i<100;i++){copies.add(new ArrayList(a));} return copies.size();",
            "def copies=[]; for(int i=0;i<100;i++){copies.add(IntStream.range(0,32).map(x -> x + 1));} return copies.size();",
            "def copies=[]; for(int i=0;i<100;i++){copies.add(IntStream.range(0,32).collect(Collectors.toList()));} return copies.size();",
            "def copies=[]; for(int i=0;i<100;i++){copies.add(IntStream.range(0,32).collect(Collectors.groupingBy(x -> x % 2)));} return copies.size();",
            "def a=[]; for(int i=0;i<32;i++){a.add(i);} def copies=[]; for(int i=0;i<100;i++){copies.add(a.iterator());} return copies.size();",
            "def a=new TreeMap(); for(int i=0;i<32;i++){a[i]=i;} def copies=[]; for(int i=0;i<100;i++){copies.add(a.headMap(16));} return copies.size();",
        ] {
            for limit in ["64kb", "16mb"] {
                breakers.apply_settings(&|key| match key {
                    "indices.breaker.request.limit" => Some(limit.into()),
                    "indices.breaker.total.limit" => Some("-1".into()),
                    _ => None,
                });
                let scope = crate::memory::RequestScope::new("metric copies", breakers);
                let out = crate::memory::scoped(Some(scope), || reduce("m", json!({"reduce_script":script}).as_object().unwrap(), vec![]));
                if limit == "64kb" { assert_eq!(out.unwrap_err().status, 429, "{script}"); }
                else { assert_eq!(out.unwrap()["value"], 100, "{script}"); }
                assert_eq!(breakers.request.reserved(), 0);
            }
        }
    }

    #[test]
    fn metric_collection_growth_breaks_before_mutation_and_cannot_be_caught() {
        let breakers = crate::memory::Breakers::leak_for_tests(1 << 30);
        for script in [
            "def a = []; for (int i = 0; i < 10000; i++) { a.add(i); } return a.size();",
            "def a = [:]; for (int i = 0; i < 10000; i++) { a[i] = i; } return a.size();",
            "try { def a = new int[10000]; return a.length; } catch (Exception e) { return 0; }",
            "return Collections.nCopies(10000, 1).size();",
        ] {
            let run = |limit: &str| {
                breakers.apply_settings(&|key| match key {
                    "indices.breaker.request.limit" => Some(limit.into()),
                    "indices.breaker.total.limit" => Some("-1".into()),
                    _ => None,
                });
                let scope = crate::memory::RequestScope::new("metric", breakers);
                let out = crate::memory::scoped(Some(scope), || reduce("m", json!({"reduce_script":script}).as_object().unwrap(), vec![]));
                assert_eq!(breakers.request.reserved(), 0, "phase releases on success and error");
                out
            };
            let error = run("64kb").expect_err(script);
            assert_eq!(error.status, 429, "{error:?}");
            assert!(crate::memory::is_breaking(&error), "{error:?}");
            assert_eq!(run("16mb").unwrap()["value"], 10000);
        }
    }
}
