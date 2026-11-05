//! ES error rendering with admission at each cloned string, map and vector.
//! Mirrors EsError's root-cause and trace rules; never first builds an unowned
//! error tree and then charges its serialized length.
use super::*;
use elasticsearch_engine::{error::{BREAKING_FIELDS,JavaClass,NULL_REASON},memory::Reservation};
use serde_json::Map;
use super::response_tree::{copy,formatted,insert,push};
fn string(memory:&mut Reservation,value:&str)->Result<Value,EsError>{Ok(Value::String(formatted(memory,value)?))}
fn body(error:&EsError,cause:bool,memory:&mut Reservation)->Result<Map<String,Value>,EsError>{
    let mut out=Map::new();
    let kind=string(memory,&error.kind)?;insert(memory,&mut out,"type",kind)?;
    let reason=if error.reason==NULL_REASON{Value::Null}else{string(memory,&error.reason)?};insert(memory,&mut out,"reason",reason)?;
    let breaking=error.kind=="circuit_breaking_exception";
    for(key,value)in &error.extra {if !key.starts_with('@')&&!(breaking&&BREAKING_FIELDS.contains(&key.as_str())){let value=copy(memory,value)?;insert(memory,&mut out,key,value)?;}}
    // ES renders all three on every circuit_breaking_exception, after the
    // other metadata (see EsError::body).
    if breaking {for field in BREAKING_FIELDS{let value=copy(memory,&error.breaking_value(field))?;insert(memory,&mut out,field,value)?;}}
    if cause {if let Some(cause)=&error.caused_by{let value=Value::Object(body(cause,true,memory)?);insert(memory,&mut out,"caused_by",value)?;}}
    Ok(out)
}
struct Trace<'a>(&'a Map<String,Value>);
impl std::fmt::Display for Trace<'_>{
    fn fmt(&self,out:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
        let mut current=Some(self.0);let mut first=true;
        while let Some(error)=current {
            if !first {out.write_str("\nCaused by: ")?;} first=false;
            let class=JavaClass(error.get("type").and_then(Value::as_str).unwrap_or("exception"));
            let reason=error.get("reason").and_then(Value::as_str).unwrap_or("null");
            write!(out,"{class}: {reason}\n\tat {class}.<init>(Unknown Source)")?;
            current=error.get("caused_by").and_then(Value::as_object);
        }
        Ok(())
    }
}
fn traces(value:&mut Value,memory:&mut Reservation)->Result<(),EsError>{
    match value {
        Value::Object(map)=>{
            if map.get("type").is_some_and(Value::is_string)&&map.contains_key("reason")&&!map.contains_key("stack_trace"){
                let trace=formatted(memory,Trace(map))?;insert(memory,map,"stack_trace",Value::String(trace))?;
            }
            for(key,value)in map.iter_mut(){if key!="stack_trace"{traces(value,memory)?;}}
        },
        Value::Array(values)=>for value in values{traces(value,memory)?;},
        _=>{},
    }
    Ok(())
}
fn shard_roots(error:&EsError,trace:bool,memory:&mut Reservation)->Result<Option<Vec<Value>>,EsError>{
    if error.kind!="search_phase_execution_exception"||error.caused_by.is_some(){return Ok(None)}
    let Some(shards)=error.extra.iter().find(|(key,_)|key=="failed_shards").and_then(|(_,v)|v.as_array())else{return Ok(None)};
    let xc=|kind:&str|matches!(kind,"x_content_parse_exception"|"x_content_e_o_f_exception"|"named_object_not_found_exception");
    let mut roots=Vec::new();
    for shard in shards {
        let Some(mut current)=shard.get("reason").and_then(Value::as_object)else{continue};
        while let Some(cause)=current.get("caused_by").and_then(Value::as_object){
            let kind=current.get("type").and_then(Value::as_str).unwrap_or("");let child=cause.get("type").and_then(Value::as_str).unwrap_or("");
            if !(if xc(kind){xc(child)||EsError::is_es_exception(child)}else{EsError::is_es_exception(kind)&&EsError::is_es_exception(child)}){break}
            current=cause;
        }
        let mut root=Map::new();
        for(key,value)in current{if key!="caused_by"{let value=copy(memory,value)?;insert(memory,&mut root,key,value)?;}}
        if trace{let value=Value::String(formatted(memory,Trace(current))?);insert(memory,&mut root,"stack_trace",value)?;}
        push(memory,&mut roots,Value::Object(root))?;
    }
    Ok((!roots.is_empty()).then_some(roots))
}
pub(super) fn render(error:&EsError,trace:bool,memory:&mut Reservation)->Result<Value,EsError>{
    let roots=if error.kind=="search_phase_execution_exception"&&error.reason.is_empty(){Vec::new()}
        else if let Some(roots)=shard_roots(error,trace,memory)?{roots}
        else{
            let root=error.root();let mut map=body(root,false,memory)?;
            if trace{let full=body(root,true,memory)?;let value=Value::String(formatted(memory,Trace(&full))?);insert(memory,&mut map,"stack_trace",value)?;}
            let mut roots=Vec::new();push(memory,&mut roots,Value::Object(map))?;roots
        };
    let mut output=Map::new();insert(memory,&mut output,"root_cause",Value::Array(roots))?;
    let mut own=Value::Object(body(error,true,memory)?);if trace{traces(&mut own,memory)?;}
    if let Value::Object(own)=own{for(key,value)in own{insert(memory,&mut output,&key,value)?;}}
    let mut result=Map::new();insert(memory,&mut result,"error",Value::Object(output))?;insert(memory,&mut result,"status",error.status.into())?;
    Ok(Value::Object(result))
}
struct SimpleClass<'a>(&'a str);
impl std::fmt::Display for SimpleClass<'_>{fn fmt(&self,out:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
    if matches!(self.0,"exception"|"timeout_exception"|"status_exception"|"parse_exception"|"security_exception"|"generation_exception"){out.write_str("Elasticsearch")?;}
    for part in self.0.split('_'){let mut chars=part.chars();if let Some(first)=chars.next(){for upper in first.to_uppercase(){write!(out,"{upper}")?;}out.write_str(chars.as_str())?;}}
    Ok(())
}}
pub(super) fn non_detailed(error:&EsError,version:RestApiVersion,memory:&mut Reservation)->Result<(u16,RespBody,Vec<(&'static str,String)>),EsError>{
    let mut output=Map::new();let mut headers=Vec::new();
    let value=if version.is_v8(){
        let value=Value::String(formatted(memory,format_args!("{}[{}]",SimpleClass(&error.kind),error.reason))?);
        let warning=formatted(memory,format_args!("299 Elasticsearch-{}-{} \"The JSON format of non-detailed errors has changed in Elasticsearch 9.0 to match the JSON structure used for detailed errors.\"",elasticsearch_engine::node::ES_VERSION,cluster::BUILD_HASH))?;
        push(memory,&mut headers,("Warning",warning))?;value
    }else{
        let mut map=Map::new();let kind=string(memory,&error.kind)?;insert(memory,&mut map,"type",kind)?;let reason=string(memory,&error.reason)?;insert(memory,&mut map,"reason",reason)?;Value::Object(map)
    };
    insert(memory,&mut output,"error",value)?;insert(memory,&mut output,"status",error.status.into())?;
    Ok((error.status,RespBody::Json(Value::Object(output)),headers))
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn rendering_preserves_error_shapes_and_trace_rules(){
        let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"error rendering fixture").unwrap();
        let errors=[EsError::index_not_found("index"),EsError::parsing("bad").caused_by(EsError::illegal_argument("cause")),
            EsError::new(500,"search_phase_execution_exception","").with("failed_shards",serde_json::json!([])),
            EsError::new(400,"search_phase_execution_exception","all shards failed").with("failed_shards",serde_json::json!([{"reason":{"type":"query_shard_exception","reason":"bad","caused_by":{"type":"mapper_parsing_exception","reason":"inner"}}}])),
            EsError::new(401,"security_exception",NULL_REASON).with("@position",1).with("suppressed",serde_json::json!([{"type":"io_exception","reason":"io"}])),
            EsError::new(429,"circuit_breaking_exception","decoded source parameter allocation refused").with("index","i"),
            elasticsearch_engine::memory::breaking("[request] Data too large".into(),10,5,elasticsearch_engine::memory::Durability::Transient)];
        for error in errors{for trace in [false,true]{assert_eq!(render(&error,trace,&mut memory).unwrap(),error.to_json_with_trace(trace));}
            for version in [RestApiVersion::V8,RestApiVersion::V9]{let actual=non_detailed(&error,version,&mut memory).unwrap();let expected=super::super::non_detailed_error(&error,version);assert_eq!(actual.0,expected.0);assert_eq!(actual.2,expected.2);if let(RespBody::Json(actual),RespBody::Json(expected))=(actual.1,expected.1){assert_eq!(actual,expected)}else{panic!("JSON error body expected")}}
        }
    }
    #[test]
    fn error_rendering_refusal_preserves_input(){
        let breakers=elasticsearch_engine::memory::Breakers::leak_for_tests(64);breakers.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let mut memory=breakers.reserve_request(0,"error refusal fixture").unwrap();let error=EsError::index_not_found("unchanged");let before=error.clone();
        assert_eq!(render(&error,true,&mut memory).unwrap_err().status,429);assert_eq!(error,before);drop(memory);assert_eq!(breakers.request.reserved(),0);
    }
}
