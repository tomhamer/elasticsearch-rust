//! Temporary JSON trees used by HTTP response transforms. Admission follows
//! concrete strings and container insertions rather than encoded output size.
use elasticsearch_engine::{error::EsError, memory::Reservation};
use serde::de::{DeserializeSeed, Error, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
pub(super) fn parse(text:&str, memory:&mut Reservation)->Result<Value,EsError> {
    // serde_json unescape scratch grows geometrically and may overlap its
    // replacement; decoded UTF-8 never exceeds the escaped input length.
    let _scratch=elasticsearch_engine::memory::breakers().reserve_request((text.len() as u64).saturating_mul(4).saturating_add(256),"HTTP JSON unescape scratch")?;
    let mut failure=None;
    let mut parser=serde_json::Deserializer::from_str(text);
    let result=Seed{memory,failure:&mut failure}.deserialize(&mut parser);
    if let Some(error)=failure{return Err(error)}
    let value=result.map_err(|e|EsError::new(400,"parse_exception",e.to_string()))?;
    parser.end().map_err(|e|EsError::new(400,"parse_exception",e.to_string()))?;
    Ok(value)
}
struct Seed<'a> { memory:&'a mut Reservation, failure:&'a mut Option<EsError> }
impl Seed<'_> {
    fn charge<E:Error>(&mut self,bytes:usize)->Result<(),E> {
        self.memory.grow(bytes as u64,"owned JSON allocation").map_err(|e| {
            let message=e.to_string(); *self.failure=Some(e); E::custom(message)
        })
    }
    fn allocation_error<E:Error>(&mut self,error:impl std::fmt::Display)->E {
        let message=error.to_string();
        *self.failure=Some(EsError::new(429,"circuit_breaking_exception",message.clone()));
        E::custom(message)
    }
    fn child(&mut self)->Seed<'_> { Seed{memory:self.memory,failure:self.failure} }
}
impl<'de> DeserializeSeed<'de> for Seed<'_> {
    type Value=Value;
    fn deserialize<D:serde::Deserializer<'de>>(self,d:D)->Result<Value,D::Error> { d.deserialize_any(self) }
}
impl<'de> Visitor<'de> for Seed<'_> {
    type Value=Value;
    fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result { f.write_str("JSON value") }
    fn visit_unit<E:Error>(self)->Result<Value,E>{Ok(Value::Null)}
    fn visit_bool<E:Error>(self,v:bool)->Result<Value,E>{Ok(v.into())}
    fn visit_i64<E:Error>(self,v:i64)->Result<Value,E>{Ok(v.into())}
    fn visit_u64<E:Error>(self,v:u64)->Result<Value,E>{Ok(v.into())}
    fn visit_f64<E:Error>(self,v:f64)->Result<Value,E>{Ok(serde_json::Number::from_f64(v).map_or(Value::Null,Value::Number))}
    fn visit_str<E:Error>(mut self,v:&str)->Result<Value,E>{
        self.charge(v.len())?;
        let mut out=String::new();out.try_reserve_exact(v.len()).map_err(|e|self.allocation_error::<E>(e))?;out.push_str(v);
        Ok(Value::String(out))
    }
    fn visit_borrowed_str<E:Error>(self,v:&'de str)->Result<Value,E>{self.visit_str(v)}
    fn visit_seq<A:SeqAccess<'de>>(mut self,mut seq:A)->Result<Value,A::Error>{
        let mut out=Vec::new();
        while let Some(value)=seq.next_element_seed(self.child())? {
            if out.len()==out.capacity() {
                let capacity=out.capacity().saturating_mul(2).max(4);
                self.charge(capacity.saturating_mul(std::mem::size_of::<Value>()))?;
                out.try_reserve_exact(capacity-out.len()).map_err(|e|self.allocation_error::<A::Error>(e))?;
            }
            out.push(value);
        }
        Ok(Value::Array(out))
    }
    fn visit_map<A:MapAccess<'de>>(mut self,mut map:A)->Result<Value,A::Error>{
        let mut out=serde_json::Map::new();
        while let Some(key)=map.next_key_seed(Key(self.child()))? {
            // Cover both serde_json Map backends: a BTreeMap node split,
            // or preserve_order IndexMap's entry/hash-table geometric growth
            // including old/new overlap. Sixteen entry+pointer units per
            // insertion exceed the live-plus-replacement capacities of both
            // layouts; key bytes are separately admitted. Duplicate inserts
            // conservatively keep their charge through the tree lifetime.
            self.charge(16usize.saturating_mul(std::mem::size_of::<(String,Value)>()+std::mem::size_of::<usize>()))?;
            let value=map.next_value_seed(self.child())?;
            out.insert(key,value);
        }
        Ok(Value::Object(out))
    }
}
struct Key<'a>(Seed<'a>);
impl<'de> DeserializeSeed<'de> for Key<'_> {
    type Value=String;
    fn deserialize<D:serde::Deserializer<'de>>(self,d:D)->Result<String,D::Error>{
        struct V<'a>(Seed<'a>);
        impl<'de> Visitor<'de> for V<'_>{
            type Value=String;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("JSON object key")}
            fn visit_str<E:Error>(mut self,v:&str)->Result<String,E>{self.0.charge(v.len())?;let mut s=String::new();s.try_reserve_exact(v.len()).map_err(|e|self.0.allocation_error::<E>(e))?;s.push_str(v);Ok(s)}
            fn visit_borrowed_str<E:Error>(self,v:&'de str)->Result<String,E>{self.visit_str(v)}
        }
        d.deserialize_str(V(self.0))
    }
}

fn denied()->EsError {EsError::new(429,"circuit_breaking_exception","HTTP response tree allocation refused")}
fn charge(memory:&mut Reservation,bytes:usize)->Result<(),EsError>{memory.grow(bytes as u64,"HTTP response tree transform")}
fn text(memory:&mut Reservation,value:&str)->Result<String,EsError>{
    charge(memory,value.len())?;let mut out=String::new();out.try_reserve_exact(value.len()).map_err(|_|denied())?;out.push_str(value);Ok(out)
}
pub(super) fn push<T>(memory:&mut Reservation,out:&mut Vec<T>,value:T)->Result<(),EsError>{
    if out.len()==out.capacity(){let capacity=out.capacity().checked_mul(2).ok_or_else(denied)?.max(4);charge(memory,capacity.checked_mul(std::mem::size_of::<T>()).ok_or_else(denied)?)?;out.try_reserve_exact(capacity-out.len()).map_err(|_|denied())?;}
    out.push(value);Ok(())
}
fn map_slot(memory:&mut Reservation)->Result<(),EsError>{
    charge(memory,16*(std::mem::size_of::<(String,Value)>()+std::mem::size_of::<usize>()))
}
pub(super) fn copy(memory:&mut Reservation,value:&Value)->Result<Value,EsError>{
    Ok(match value {
        Value::String(value)=>Value::String(text(memory,value)?),
        Value::Array(values)=>{let mut out=Vec::new();for value in values{let value=copy(memory,value)?;push(memory,&mut out,value)?}Value::Array(out)},
        Value::Object(values)=>{let mut out=serde_json::Map::new();for(k,v)in values{map_slot(memory)?;let k=text(memory,k)?;let v=copy(memory,v)?;out.insert(k,v);}Value::Object(out)},
        value=>value.clone(),
    })
}
// Paths retain borrowed slices, not recursively cloned owned pattern strings.
fn include(memory:&mut Reservation,value:&Value,patterns:&[&[&str]])->Result<Option<Value>,EsError>{
    if patterns.iter().any(|p|p.is_empty()){return copy(memory,value).map(Some)}
    Ok(match value {
        Value::Object(values)=>{let mut out=serde_json::Map::new();for(k,v)in values{
            let mut next=Vec::new();for p in patterns {
                if p[0]=="**"{push(memory,&mut next,*p)?;if p.len()>1&&wildcard(p[1],k){push(memory,&mut next,&p[2..])?}}
                else if wildcard(p[0],k){push(memory,&mut next,&p[1..])?}
            }
            if next.is_empty(){continue}if let Some(v)=include(memory,v,&next)?{map_slot(memory)?;out.insert(text(memory,k)?,v);}
        }if out.is_empty(){None}else{Some(Value::Object(out))}},
        Value::Array(values)=>{let mut out=Vec::new();for v in values{if let Some(v)=include(memory,v,patterns)?{push(memory,&mut out,v)?}}if out.is_empty(){None}else{Some(Value::Array(out))}},
        _=>None,
    })
}
fn exclude(value:&mut Value,pattern:&[&str]){
    if pattern.is_empty(){return}
    match value {
        Value::Object(values)=>values.retain(|key,child|{
            if pattern[0]=="**"{
                exclude(child,pattern);
                if pattern.len()>1&&wildcard(pattern[1],key){if pattern.len()==2{return false}exclude(child,&pattern[2..]);}
            }else if wildcard(pattern[0],key){if pattern.len()==1{return false}exclude(child,&pattern[1..]);}
            true
        }),
        Value::Array(values)=>for value in values{exclude(value,pattern)},
        _=>{}
    }
}
pub(super) fn filter(value:&Value,spec:&str,memory:&mut Reservation)->Result<Value,EsError>{
    let mut includes=Vec::new();let mut excludes=Vec::new();
    for p in spec.split(',').map(str::trim).filter(|p|!p.is_empty()){
        let (excluded,p)=p.strip_prefix('-').map_or((false,p),|p|(true,p));let mut segments=Vec::new();
        for segment in p.split('.'){push(memory,&mut segments,segment)?}
        push(memory,if excluded{&mut excludes}else{&mut includes},segments)?;
    }
    let mut refs=Vec::new();for p in &includes{push(memory,&mut refs,p.as_slice())?}
    let mut result=if refs.is_empty(){copy(memory,value)?}else{include(memory,value,&refs)?.unwrap_or_else(||Value::Object(serde_json::Map::new()))};
    for p in &excludes{exclude(&mut result,p)}Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admitted_filter_preserves_original_selectors_and_duplicate_keys(){
        let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"response transform fixture").unwrap();
        let input=r#"{"hits":{"hits":[{"_id":"1","_source":{"n":1}},{"_id":"2"}]},"took":1,"empty":{},"duplicate":1,"duplicate":2}"#;
        let value=parse(input,&mut memory).unwrap();
        assert_eq!(value,serde_json::from_str::<Value>(input).unwrap());
        for spec in ["", "hits.hits._id", "took,hits.hits._source.n", "**._id", "-hits", "hits.*._id,-**._source", "nope", "**", "-**._id", "**.hits.**._id"] {
            assert_eq!(filter(&value,spec,&mut memory).unwrap(),crate::util::filter_path(&value,spec),"{spec}");
        }
    }
}

pub(super) fn formatted(memory:&mut Reservation,value:impl std::fmt::Display)->Result<String,EsError>{
    struct Count(usize);impl std::fmt::Write for Count{fn write_str(&mut self,s:&str)->std::fmt::Result{self.0=self.0.checked_add(s.len()).ok_or(std::fmt::Error)?;Ok(())}}
    let mut count=Count(0);std::fmt::write(&mut count,format_args!("{value}")).map_err(|_|denied())?;
    charge(memory,count.0)?;let mut out=String::new();out.try_reserve_exact(count.0).map_err(|_|denied())?;
    struct Fixed<'a>(&'a mut String,usize);impl std::fmt::Write for Fixed<'_>{fn write_str(&mut self,s:&str)->std::fmt::Result{if s.len()>self.1.saturating_sub(self.0.len()){return Err(std::fmt::Error)}self.0.push_str(s);Ok(())}}
    std::fmt::write(&mut Fixed(&mut out,count.0),format_args!("{value}")).map_err(|_|denied())?;Ok(out)
}
struct Human(i64,u8);
impl std::fmt::Display for Human{
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
        if self.1==3{return elasticsearch_engine::date::write_default_utc(self.0,f)}
        if self.0<0{return write!(f,"{}",self.0)}
        let n=self.0 as u64;
        if self.1==0{
            for(unit,size)in[("pb",1u64<<50),("tb",1<<40),("gb",1<<30),("mb",1<<20),("kb",1<<10)]{
                if n>=size{let tenths=(n as u128*10/size as u128) as u64;return if tenths%10==0{write!(f,"{}{unit}",tenths/10)}else{write!(f,"{}.{d}{unit}",tenths/10,d=tenths%10)}}
            }return write!(f,"{n}b")
        }
        let n=if self.1==1{n.wrapping_mul(1_000_000)}else{n};
        if n==0{return f.write_str("0s")}
        for(unit,size)in[("d",86_400_000_000_000),("h",3_600_000_000_000),("m",60_000_000_000),("s",1_000_000_000),("ms",1_000_000),("micros",1_000),("nanos",1)]{
            if n>=size {
                struct Stack{bytes:[u8;64],len:usize}impl std::fmt::Write for Stack{fn write_str(&mut self,s:&str)->std::fmt::Result{let end=self.len.checked_add(s.len()).filter(|end|*end<=self.bytes.len()).ok_or(std::fmt::Error)?;self.bytes[self.len..end].copy_from_slice(s.as_bytes());self.len=end;Ok(())}}
                let mut out=Stack{bytes:[0;64],len:0};std::fmt::write(&mut out,format_args!("{:.1}",n as f64/size as f64))?;
                let value=std::str::from_utf8(&out.bytes[..out.len]).map_err(|_|std::fmt::Error)?;
                return write!(f,"{}{unit}",value.strip_suffix(".0").unwrap_or(value))
            }
        }f.write_str("0s")
    }
}
pub(super) fn humanize(value:&mut Value,memory:&mut Reservation)->Result<(),EsError>{
    match value {
        Value::Array(values)=>for value in values{humanize(value,memory)?},
        Value::Object(values)=>{
            for value in values.values_mut(){humanize(value,memory)?}
            if !values.iter().any(|(k,v)|v.is_number()&&(k.ends_with("_in_bytes")||k.ends_with("_in_millis")||k.ends_with("_in_nanos"))){return Ok(())}
            // Borrow original keys while moving the values into a new map.
            // The old map stays intact (with Null values) until the new map is
            // complete, so duplicate human names cannot replace actual fields.
            let mut next=serde_json::Map::new();
            for(k,v)in values.iter(){
                if let Some(n)=v.as_i64(){
                    let human=if let Some(base)=k.strip_suffix("_in_bytes"){Some((base,0))}
                        else if let Some(base)=k.strip_suffix("_in_millis"){Some((base,if base.ends_with("start_time")||base.ends_with("_date")||base=="timestamp"{3}else{1}))}
                        else{k.strip_suffix("_in_nanos").map(|base|(base,2))};
                    if let Some((base,kind))=human{if !values.contains_key(base)&&!next.contains_key(base){map_slot(memory)?;let key=text(memory,base)?;let value=formatted(memory,Human(n,kind))?;next.insert(key,Value::String(value));}}
                }
                // Ordering follows the original inserted-human-before-raw
                // contract, but no copy of the raw value tree is made.
                map_slot(memory)?;next.insert(text(memory,k)?,Value::Null);
            }
            for(k,v)in std::mem::take(values){*next.get_mut(&k).expect("original field slot")=v;}
            *values=next;
        },_=>{}
    }Ok(())
}

fn wildcard(pattern:&str,value:&str)->bool {
    if !pattern.contains('*'){return pattern==value}
    let mut parts=pattern.split('*').peekable();let mut first=true;let mut at=0;
    while let Some(part)=parts.next(){
        let is_first=first;first=false;if part.is_empty(){continue}
        if is_first{if !value.starts_with(part){return false}at=part.len();}
        else if parts.peek().is_none(){return value.len()>=at+part.len()&&value[at..].ends_with(part)}
        else if let Some(found)=value[at..].find(part){at+=found+part.len()}else{return false}
    }true
}
#[cfg(test)]
mod formatting_tests {
    use super::*;
    #[test]
    fn human_fields_and_borrowed_wildcards_match_existing_contract(){
        let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"human transform fixture").unwrap();
        let mut value=serde_json::json!({"store":{"size_in_bytes":4608,"reserved_in_bytes":0,"size":"actual"},"indexing":{"index_time_in_millis":1500},"t":[{"running_time_in_nanos":2000}],"timestamp_in_millis":-1001,"start_time_in_millis":0,"other_in_bytes":-1});
        let mut expected=value.clone();crate::util::humanize(&mut expected);humanize(&mut value,&mut memory).unwrap();assert_eq!(value,expected);
        for pattern in ["*", "**", "a*", "*a", "a*b*c", "*λ*", "ab", "*a*", "a**b"]{for value in ["", "a", "ab", "abc", "aλb", "λ", "aaa"]{assert_eq!(wildcard(pattern,value),elasticsearch_engine::mapping::wildcard_match(pattern,value));}}
        for millis in [-1001,0,1,1234567890,i64::MAX]{assert_eq!(format!("{}",Human(millis,3)),elasticsearch_engine::date::DateFormatter::default().format(millis,elasticsearch_engine::date::Tz::UTC));}
    }
}

use elasticsearch_engine::source_json::Source;
use serde::Deserialize;
struct SourceSeed<'a>{memory:&'a mut Reservation,failure:&'a mut Option<EsError>,depth:usize}
impl<'de> DeserializeSeed<'de> for SourceSeed<'_>{
    type Value=Source;
    fn deserialize<D:serde::Deserializer<'de>>(self,d:D)->Result<Source,D::Error>{
        let raw=<&serde_json::value::RawValue>::deserialize(d)?;
        match source_raw(raw.get(),self.memory,self.depth){Ok(source)=>Ok(source),Err(error)=>{let message=error.to_string();*self.failure=Some(error);Err(D::Error::custom(message))}}
    }
}
struct SourceContainer<'a>{memory:&'a mut Reservation,failure:&'a mut Option<EsError>,depth:usize}
impl<'de> Visitor<'de> for SourceContainer<'_>{
    type Value=Source;
    fn expecting(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.write_str("lossless JSON container")}
    fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Source,A::Error>{
        let mut out=Vec::new();while let Some(value)=seq.next_element_seed(SourceSeed{memory:self.memory,failure:self.failure,depth:self.depth+1})?{
            if let Err(error)=push(self.memory,&mut out,value){let message=error.to_string();*self.failure=Some(error);return Err(A::Error::custom(message))}
        }Ok(Source::Array(out))
    }
    fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Source,A::Error>{
        let mut out=std::collections::BTreeMap::new();
        while let Some(key)=map.next_key_seed(Key(Seed{memory:self.memory,failure:self.failure}))?{
            if let Err(error)=source_slot(self.memory){let message=error.to_string();*self.failure=Some(error);return Err(A::Error::custom(message))}
            let value=map.next_value_seed(SourceSeed{memory:self.memory,failure:self.failure,depth:self.depth+1})?;out.insert(key,value);
        }Ok(Source::Object(out))
    }
}
fn source_slot(memory:&mut Reservation)->Result<(),EsError>{charge(memory,16*(std::mem::size_of::<(String,Source)>()+std::mem::size_of::<usize>()))}
fn source_raw(raw:&str,memory:&mut Reservation,depth:usize)->Result<Source,EsError>{
    if depth>128{return Err(EsError::new(400,"parse_exception","source nesting limit exceeded"))}
    match raw.as_bytes().first(){
        Some(b'{')|Some(b'[')=>{
            let _scratch=elasticsearch_engine::memory::breakers().reserve_request((raw.len() as u64).saturating_mul(4).saturating_add(256),"lossless JSON key scratch")?;
            let mut failure=None;let mut parser=serde_json::Deserializer::from_str(raw);
            let result=serde::Deserializer::deserialize_any(&mut parser,SourceContainer{memory,failure:&mut failure,depth});
            if let Some(error)=failure{return Err(error)}
            result.map_err(|e|EsError::new(400,"parse_exception",e.to_string()))
        },
        _=>{let value=text(memory,raw)?;let raw=serde_json::value::RawValue::from_string(value).map_err(|e|EsError::new(400,"parse_exception",e.to_string()))?;Ok(Source::Scalar(raw))}
    }
}
pub(super) fn parse_source(input:&str,memory:&mut Reservation)->Result<Source,EsError>{
    // Validate and borrow the entire token first. RawValue accepts Java's lone
    // UTF-16 surrogate escapes without replacing them or allocating a copy.
    let raw:&serde_json::value::RawValue=serde_json::from_str(input).map_err(|e|EsError::new(400,"parse_exception",e.to_string()))?;
    source_raw(raw.get(),memory,0)
}
fn source_copy(memory:&mut Reservation,value:&Source)->Result<Source,EsError>{
    Ok(match value{
        Source::Scalar(raw)=>Source::Scalar(serde_json::value::RawValue::from_string(text(memory,raw.get())?).map_err(|e|EsError::new(400,"parse_exception",e.to_string()))?),
        Source::Array(values)=>{let mut out=Vec::new();for v in values{let v=source_copy(memory,v)?;push(memory,&mut out,v)?}Source::Array(out)},
        Source::Object(values)=>{let mut out=std::collections::BTreeMap::new();for(k,v)in values{source_slot(memory)?;let k=text(memory,k)?;let v=source_copy(memory,v)?;out.insert(k,v);}Source::Object(out)}
    })
}
fn source_include(memory:&mut Reservation,value:&Source,patterns:&[&[&str]])->Result<Option<Source>,EsError>{
    if patterns.iter().any(|p|p.is_empty()){return source_copy(memory,value).map(Some)}
    Ok(match value{
        Source::Object(values)=>{let mut out=std::collections::BTreeMap::new();for(k,v)in values{
            let mut next=Vec::new();for p in patterns{if p[0]=="**"{push(memory,&mut next,*p)?;if p.len()>1&&wildcard(p[1],k){push(memory,&mut next,&p[2..])?}}else if wildcard(p[0],k){push(memory,&mut next,&p[1..])?}}
            if next.is_empty(){continue}if let Some(v)=source_include(memory,v,&next)?{source_slot(memory)?;out.insert(text(memory,k)?,v);}
        }if out.is_empty(){None}else{Some(Source::Object(out))}},
        Source::Array(values)=>{let mut out=Vec::new();for v in values{if let Some(v)=source_include(memory,v,patterns)?{push(memory,&mut out,v)?}}if out.is_empty(){None}else{Some(Source::Array(out))}},
        _=>None,
    })
}
fn source_exclude(value:&mut Source,pattern:&[&str]){
    if pattern.is_empty(){return}match value{
        Source::Object(values)=>values.retain(|key,child|{
            if pattern[0]=="**"{source_exclude(child,pattern);if pattern.len()>1&&wildcard(pattern[1],key){if pattern.len()==2{return false}source_exclude(child,&pattern[2..]);}}
            else if wildcard(pattern[0],key){if pattern.len()==1{return false}source_exclude(child,&pattern[1..]);}true
        }),Source::Array(values)=>for v in values{source_exclude(v,pattern)},_=>{}
    }
}
pub(super) fn filter_source(value:&Source,spec:&str,memory:&mut Reservation)->Result<Source,EsError>{
    let mut includes=Vec::new();let mut excludes=Vec::new();for p in spec.split(',').map(str::trim).filter(|p|!p.is_empty()){
        let(excluded,p)=p.strip_prefix('-').map_or((false,p),|p|(true,p));let mut segments=Vec::new();for part in p.split('.'){push(memory,&mut segments,part)?}push(memory,if excluded{&mut excludes}else{&mut includes},segments)?;
    }
    let mut refs=Vec::new();for p in &includes{push(memory,&mut refs,p.as_slice())?}
    let mut out=if refs.is_empty(){source_copy(memory,value)?}else{source_include(memory,value,&refs)?.unwrap_or_else(||Source::Object(std::collections::BTreeMap::new()))};
    for p in excludes{source_exclude(&mut out,&p)}Ok(out)
}

#[cfg(test)]
mod lossless_tests {
    use super::*;
    #[test]
    fn admitted_lossless_filter_preserves_surrogates_and_scalar_tokens(){
        let input=r#"{"hits":{"hits":[{"_id":"1","_source":{"raw":"\ud800","number":1e03}},{"_id":"2","_source":"\udfff"}]},"took":2}"#;
        let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"lossless fixture").unwrap();
        let old=Source::parse(input).unwrap();let source=parse_source(input,&mut memory).unwrap();assert_eq!(source.text(),old.text());
        for spec in ["", "hits.hits._source", "**._id", "-**._id", "nope", "**"]{
            assert_eq!(filter_source(&source,spec,&mut memory).unwrap().text(),old.select_structure(|v|crate::util::filter_path(v,spec)).text(),"{spec}");
        }
    }
}
#[cfg(test)]
mod refusal_tests {
    use super::*;
    #[test]
    fn tree_and_filter_refusal_preserve_input_and_release_admitted_storage(){
        let budget=elasticsearch_engine::memory::Breakers::leak_for_tests(64);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let mut memory=budget.reserve_request(0,"response denial fixture").unwrap();
        assert_eq!(parse(r#"{"field":[1,2,3]}"#,&mut memory).unwrap_err().status,429);
        let input=serde_json::json!({"field":[1,2,3]});
        assert_eq!(filter(&input,"field",&mut memory).unwrap_err().status,429);
        assert_eq!(input,serde_json::json!({"field":[1,2,3]}));
        drop(memory);assert_eq!(budget.request.reserved(),0);
    }
}

pub(super) fn insert(memory:&mut Reservation,map:&mut serde_json::Map<String,Value>,key:&str,value:Value)->Result<(),EsError>{
    map_slot(memory)?;let key=text(memory,key)?;map.insert(key,value);Ok(())
}
