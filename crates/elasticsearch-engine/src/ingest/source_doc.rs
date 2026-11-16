//! Lossless ingest document. Java string values remain raw JSON tokens across
//! field moves, list mutation and script execution; metadata keys stay Unicode.
use crate::source_json::Source;
use crate::error::{EsError,Result};
use std::collections::BTreeMap;
use serde_json::Value;

#[derive(Clone,Debug)]
pub struct SourceDocument {
    pub ctx:Source,
    pub ingest:Source,
    pub rerouted:bool,
}
fn object()->Source {Source::Object(BTreeMap::new())}
fn missing(path:&str)->EsError {EsError::illegal_argument(format!("field [{path}] not present as part of path [{path}]"))}
fn parts(path:&str)->Result<(bool,Vec<&str>)> {
    let (ingest,path)=if path=="_ingest"{(true,"")}else if let Some(p)=path.strip_prefix("_ingest."){(true,p)}else{(false,path.strip_prefix("_source.").unwrap_or(path))};
    if path.is_empty(){if ingest{return Ok((true,Vec::new()));}return Err(EsError::illegal_argument("path cannot be null nor empty"));}
    let segments:Vec<_>=path.split('.').collect();
    // Flexible paths have no list index syntax yet (`parseFlexibleFields`).
    let flexible=super::flexible_access();
    if segments.iter().any(|s|s.is_empty()||(flexible&&(s.contains('[')||s.contains(']')))){return Err(EsError::illegal_argument(format!("path [{path}] is not valid")));}
    Ok((ingest,segments))
}
fn invalid(path:&str)->EsError {EsError::illegal_argument(format!("path [{path}] is not valid"))}
fn type_of(v:&Source)->&'static str {match v {Source::Object(_)=>"java.util.HashMap",Source::Array(_)=>"java.util.ArrayList",Source::Scalar(raw)=>serde_json::from_str::<Value>(raw.get()).map(|v|super::doc::type_name(&v)).unwrap_or("java.lang.String")}}
fn list_index(segment:&str,len:usize)->Result<usize> {
    let i=segment.parse::<usize>().map_err(|_|EsError::illegal_argument(format!("[{segment}] is not an integer, cannot be used as an index")))?;
    if i>=len{return Err(EsError::illegal_argument(format!("[{i}] is out of bounds for array with length [{len}]")));}Ok(i)
}
impl SourceDocument {
    pub fn new(source:Source,index:&str,id:Option<&str>,routing:Option<&str>,timestamp:String)->Result<Self> {
        let Source::Object(mut ctx)=source else{return Err(EsError::illegal_argument("source must be an object"))};
        // ES `CtxMap`: the request's metadata and the source are separate
        // maps, and a source key naming a metadata key is rejected (it
        // never becomes the write's `_id` / `_routing`: production review
        // #192, an id-less write adopting a source `_id`).
        let bad:Vec<String>=["_id","_index","_routing","_version","_version_type"].iter().filter_map(|k|ctx.get(*k).map(|v|{
            let text=v.text();
            let shown=match serde_json::from_str::<Value>(&text){Ok(Value::String(s))=>s,_=>text.to_string()};
            format!("{k}:{shown}")
        })).collect();
        if !bad.is_empty(){return Err(EsError::illegal_argument(format!("unexpected metadata [{}] in source",bad.join(", "))));}
        ctx.insert("_index".into(),Source::from_value(&Value::String(index.into())));
        for (key,value) in [("_id",id),("_routing",routing)]{if let Some(value)=value{ctx.insert(key.into(),Source::from_value(&Value::String(value.into())));}}
        Ok(Self{ctx:Source::Object(ctx),ingest:Source::Object(BTreeMap::from([("timestamp".into(),Source::from_value(&Value::String(timestamp)))])),rerouted:false})
    }
    /// Put a top-level metadata value (already valid) without path parsing.
    pub fn set_meta(&mut self,key:&str,value:Source){if let Source::Object(m)=&mut self.ctx{m.insert(key.into(),value);}}
    pub fn get(&self,path:&str)->Result<Option<&Source>> {
        if super::flexible_access(){return self.get_flexible(path);}
        self.get_classic(path)
    }
    /// [`Self::get`] with ES's flexible access pattern: a key missing in an
    /// object is looked up joined with the following path elements (literal
    /// dotted field names); lists cannot be indexed.
    fn get_flexible(&self,path:&str)->Result<Option<&Source>> {
        let (ingest,segments)=parts(path)?;let mut cur=if ingest{&self.ingest}else{&self.ctx};
        let mut i=0;
        while i<segments.len() {
            cur=match cur {
                Source::Object(m)=>match super::doc::flexible_step(|k|m.contains_key(k),&segments,i,segments.len()){Ok((key,next))=>{i=next;m.get(key.as_str()).expect("key found above")},Err(_)=>return Ok(None)},
                Source::Array(_)=>return Err(invalid(path)),
                Source::Scalar(v) if v.get()=="null"=>return Ok(None),
                other=>return Err(EsError::illegal_argument(format!("cannot resolve [{}] from object of type [{}] as part of path [{path}]",segments[i],type_of(other)))),
            };
        }Ok(Some(cur))
    }
    /// The flexible "not present" error names the joined key it tried last.
    fn missing_flexible(&self,path:&str)->EsError {
        let Ok((ingest,segments))=parts(path) else {return missing(path)};let mut cur=if ingest{&self.ingest}else{&self.ctx};
        let mut i=0;
        while i<segments.len() {
            match cur {
                Source::Object(m)=>match super::doc::flexible_step(|k|m.contains_key(k),&segments,i,segments.len()){Ok((key,next))=>{i=next;cur=&m[key.as_str()];},Err(tried)=>return EsError::illegal_argument(format!("field [{tried}] not present as part of path [{path}]"))},
                _=>break,
            }
        }missing(path)
    }
    /// [`Self::get`] with the classic access pattern (also what mustache
    /// templates resolve names with, whatever the pipeline's pattern).
    pub fn get_classic(&self,path:&str)->Result<Option<&Source>> {
        let (ingest,segments)=parts(path)?;let mut cur=if ingest{&self.ingest}else{&self.ctx};
        for segment in segments {
            cur=match cur {
                Source::Object(m)=>match m.get(segment){Some(v)=>v,None=>return Ok(None)},
                Source::Array(a)=>&a[list_index(segment,a.len())?],
                Source::Scalar(v) if v.get()=="null"=>return Ok(None),
                _=>return Err(EsError::illegal_argument(format!("cannot resolve [{segment}] as part of path [{path}]"))),
            };
        }Ok(Some(cur))
    }
    pub fn require(&self,path:&str)->Result<&Source>{match self.get(path)? {Some(v)=>Ok(v),None if super::flexible_access()=>Err(self.missing_flexible(path)),None=>Err(missing(path))}}
    /// ES `setFieldValue` with the flexible access pattern: missing objects
    /// are not created; the remaining elements become one dotted key in the
    /// deepest object that exists.
    fn set_flexible(&mut self,path:&str,value:Source)->Result<()> {
        let (ingest,segments)=parts(path)?;
        let Some((last,_))=segments.split_last() else {
            if matches!(value,Source::Object(_)){self.ingest=value;return Ok(());}
            return Err(EsError::illegal_argument("_ingest must be an object"));
        };
        let limit=segments.len()-1;let mut leaf=last.to_string();let mut at_root=true;
        let mut cur=if ingest{&mut self.ingest}else{&mut self.ctx};
        let mut i=0;
        while i<limit {
            let step=match &*cur {
                Source::Object(m)=>super::doc::flexible_step(|k|m.contains_key(k),&segments,i,limit),
                Source::Array(_)=>return Err(invalid(path)),
                Source::Scalar(v) if v.get()=="null"=>return Err(EsError::illegal_argument(format!("cannot resolve [{}] from null as part of path [{path}]",segments[i]))),
                other=>return Err(EsError::illegal_argument(format!("cannot resolve [{}] from object of type [{}] as part of path [{path}]",segments[i],type_of(other)))),
            };
            match step {
                Ok((key,next))=>{cur=match cur {Source::Object(m)=>m.get_mut(&key).expect("key found above"),_=>unreachable!("an object above")};at_root=false;i=next;}
                Err(combined)=>{leaf=format!("{combined}.{leaf}");break;}
            }
        }
        match cur {
            Source::Object(m)=>{
                if !ingest&&at_root&&super::doc::METADATA_KEYS.contains(&leaf.as_str()) {
                    let metadata:Value=serde_json::from_str(&value.text())?;
                    super::doc::validate_meta(&leaf,&metadata)?;
                }
                m.insert(leaf,value);Ok(())
            }
            Source::Array(_)=>Err(invalid(path)),
            Source::Scalar(v) if v.get()=="null"=>Err(EsError::illegal_argument(format!("cannot set [{leaf}] with null parent as part of path [{path}]"))),
            other=>Err(EsError::illegal_argument(format!("cannot set [{leaf}] with parent object of type [{}] as part of path [{path}]",type_of(other)))),
        }
    }
    /// ES `removeField` with the flexible access pattern.
    fn remove_flexible(&mut self,path:&str)->Result<Source> {
        let (ingest,segments)=parts(path)?;
        let (last,_)=segments.split_last().ok_or_else(||EsError::illegal_argument("cannot remove _ingest"))?;
        let limit=segments.len()-1;let mut leaf=last.to_string();
        let mut cur=if ingest{&mut self.ingest}else{&mut self.ctx};
        let mut i=0;
        while i<limit {
            let step=match &*cur {
                Source::Object(m)=>super::doc::flexible_step(|k|m.contains_key(k),&segments,i,limit),
                Source::Array(_)=>return Err(invalid(path)),
                Source::Scalar(v) if v.get()=="null"=>return Err(missing(path)),
                other=>return Err(EsError::illegal_argument(format!("cannot resolve [{}] from object of type [{}] as part of path [{path}]",segments[i],type_of(other)))),
            };
            match step {
                Ok((key,next))=>{cur=match cur {Source::Object(m)=>m.get_mut(&key).expect("key found above"),_=>unreachable!("an object above")};i=next;}
                Err(combined)=>{leaf=format!("{combined}.{leaf}");break;}
            }
        }
        match cur {
            Source::Object(m)=>m.remove(&leaf).ok_or_else(||EsError::illegal_argument(format!("field [{leaf}] not present as part of path [{path}]"))),
            Source::Array(_)=>Err(invalid(path)),
            _=>Err(missing(path)),
        }
    }
    pub fn set(&mut self,path:&str,value:Source)->Result<()> {
        if super::flexible_access(){return self.set_flexible(path,value);}
        let (ingest,segments)=parts(path)?;let mut cur=if ingest{&mut self.ingest}else{&mut self.ctx};
        if segments.is_empty(){if matches!(value,Source::Object(_)){*cur=value;return Ok(());}return Err(EsError::illegal_argument("_ingest must be an object"));}
        let (leaf,parents)=segments.split_last().unwrap();
        if !ingest&&parents.is_empty()&&super::doc::METADATA_KEYS.contains(leaf) {
            let metadata:Value=serde_json::from_str(&value.text())?;
            super::doc::validate_meta(leaf,&metadata)?;
        }
        for segment in parents {
            if matches!(cur,Source::Scalar(v) if v.get()=="null"){*cur=object();}
            cur=match cur {
                Source::Object(m)=>m.entry((*segment).into()).or_insert_with(object),
                Source::Array(a)=>{let i=list_index(segment,a.len())?;&mut a[i]},
                _=>return Err(EsError::illegal_argument(format!("cannot set [{segment}] as part of path [{path}]"))),
            };
        }
        if matches!(cur,Source::Scalar(v) if v.get()=="null"){*cur=object();}
        match cur {
            Source::Object(m)=>{m.insert((*leaf).into(),value);},
            Source::Array(a)=>{let i=list_index(leaf,a.len())?;a[i]=value;},
            _=>return Err(EsError::illegal_argument(format!("cannot set [{path}]"))),
        }Ok(())
    }
    pub fn remove(&mut self,path:&str)->Result<Source> {
        if super::flexible_access(){return self.remove_flexible(path);}
        let (ingest,segments)=parts(path)?;let mut cur=if ingest{&mut self.ingest}else{&mut self.ctx};
        let (leaf,parents)=segments.split_last().ok_or_else(||EsError::illegal_argument("cannot remove _ingest"))?;
        for segment in parents {cur=match cur {Source::Object(m)=>m.get_mut(*segment).ok_or_else(||missing(path))?,Source::Array(a)=>{let i=list_index(segment,a.len())?;&mut a[i]},_=>return Err(missing(path))};}
        match cur {Source::Object(m)=>m.remove(*leaf).ok_or_else(||missing(path)),Source::Array(a)=>{let i=list_index(leaf,a.len())?;Ok(a.remove(i))},_=>Err(missing(path))}
    }
    pub fn append(&mut self,path:&str,value:Source,duplicates:bool)->Result<()> {
        let old=self.get(path)?.cloned();
        let mut values=match old {None=>Vec::new(),Some(Source::Array(a))=>a,Some(v)=>vec![v]};
        let incoming=match value {Source::Array(a)=>a,v=>vec![v]};
        for v in incoming {if duplicates||!values.iter().any(|old|old.equivalent(&v)){values.push(v);}}
        self.set(path,Source::Array(values))
    }
    pub fn script_context(&self)->Source { self.ctx.clone() }
    pub fn install_context(&mut self,context:Source)->Result<()> {
        if !matches!(&context,Source::Object(_)){return Err(EsError::illegal_argument("ingest context must be an object"));}
        self.ctx=context;Ok(())
    }
    pub fn meta_str(&self,key:&str)->Option<String>{self.ctx.get(key).and_then(|v|serde_json::from_str::<Value>(&v.text()).ok()).filter(|v|!v.is_null()).map(|v|super::doc::value_to_string(&v))}
    pub fn source_value(&self)->Source {let mut value=self.ctx.clone();if let Source::Object(m)=&mut value {for key in super::doc::METADATA_KEYS {m.remove(*key);}}value}
    /// [`SourceDocument::into_parts`], the source handed back whatever the
    /// metadata validation says (every metadata key removed from it).
    pub fn split_meta(self)->(Source,Result<serde_json::Map<String,Value>>) {
        let mut source=self.ctx;let mut removed=Vec::new();
        if let Source::Object(m)=&mut source {for key in super::doc::METADATA_KEYS {if let Some(value)=m.remove(*key){removed.push((*key,value));}}}
        let metadata=(|| {
            let mut metadata=serde_json::Map::new();
            for (key,value) in removed {let value:Value=serde_json::from_str(&value.text())?;super::doc::validate_meta(key,&value)?;metadata.insert(key.into(),value);}
            Ok(metadata)
        })();
        (source,metadata)
    }
    pub fn into_parts(self)->Result<(Source,serde_json::Map<String,Value>)> {
        let mut source=self.ctx;let mut metadata=serde_json::Map::new();
        if let Source::Object(m)=&mut source {for key in super::doc::METADATA_KEYS {if let Some(value)=m.remove(*key){let value:Value=serde_json::from_str(&value.text())?;super::doc::validate_meta(key,&value)?;metadata.insert((*key).into(),value);}}}
        Ok((source,metadata))
    }
    pub fn output(&self)->Source {
        let mut source=self.ctx.clone();let mut output=BTreeMap::new();
        if let Source::Object(m)=&mut source {for &key in super::doc::METADATA_KEYS {if let Some(v)=m.remove(key){if v.text()!="null" {let v=if key=="_version" {Source::from_value(&Value::String(v.text()))}else{v};output.insert(key.into(),v);}}}}
        output.insert("_source".into(),source);output.insert("_ingest".into(),self.ingest.clone());Source::Object(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn field_operations_never_decode_java_string_values() {
        let mut doc=SourceDocument::new(Source::parse(r#"{"s":"\ud800","a":["\udfff"],"nested":{"s":"\ud83d\ude00"}}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        doc.set("copy",doc.require("s").unwrap().clone()).unwrap();
        let moved=doc.remove("s").unwrap();doc.set("renamed",moved).unwrap();
        doc.append("a",doc.require("copy").unwrap().clone(),false).unwrap();
        doc.append("a",Source::parse(r#""\uD800""#).unwrap(),false).unwrap();
        assert!(doc.get("s").unwrap().is_none());
        assert!(doc.require("copy").unwrap().equivalent(doc.require("renamed").unwrap()));
        assert!(doc.require("a").unwrap().equivalent(&Source::parse(r#"["\udfff","\ud800"]"#).unwrap()));
        let mut context=doc.script_context();
        super::super::Script::new("ctx.generated=(char)55296".into(),"painless".into(),serde_json::json!({})).execute_source(&mut context).unwrap();
        doc.install_context(context).unwrap();
        assert!(doc.require("generated").unwrap().equivalent(&Source::parse(r#""\ud800""#).unwrap()));
    }
}
