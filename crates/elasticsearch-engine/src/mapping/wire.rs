//! Borrowed mapping serialization for replication. No intermediate JSON tree.
use super::*;
use serde::{Serialize, Serializer, ser::{SerializeMap, SerializeSeq}};

pub(super) struct Wire<'a>(pub &'a Mappings);
struct DynamicWire(Dynamic);
impl Serialize for DynamicWire {
    fn serialize<S: Serializer>(&self, s:S)->std::result::Result<S::Ok,S::Error> {
        s.serialize_str(match self.0 { Dynamic::True=>"true", Dynamic::False=>"false", Dynamic::Strict=>"strict", Dynamic::Runtime=>"runtime" })
    }
}
struct Properties<'a>(&'a BTreeMap<String,Mapper>,bool);
impl Serialize for Properties<'_> {
    fn serialize<S: Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error> {
        let mut out=s.serialize_map(None)?;
        for(k,v)in self.0 { if !self.1 || (k!=crate::tsdb::TSID_FIELD && k!=crate::tsdb::ROUTING_HASH_FIELD) { out.serialize_entry(k,&MapperWire(v))?; } }
        out.end()
    }
}
struct MapperWire<'a>(&'a Mapper);
impl Serialize for MapperWire<'_> {
    fn serialize<S: Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error> {
        match self.0 { Mapper::Field(f)=>FieldWire(f).serialize(s), Mapper::Object(o)=> {
            let mut m=s.serialize_map(None)?;
            if o.nested {m.serialize_entry("type","nested")?;} else if o.properties.is_empty(){m.serialize_entry("type","object")?;}
            if let Some(d)=o.dynamic {m.serialize_entry("dynamic",&DynamicWire(d))?;}
            if o.enabled==Some(false){m.serialize_entry("enabled",&false)?;}
            for(k,v)in &o.params {m.serialize_entry(k,v)?;}
            if !o.properties.is_empty(){m.serialize_entry("properties",&Properties(&o.properties,false))?;}
            m.end()
        }}
    }
}
struct FieldWire<'a>(&'a FieldMapping);
struct Fields<'a>(&'a BTreeMap<String,FieldMapping>);
impl Serialize for Fields<'_> {
    fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{let mut m=s.serialize_map(None)?;for(k,v)in self.0{m.serialize_entry(k,&FieldWire(v))?;}m.end()}
}
impl Serialize for FieldWire<'_> {
    fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{
        let f=self.0;let mut m=s.serialize_map(None)?;m.serialize_entry("type",f.ty.name())?;
        for(k,v)in &f.params{m.serialize_entry(k,v)?;}
        if !f.fields.is_empty(){m.serialize_entry("fields",&Fields(&f.fields))?;}m.end()
    }
}
struct Templates<'a>(&'a [DynamicTemplate]);
struct Template<'a>(&'a DynamicTemplate);
impl Serialize for Template<'_>{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{let mut m=s.serialize_map(Some(1))?;m.serialize_entry(&self.0.name,&self.0.raw)?;m.end()}}
impl Serialize for Templates<'_>{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{let mut a=s.serialize_seq(Some(self.0.len()))?;for t in self.0{a.serialize_element(&Template(t))?;}a.end()}}
struct Source<'a>(&'a Mappings,&'a Value);
impl Serialize for Source<'_>{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{
    let Some(src)=self.1.as_object() else{return self.1.serialize(s)};
    let mut m=s.serialize_map(None)?;for(k,v)in src{if k!="mode"{m.serialize_entry(k,v)?;}}
    if src.contains_key("mode")||self.0.source_mode_setting.is_some(){if let Some(mode)=self.0.effective_source_mode(){m.serialize_entry("mode",match mode{crate::synthetic::SourceMode::Stored=>"stored",crate::synthetic::SourceMode::Synthetic=>"synthetic",crate::synthetic::SourceMode::Disabled=>"disabled"})?;}}
    m.end()
}}
struct Prefixes<'a>(&'a BTreeMap<String,PrefixProps>);
struct Prefix<'a>(&'a PrefixProps);
impl Serialize for Prefix<'_>{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{let mut m=s.serialize_map(None)?;
    if let Some(d)=self.0.dynamic{m.serialize_entry("dynamic",&DynamicWire(d))?;}
    if let Some(p)=self.0.passthrough{m.serialize_entry("passthrough",&p)?;}
    if let Some(e)=self.0.enabled{m.serialize_entry("enabled",&e)?;}m.end()
}}
impl Serialize for Prefixes<'_>{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{let mut m=s.serialize_map(None)?;for(k,v)in self.0{m.serialize_entry(k,&Prefix(v))?;}m.end()}}
#[derive(Serialize)] struct Enabled {enabled:bool}
#[derive(Serialize)] struct IdMode<'a>{mode:&'a str}
fn is_false(v:&bool)->bool{!*v}
#[derive(Serialize)] struct Routing {#[serde(skip_serializing_if="is_false")] required:bool,#[serde(skip_serializing_if="is_false")] doc_values:bool}
impl Serialize for Wire<'_>{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{
    let v=self.0;let mut m=s.serialize_map(None)?;
    if let Some(d)=v.root.dynamic{m.serialize_entry("dynamic",&DynamicWire(d))?;}
    if let Some(e)=v.root.enabled{m.serialize_entry("enabled",&e)?;}
    if !v.dynamic_templates.is_empty()||v.dt_explicit{m.serialize_entry("dynamic_templates",&Templates(&v.dynamic_templates))?;}
    if let Some(d)=v.date_detection{m.serialize_entry("date_detection",&d)?;}
    if let Some(d)=v.numeric_detection{m.serialize_entry("numeric_detection",&d)?;}
    if let Some(f)=&v.dynamic_date_formats{m.serialize_entry("dynamic_date_formats",f)?;}
    if let Some(source)=&v.source{m.serialize_entry("_source",&Source(v,source))?;}
    if let Some(r)=&v.routing{let r=Routing{required:r.get("required").and_then(Value::as_bool).unwrap_or(false),doc_values:r.get("doc_values").and_then(Value::as_bool).unwrap_or(false)};if r.required||r.doc_values{m.serialize_entry("_routing",&r)?;}}
    if let Some(x)=&v.meta{m.serialize_entry("_meta",x)?;}
    if let Some(x)=&v.runtime{m.serialize_entry("runtime",x)?;}
    if let Some(x)=&v.size{m.serialize_entry("_size",x)?;}
    if v.data_stream_timestamp{m.serialize_entry("_data_stream_timestamp",&Enabled{enabled:true})?;}
    if let Some(mode)=&v.id_mode{m.serialize_entry("_id",&IdMode{mode})?;}
    if let Some(so)=v.root.params.get("subobjects"){m.serialize_entry("subobjects",so)?;}
    if !v.root.prefix.is_empty(){m.serialize_entry("prefix_properties",&Prefixes(&v.root.prefix))?;}
    if v.root.properties.keys().any(|k|k!=crate::tsdb::TSID_FIELD && k!=crate::tsdb::ROUTING_HASH_FIELD){m.serialize_entry("properties",&Properties(&v.root.properties,true))?;}
    m.end()
}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_mapping_wire_matches_public_shape_and_admits_before_string_growth() {
        for input in [
            json!({"properties":{"text":{"type":"text","fields":{"raw":{"type":"keyword"}}},"nested":{"type":"nested","properties":{"n":{"type":"long"}}}}}),
            json!({"dynamic":"strict","dynamic_templates":[],"_source":{"mode":"stored","excludes":["secret"]},"_routing":{"required":true},"_meta":{"label":"λ"},"properties":{"empty":{"type":"object","enabled":false}}}),
        ] {
            let mut mapping=Mappings::parse(&input,&AnalysisRegistry::default()).unwrap();
            mapping.source_mode_setting=Some("synthetic".into());
            let budget=crate::memory::Breakers::leak_for_tests(1<<20);
            budget.apply_settings(&|k|(k=="indices.breaker.total.use_real_memory").then(||"false".into()));
            let (encoded,owner)=mapping.to_json_admitted_with(budget).unwrap();
            assert_eq!(serde_json::from_str::<Value>(&encoded).unwrap(),mapping.to_json());
            assert!(owner.bytes()>=encoded.capacity() as u64);
        }
        let mut mapping=Mappings::default();mapping.meta=Some(json!({"payload":"x".repeat(8192)}));
        let tiny=crate::memory::Breakers::leak_for_tests(1024);
        tiny.apply_settings(&|k|(k=="indices.breaker.total.use_real_memory").then(||"false".into()));
        assert_eq!(mapping.to_json_admitted_with(tiny).err().unwrap().kind,"circuit_breaking_exception");
        assert_eq!(tiny.stats().iter().find(|s|s.name=="request").unwrap().estimated,0);
    }
}
