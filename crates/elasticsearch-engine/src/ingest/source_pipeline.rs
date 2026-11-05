//! Ingest execution with lossless values. Conversion to the legacy processor
//! interface is checked, never replacement-character decoding.
use super::*;
use super::source_doc::SourceDocument;
use crate::source_json::Source;

fn text(v:&Source)->Result<String> {
    let value:Value=serde_json::from_str(&v.text())?;
    value.as_str().map(str::to_owned).ok_or_else(||EsError::illegal_argument("expected a string"))
}
fn field(cfg:&Source,key:&str)->Result<String>{text(cfg.get(key).ok_or_else(||EsError::illegal_argument(format!("[{key}] required property is missing")))?)}
fn rendered_field(doc:&SourceDocument,cfg:&Source,key:&str)->Result<String>{doc.render_path(&field(cfg,key)?)}
fn media(cfg:&Source)->Result<super::jtext::MediaType>{let name=cfg.get("media_type").map(text).transpose()?.unwrap_or("application/json".into());super::jtext::MediaType::parse(&name).ok_or_else(||EsError::illegal_argument("unsupported media_type"))}
fn boolean(cfg:&Source,key:&str,default:bool)->bool {cfg.get(key).and_then(|v|serde_json::from_str::<Value>(&v.text()).ok()).and_then(|v|v.as_bool()).unwrap_or(default)}
fn legacy(n:&ProcessorNode,doc:&mut SourceDocument,cx:&PipelineContext)->PResult {
    let ctx:Value=serde_json::from_str(&doc.ctx.text()).map_err(EsError::from)?;
    let ingest:Value=serde_json::from_str(&doc.ingest.text()).map_err(EsError::from)?;
    let mut old=IngestDocument{ctx,ingest,rerouted:doc.rerouted};
    let result=n.imp.process(&mut old,cx);
    doc.ctx=Source::from_value(&old.ctx);doc.ingest=Source::from_value(&old.ingest);doc.rerouted=old.rerouted;
    result
}
/// These processors read one configured field and write one configured target.
/// Project only that field into their Unicode value domain; unrelated source
/// values never leave SourceDocument and are never normalized or substituted.
fn single_field_legacy(n:&ProcessorNode,doc:&mut SourceDocument,cx:&PipelineContext)->PResult {
    let path=field(&n.source_config,"field")?;
    let target=n.source_config.get("target_field").map(text).transpose()?.unwrap_or_else(||path.clone());
    let mut selected=IngestDocument{ctx:json!({}),ingest:json!({}),rerouted:false};
    if let Some(value)=doc.get(&path)?{selected.set(&path,serde_json::from_str(&value.text()).map_err(EsError::from)?)?;}
    n.imp.process(&mut selected,cx)?;
    if let Some(value)=selected.get(&target)?{doc.set(&target,Source::from_value(value))?;}
    Ok(())
}
fn process(n:&ProcessorNode,doc:&mut SourceDocument,cx:&PipelineContext,trace:&mut Option<Vec<Source>>)->PResult {
    let cfg=&n.source_config;
    // Preserve existing processor semantics whenever its complete value domain
    // is Unicode. Script and nested execution can introduce lone units, so
    // they always stay on the lossless path even for ordinary input.
    // `set` runs natively on the source document: through the legacy path
    // each document was serialized and reparsed four times per processor
    // (the validity check, then the legacy round trip).
    if !matches!(n.kind.as_str(),"script"|"pipeline"|"foreach"|"urldecode"|"csv"|"json"|"set") && serde_json::from_str::<Value>(&doc.ctx.text()).is_ok() && serde_json::from_str::<Value>(&doc.ingest.text()).is_ok() && serde_json::from_str::<Value>(&cfg.text()).is_ok() {return legacy(n,doc,cx);}
    match n.kind.as_str() {
        "script"=>{
            // An inline script's spec is the same at every run: built once.
            let cached=n.script.get_or_init(|| {
                let spec=super::source_config::script_spec(cfg);
                if spec.get("id").is_some() {return None;}
                Script::from_source(&spec).ok()
            });
            match cached {
                // On the document's ctx itself (left as it was on failure).
                Some(script)=>script.execute_on_ctx(&mut doc.ctx)?,
                None=>{let mut context=doc.script_context();Script::from_source(&super::source_config::script_spec(cfg))?.execute_source(&mut context)?;doc.install_context(context)?;}
            }
        },
        "set"=>{
            let path=rendered_field(doc,cfg,"field")?;
            if !boolean(cfg,"override",true)&&doc.get(&path)?.is_some_and(|v|v.text()!="null"){return Ok(());}
            let ignore_empty=boolean(cfg,"ignore_empty_value",false);
            // ES `getFieldValue(copyFrom, Object.class, ignoreEmptyValue)`: with
            // `ignore_empty_value` a missing source is null, so nothing is set
            // (apm-data's `logs-apm@pipeline` copies an absent
            // `data_stream.dataset`; requiring it sent every apm log document
            // to the failure store). As the legacy `set` (`SetP`) does.
            let value=if let Some(copy)=cfg.get("copy_from"){
                let path=doc.render_path(&text(copy)?)?;
                if ignore_empty{match doc.get(&path){Ok(Some(v))=>v.clone(),_=>return Ok(())}}else{doc.require(&path)?.clone()}
            }else{doc.render_source(cfg.get("value").ok_or_else(||EsError::illegal_argument("[value] required property is missing"))?,media(cfg)?)?};
            if ignore_empty&&(value.text()=="null"||value.text()=="\"\""){return Ok(());}
            doc.set(&path,value)?;
        },
        "append"=>{let path=rendered_field(doc,cfg,"field")?;let value=doc.render_source(cfg.get("value").ok_or_else(||EsError::illegal_argument("[value] required property is missing"))?,media(cfg)?)?;doc.append(&path,value,boolean(cfg,"allow_duplicates",true))?;},
        "rename"=>{
            let path=rendered_field(doc,cfg,"field")?;let target=rendered_field(doc,cfg,"target_field")?;
            if !doc.get(&path)?.is_some(){if boolean(cfg,"ignore_missing",false){return Ok(());}doc.require(&path)?;}
            if doc.get(&target)?.is_some()&&!boolean(cfg,"override",false){return Err(EsError::illegal_argument(format!("field [{target}] already exists")).into());}
            let value=doc.remove(&path)?;doc.set(&target,value)?;
        },
        "remove" if cfg.get("keep").is_some()=>{
            let names=cfg.get("keep").unwrap();let names=match names{Source::Array(a)=>a.clone(),v=>vec![v.clone()]};
            let mut kept=SourceDocument{ctx:Source::Object(BTreeMap::new()),ingest:doc.ingest.clone(),rerouted:doc.rerouted};
            for name in names{let path=doc.render_path(&text(&name)?)?;if let Some(value)=doc.get(&path)?{kept.set(&path,value.clone())?;}}
            if let(Source::Object(source),Source::Object(target))=(&doc.ctx,&mut kept.ctx){for key in super::doc::METADATA_KEYS{if let Some(value)=source.get(*key){target.insert((*key).into(),value.clone());}}}
            doc.ctx=kept.ctx;
        },
        "remove" if cfg.get("field").is_some()=>{
            let names=cfg.get("field").unwrap();let names=match names{Source::Array(a)=>a.clone(),v=>vec![v.clone()]};
            for name in names{let path=doc.render_path(&text(&name)?)?;if doc.get(&path)?.is_none()&&boolean(cfg,"ignore_missing",false){continue;}doc.remove(&path)?;}
        },
        "uppercase"|"lowercase"|"trim"=>{
            let path=rendered_field(doc,cfg,"field")?;
            let Some(value)=doc.get(&path)?.cloned() else{if boolean(cfg,"ignore_missing",false){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] not present")).into());};
            let transform=|value:&Source|->Result<Source>{crate::script::ingest_string_transform(value,match n.kind.as_str(){"uppercase"=>"toUpperCase","lowercase"=>"toLowerCase",_=>"trim"})};
            let value=match value{Source::Array(a)=>Source::Array(a.iter().map(transform).collect::<Result<_>>()?),v=>transform(&v)?};
            let target=cfg.get("target_field").map(|v|text(v).and_then(|s|doc.render_path(&s))).transpose()?.unwrap_or(path);doc.set(&target,value)?;
        },
        "attachment"|"user_agent"|"registered_domain"|"network_direction"|"community_id"=>{
            // Explicit read/write contracts avoid converting unrelated fields
            // for processors whose inputs have a Unicode/numeric domain.
            let mut inputs=Vec::new();let mut outputs=Vec::new();
            let configured=|key:&str,default:&str|cfg.get(key).map(text).transpose().map(|v|v.unwrap_or(default.into()));
            match n.kind.as_str(){
                "network_direction"=>{inputs.extend([configured("source_ip","source.ip")?,configured("destination_ip","destination.ip")?]);if let Some(v)=cfg.get("internal_networks_field"){inputs.push(text(v)?);}outputs.push(configured("target_field","network.direction")?);},
                "community_id"=>{for(key,default)in [("source_ip","source.ip"),("source_port","source.port"),("destination_ip","destination.ip"),("destination_port","destination.port"),("iana_number","network.iana_number"),("transport","network.transport"),("icmp_type","icmp.type"),("icmp_code","icmp.code")]{inputs.push(configured(key,default)?);}outputs.push(configured("target_field","network.community_id")?);},
                "registered_domain"=>{inputs.push(field(cfg,"field")?);let target=configured("target_field","")?;for key in ["domain","registered_domain","top_level_domain","subdomain"]{outputs.push(if target.is_empty(){key.into()}else{format!("{target}.{key}")});}},
                kind=>{inputs.push(field(cfg,"field")?);outputs.push(configured("target_field",if kind=="attachment"{"attachment"}else{"user_agent"})?);if kind=="attachment"{for key in ["indexed_chars_field","resource_name"]{if let Some(v)=cfg.get(key){inputs.push(text(v)?);}}}},
            }
            let mut selected=IngestDocument{ctx:json!({}),ingest:json!({}),rerouted:false};
            for path in &inputs{if let Some(value)=doc.get(path)?{selected.set(path,serde_json::from_str(&value.text()).map_err(EsError::from)?)?;}}
            n.imp.process(&mut selected,cx)?;
            for path in outputs{if let Some(value)=selected.get(&path)?{doc.set(&path,Source::from_value(value))?;}}
            if n.kind=="attachment"&&boolean(cfg,"remove_binary",false){let path=field(cfg,"field")?;if selected.get(&path)?.is_none()&&doc.get(&path)?.is_some(){doc.remove(&path)?;}}
        },
        "reroute"=>{
            if let Some(destination)=cfg.get("destination") {
                doc.set("_index",doc.render_source(destination,super::jtext::MediaType::Json)?)?;
            } else {
                let index=doc.meta_str("_index").unwrap_or_default();
                let mut parts=index.splitn(3,'-');
                let old=[parts.next().unwrap_or(""),parts.next().unwrap_or(""),parts.next().ok_or_else(||EsError::illegal_argument("invalid data stream name"))?];
                let mut chosen=Vec::new();
                for (i,key) in ["type","dataset","namespace"].iter().enumerate(){
                    let fallback=Source::from_value(&json!(format!("{{{{data_stream.{key}}}}}")));
                    let value=cfg.get(key).unwrap_or(&fallback);
                    let options=match value{Source::Array(a)=>a.clone(),v=>vec![v.clone()]};
                    let mut selected=None;
                    for option in options {
                        let template=text(&option)?;
                        if !template.contains("{{"){selected=Some(option);break;}
                        let rendered=doc.render_source(&option,super::jtext::MediaType::Json)?;
                        let lower=crate::script::ingest_string_transform(&rendered,"toLowerCase")?;
                        let mut units=crate::script::source_from_json(&lower).string_units();
                        let bad=if i==2{b"\\/*?\"<>| ,#:".as_slice()}else{b"\\/*?\"<>| ,#:-".as_slice()};
                        for unit in &mut units{if *unit<128&&bad.contains(&(*unit as u8)){*unit=95;}}
                        units.truncate(100);
                        if !units.is_empty(){selected=Some(crate::script::source_to_json(&crate::script::Val::utf16(units))?);break;}
                    }
                    chosen.push(selected.unwrap_or_else(||Source::from_value(&json!(old[i]))));
                }
                let index=chosen.iter().map(text).collect::<Result<Vec<_>>>()?.join("-");
                doc.set("_index",Source::from_value(&json!(index)))?;
                for (path,value) in [("data_stream.type",chosen[0].clone()),("data_stream.dataset",chosen[1].clone()),("data_stream.namespace",chosen[2].clone()),("event.dataset",chosen[1].clone())]{
                    let dotted=matches!(&doc.ctx,Source::Object(m)if m.contains_key(path));
                    if path=="event.dataset"&&!dotted&&doc.get(path)?.is_none(){continue;}
                    if dotted{if let Source::Object(m)=&mut doc.ctx{m.insert(path.into(),value);}}else{doc.set(path,value)?;}
                }
            }
            doc.rerouted=true;
        },
        "fingerprint"=>{
            fn bytes_domain(value:&Source)->Result<Value>{Ok(match value{
                Source::Array(values)=>Value::Array(values.iter().map(bytes_domain).collect::<Result<_>>()?),
                Source::Object(values)=>Value::Object(values.iter().map(|(k,v)|Ok((k.clone(),bytes_domain(v)?))).collect::<Result<_>>()?),
                Source::Scalar(raw)if raw.get().starts_with('"')=>json!(char::decode_utf16(crate::script::source_from_json(value).string_units()).map(|c|c.unwrap_or('?')).collect::<String>()),
                _=>serde_json::from_str(&value.text())?,
            })}
            let fields:Vec<String>=serde_json::from_str(&cfg.get("fields").ok_or_else(||EsError::illegal_argument("fields required"))?.text()).map_err(EsError::from)?;
            let target=cfg.get("target_field").map(text).transpose()?.unwrap_or("fingerprint".into());
            let mut selected=IngestDocument{ctx:json!({}),ingest:json!({}),rerouted:false};
            for path in fields{if let Some(value)=doc.get(&path)?{selected.set(&path,bytes_domain(value)?)?;}}
            n.imp.process(&mut selected,cx)?;
            if let Some(value)=selected.get(&target)?{doc.set(&target,Source::from_value(value))?;}
        },
        "uri_parts"=>{
            let path=field(cfg,"field")?;let target=cfg.get("target_field").map(text).transpose()?.unwrap_or("url".into());
            let Some(value)=doc.get(&path)?.filter(|v|v.text()!="null")else{if boolean(cfg,"ignore_missing",false){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] missing or null")).into());};
            let result=super::source_uri::parts(value,boolean(cfg,"keep_original",true))?;doc.set(&target,result)?;
            if boolean(cfg,"remove_if_successful",false)&&path!=target{doc.remove(&path)?;}
        },
        "date"|"date_index_name"=>{
            // Date parsing consumes only the selected scalar. Render its
            // template dependencies against the full lossless document first.
            let mut config=cfg.clone();
            if let Source::Object(config)=&mut config {
                for key in ["timezone","locale","index_name_prefix","date_rounding","index_name_format"] {
                    if let Some(value)=config.get_mut(key){*value=doc.render_source(value,super::jtext::MediaType::Plain)?;}
                }
            }
            let mut metadata:Map<String,Value>=serde_json::from_str(&config.text()).map_err(EsError::from)?;
            for key in ["if","on_failure","ignore_failure","tag","description"]{metadata.shift_remove(key);}
            let mut compiler=Cfg{kind:n.kind.clone(),tag:None,m:metadata,raw:None};
            let processor=super::processors::build(&mut compiler)?;
            let path=field(cfg,"field")?;
            let target=if n.kind=="date_index_name"{"_index".into()}else{cfg.get("target_field").map(text).transpose()?.unwrap_or("@timestamp".into())};
            let mut selected=IngestDocument{ctx:json!({}),ingest:json!({}),rerouted:false};
            if let Some(value)=doc.get(&path)?{selected.set(&path,serde_json::from_str(&value.text()).map_err(EsError::from)?)?;}
            processor.process(&mut selected,cx)?;
            if let Some(value)=selected.get(&target)?{doc.set(&target,Source::from_value(value))?;}
        },
        "set_security_user"=>{
            let mut selected=IngestDocument{ctx:json!({}),ingest:json!({}),rerouted:false};
            n.imp.process(&mut selected,cx)?;
            let path=field(cfg,"field")?;
            if let Some(value)=selected.get(&path)?{doc.set(&path,Source::from_value(value))?;}
        },
        "grok"|"redact"=>{
            let path=field(cfg,"field")?;
            let Some(value)=doc.get(&path)?.cloned().filter(|v|v.text()!="null")else{if boolean(cfg,"ignore_missing",n.kind=="redact"){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] missing or null")).into());};
            let original=value.clone();
            let value=crate::script::source_from_json(&value);
            if !matches!(&value,crate::script::Val::Str(_)|crate::script::Val::Utf16(_)){return Err(EsError::illegal_argument("grok requires a string").into());}
            // Grok's matching/capture boundary uses Java String.getBytes(UTF_8).
            let encoded:String=char::decode_utf16(value.string_units()).map(|c|c.unwrap_or('?')).collect();
            let ecs=cfg.get("ecs_compatibility").map(text).transpose()?.as_deref()==Some("v1");
            let mut definitions=super::grok::bank_es(ecs).clone();
            if let Some(Source::Object(defs))=cfg.get("pattern_definitions"){for (key,value)in defs{
                let units=crate::script::source_from_json(value).string_units();definitions.insert(key.clone(),char::decode_utf16(units).map(|c|c.unwrap_or('?')).collect());
            }}
            let Some(Source::Array(patterns))=cfg.get("patterns")else{return Err(EsError::illegal_argument("patterns must be a list").into());};
            let mut matched=false;let mut redactions=Vec::new();
            for (index,pattern)in patterns.iter().enumerate(){
                let pattern:String=char::decode_utf16(crate::script::source_from_json(pattern).string_units()).map(|c|c.unwrap_or('?')).collect();
                let grok=super::grok::Grok::compile_es(&pattern,&definitions)?;
                if n.kind=="redact" {redactions.extend(grok.spans(&encoded));continue;}
                if let Some(captures)=grok.captures_checked(&encoded)? {
                    for (key,value)in captures{doc.set(&key,Source::from_value(&value))?;}
                    if boolean(cfg,"trace_match",false){doc.set("_ingest._grok_match_index",Source::from_value(&json!(index.to_string())))?;}
                    matched=true;break;
                }
            }
            if n.kind=="redact" {
                // A non-match returns the original Java string. A match
                // rebuilds from UTF8 bytes, so even unmatched lone units cross
                // the byte boundary only when an actual redaction occurs.
                if !redactions.is_empty(){
                    redactions.sort_by_key(|(start,end,_)|(*start,std::cmp::Reverse(*end)));
                    let prefix=cfg.get("prefix").map(|v|crate::script::source_from_json(v).string_units()).unwrap_or(vec![60]);
                    let suffix=cfg.get("suffix").map(|v|crate::script::source_from_json(v).string_units()).unwrap_or(vec![62]);
                    let mut output=Vec::new();let mut last=0;
                    for(start,end,label)in redactions{if start<last{continue;}output.extend(encoded[last..start].encode_utf16());output.extend_from_slice(&prefix);output.extend(label.encode_utf16());output.extend_from_slice(&suffix);last=end;}
                    output.extend(encoded[last..].encode_utf16());
                    let encoded_output:String=char::decode_utf16(output).map(|c|c.unwrap_or('?')).collect();
                    let result=Source::from_value(&json!(encoded_output));
                    let changed=!result.equivalent(&original);doc.set(&path,result)?;
                    if changed&&boolean(cfg,"trace_redact",false){doc.set("_ingest._redact._is_redacted",Source::from_value(&json!(true)))?;}
                }
            }else if !matched{return Err(EsError::illegal_argument(format!("Provided Grok expressions do not match field value: [{encoded}]")).into());}
        },
        "html_strip"=>{
            let path=rendered_field(doc,cfg,"field")?;
            let Some(value)=doc.get(&path)?.cloned().filter(|v|v.text()!="null") else{if boolean(cfg,"ignore_missing",false){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] missing or null")).into());};
            let transform=|value:&Source|->Result<Source>{
                let value=crate::script::source_from_json(value);
                if !matches!(value,crate::script::Val::Str(_)|crate::script::Val::Utf16(_)){return Err(EsError::illegal_argument("html_strip requires a string"));}
                crate::script::source_to_json(&crate::script::Val::utf16(super::util::html_strip_units(&value.string_units())))
            };
            let output=match value{Source::Array(a)=>Source::Array(a.iter().map(transform).collect::<Result<_>>()?),v=>transform(&v)?};
            let target=cfg.get("target_field").map(text).transpose()?.unwrap_or(path);doc.set(&target,output)?;
        },
        "dot_expander"=>{
            let name=field(cfg,"field")?;let path=cfg.get("path").map(text).transpose()?;
            let base=match &path{Some(p)=>doc.get(p)?.cloned(),None=>Some(doc.ctx.clone())};
            let Some(Source::Object(base))=base else{return Ok(());};
            let keys:Vec<_>=base.keys().filter(|k|k.contains('.')&&(name=="*"||**k==name)&&!super::doc::METADATA_KEYS.contains(&k.as_str())).cloned().collect();
            for key in keys {
                let value=if let Some(path)=&path {
                    let Some(Source::Object(mut parent))=doc.get(path)?.cloned()else{continue;};
                    let value=parent.remove(&key).unwrap();doc.set(path,Source::Object(parent))?;value
                }else if let Source::Object(parent)=&mut doc.ctx{parent.remove(&key).unwrap()}else{unreachable!()};
                let target=path.as_ref().map(|p|format!("{p}.{key}")).unwrap_or(key.clone());
                let segments:Vec<_>=target.split('.').collect();
                for i in 1..segments.len(){let prefix=segments[..i].join(".");if let Ok(Some(v))=doc.get(&prefix){if !matches!(v,Source::Object(_)|Source::Array(_))&&v.text()!="null"{return Err(EsError::illegal_argument(format!("cannot expand [{key}], because [{prefix}] is not an object field, but a value field")).into());}}}
                if doc.get(&target)?.is_some()&&!boolean(cfg,"override",false){doc.append(&target,value,true)?;}else{doc.set(&target,value)?;}
            }
        },
        "dissect"=>return super::processors::dissect_source(cfg,doc),
        "kv"=>{
            let path=field(cfg,"field")?;let value=doc.get(&path)?.cloned();
            let Some(value)=value.filter(|v|v.text()!="null")else{if boolean(cfg,"ignore_missing",false){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] doesn't exist")).into());};
            let split=cfg.get("field_split").ok_or_else(||EsError::illegal_argument("field_split required"))?;
            let separator=cfg.get("value_split").ok_or_else(||EsError::illegal_argument("value_split required"))?;
            let prefix=cfg.get("prefix").map(text).transpose()?.unwrap_or_default();
            let target=cfg.get("target_field").map(text).transpose()?;
            let trim=|value:&Source,key:&str|->Result<Source>{
                let units=crate::script::source_from_json(value).string_units();
                let Some(chars)=cfg.get(key)else{return Ok(value.clone());};let chars=crate::script::source_from_json(chars).string_units();
                let start=units.iter().position(|u|!chars.contains(u)).unwrap_or(units.len());
                let end=units.iter().rposition(|u|!chars.contains(u)).map(|i|i+1).unwrap_or(start);
                crate::script::source_to_json(&crate::script::Val::utf16(units[start..end].to_vec()))
            };
            for pair in crate::script::ingest_regex_split(&value,split,0)? {
                let parts=crate::script::ingest_regex_split(&pair,separator,2)?;
                if parts.len()!=2{return Err(EsError::illegal_argument(format!("field [{path}] does not contain value_split")).into());}
                let key=text(&trim(&parts[0],"trim_key")?)?;
                let selected=|name:&str|->Result<Option<bool>>{cfg.get(name).map(|names|serde_json::from_str::<Vec<String>>(&names.text()).map(|names|names.contains(&key)).map_err(EsError::from)).transpose()};
                if selected("include_keys")?==Some(false)||selected("exclude_keys")?==Some(true){continue;}
                let mut value=parts[1].clone();
                if boolean(cfg,"strip_brackets",false){let mut units=crate::script::source_from_json(&value).string_units();
                    if units.last().is_some_and(|u|[93,41,62,34,39].contains(u)){units.pop();}
                    if units.first().is_some_and(|u|[40,91,60,34,39].contains(u)){units.remove(0);}
                    value=crate::script::source_to_json(&crate::script::Val::utf16(units))?;
                }
                let value=trim(&value,"trim_value")?;
                let path=format!("{}{prefix}{key}",target.as_ref().map(|s|format!("{s}.")).unwrap_or_default());
                if doc.get(&path)?.is_some(){doc.append(&path,value,true)?;}else{doc.set(&path,value)?;}
            }
        },
        "json"=>{
            let path=field(cfg,"field")?;
            let parsed=super::source_data::parse(doc.require(&path)?,boolean(cfg,"allow_duplicate_keys",false),boolean(cfg,"strict_json_parsing",true))?;
            if boolean(cfg,"add_to_root",false){
                let Source::Object(values)=parsed else{return Err(EsError::illegal_argument("cannot add non-map fields to root of document").into());};
                if cfg.get("add_to_root_conflict_strategy").map(text).transpose()?.as_deref()==Some("merge"){doc.ctx.merge(&Source::Object(values));}
                else if let Source::Object(root)=&mut doc.ctx{root.extend(values);}
            }else{let target=cfg.get("target_field").map(text).transpose()?.unwrap_or(path);doc.set(&target,parsed)?;}
        },
        "urldecode"=>{
            let path=field(cfg,"field")?;let value=doc.get(&path)?.cloned();
            let Some(value)=value.filter(|v|v.text()!="null") else{if boolean(cfg,"ignore_missing",false){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] missing or null")).into());};
            let output=match value{Source::Array(a)=>Source::Array(a.iter().map(crate::script::ingest_url_decode).collect::<Result<_>>()?),v=>crate::script::ingest_url_decode(&v)?};
            let target=cfg.get("target_field").map(text).transpose()?.unwrap_or(path);doc.set(&target,output)?;
        },
        "csv"=>{
            let targets:Vec<String>=serde_json::from_str(&cfg.get("target_fields").ok_or_else(||EsError::illegal_argument("target_fields required"))?.text()).map_err(EsError::from)?;
            let processor=super::p_data::CsvP {
                field:field(cfg,"field")?,targets,
                sep:cfg.get("separator").map(text).transpose()?.unwrap_or(",".into()).chars().next().unwrap(),
                quote:cfg.get("quote").map(text).transpose()?.unwrap_or("\"".into()).chars().next().unwrap(),
                trim:boolean(cfg,"trim",false),empty_value:None,ignore_missing:boolean(cfg,"ignore_missing",false),
            };
            processor.process_source(doc,cfg.get("empty_value"))?;
        },
        "join"=>{
            let path=rendered_field(doc,cfg,"field")?;let separator=cfg.get("separator").ok_or_else(||EsError::illegal_argument("[separator] required"))?;
            let Source::Array(values)=doc.require(&path)? else{return Err(EsError::illegal_argument("join field must be a list").into())};
            let mut output=Vec::new();for (i,value) in values.iter().enumerate(){if i>0{output.extend(crate::script::source_from_json(separator).string_units());}output.extend(crate::script::source_from_json(value).string_units());}
            let value=crate::script::source_to_json(&crate::script::Val::utf16(output))?;
            let target=cfg.get("target_field").map(|v|text(v).and_then(|s|doc.render_path(&s))).transpose()?.unwrap_or(path);doc.set(&target,value)?;
        },
        "split"|"gsub"=>{
            let path=rendered_field(doc,cfg,"field")?;let Some(value)=doc.get(&path)?.cloned() else{if boolean(cfg,"ignore_missing",false){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] not present")).into());};
            let pattern=cfg.get(if n.kind=="split"{"separator"}else{"pattern"}).ok_or_else(||EsError::illegal_argument("regex pattern required"))?;
            let replacement=if n.kind=="gsub"{Some(cfg.get("replacement").ok_or_else(||EsError::illegal_argument("[replacement] required"))?)}else{None};
            let transform=|value:&Source|crate::script::ingest_regex_transform(value,pattern,replacement,boolean(cfg,"preserve_trailing",false));
            let output=if n.kind=="gsub" {match value {Source::Array(a)=>Source::Array(a.iter().map(transform).collect::<Result<_>>()?),v=>transform(&v)?}}else{transform(&value)?};
            let target=cfg.get("target_field").map(|v|text(v).and_then(|s|doc.render_path(&s))).transpose()?.unwrap_or(path);doc.set(&target,output)?;
        },
        "sort"=>{
            let path=rendered_field(doc,cfg,"field")?;let Source::Array(mut values)=doc.require(&path)?.clone() else{return Err(EsError::illegal_argument("sort field must be a list").into())};
            if values.iter().all(|v|matches!(crate::script::source_from_json(v),crate::script::Val::Str(_)|crate::script::Val::Utf16(_))) {
                values.sort_by_key(|v|crate::script::source_from_json(v).string_units());
                if cfg.get("order").map(text).transpose()?.as_deref()==Some("desc"){values.reverse();}
                let target=cfg.get("target_field").map(|v|text(v).and_then(|s|doc.render_path(&s))).transpose()?.unwrap_or(path);doc.set(&target,Source::Array(values))?;
            }else{return single_field_legacy(n,doc,cx);}
        },
        "convert" if cfg.get("type").map(text).transpose()?.as_deref()==Some("string")=>{
            let path=rendered_field(doc,cfg,"field")?;let Some(value)=doc.get(&path)?.cloned() else{if boolean(cfg,"ignore_missing",false){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] not present")).into());};
            let transform=|v:&Source|crate::script::source_to_json(&crate::script::Val::utf16(crate::script::source_from_json(v).string_units()));
            let value=match value{Source::Array(a)=>Source::Array(a.iter().map(transform).collect::<Result<_>>()?),v=>transform(&v)?};
            let target=cfg.get("target_field").map(|v|text(v).and_then(|s|doc.render_path(&s))).transpose()?.unwrap_or(path);doc.set(&target,value)?;
        },
        "convert" if cfg.get("type").map(text).transpose()?.as_deref()==Some("auto")=>{
            let path=field(cfg,"field")?;let value=doc.get(&path)?.cloned();
            let Some(value)=value.filter(|v|v.text()!="null")else{if boolean(cfg,"ignore_missing",false){return Ok(());}return single_field_legacy(n,doc,cx);};
            let convert=|value:&Source|->Result<Source>{
                match serde_json::from_str::<Value>(&value.text()) {
                    Ok(value)=>Ok(Source::from_value(&super::p_data::convert_one(&value,super::p_data::ConvType::Auto)?)),
                    Err(_) if matches!(crate::script::source_from_json(value),crate::script::Val::Utf16(_))=>Ok(value.clone()),
                    Err(e)=>Err(e.into()),
                }
            };
            let output=match value{Source::Array(a)=>Source::Array(a.iter().map(convert).collect::<Result<_>>()?),v=>convert(&v)?};
            let target=cfg.get("target_field").map(text).transpose()?.unwrap_or(path);doc.set(&target,output)?;
        },
        "convert"|"bytes"=>return single_field_legacy(n,doc,cx),
        "pipeline"=>{let name=rendered_field(doc,cfg,"name")?;if let Some(p)=cx.pipeline(&name){run(&p,doc,cx,trace)?;}else if !boolean(cfg,"ignore_missing_pipeline",false){return Err(EsError::illegal_argument(format!("pipeline with id [{name}] does not exist")).into());}},
        "foreach"=>{
            let path=rendered_field(doc,cfg,"field")?;let Some(value)=doc.get(&path)?.cloned() else{if boolean(cfg,"ignore_missing",false){return Ok(());}return Err(EsError::illegal_argument(format!("field [{path}] not present")).into());};
            let nested=cfg.get("processor").ok_or_else(||EsError::illegal_argument("[processor] required property is missing"))?;
            let wrapper=Source::Object(BTreeMap::from([("processors".into(),Source::Array(vec![nested.clone()]))]));
            let pipeline=compile_pipeline_source("_foreach",&wrapper)?;
            let previous=doc.get("_ingest._value")?.cloned();let previous_key=doc.get("_ingest._key")?.cloned();
            let result=(||->std::result::Result<Source,ProcessorError>{match value {
                Source::Array(mut values)=>{for value in &mut values{doc.set("_ingest._value",value.clone())?;nodes(&pipeline.processors,doc,cx,&mut None)?;*value=doc.require("_ingest._value")?.clone();}Ok(Source::Array(values))},
                Source::Object(values)=>{let mut result=BTreeMap::new();for(key,value)in values{doc.set("_ingest._key",Source::from_value(&json!(key)))?;doc.set("_ingest._value",value)?;nodes(&pipeline.processors,doc,cx,&mut None)?;let key=text(doc.require("_ingest._key")?)?;result.insert(key,doc.require("_ingest._value")?.clone());}Ok(Source::Object(result))},
                _=>Err(EsError::illegal_argument("foreach field must be a list or map").into()),
            }})();
            doc.set("_ingest._value",previous.unwrap_or_else(||Source::from_value(&Value::Null)))?;
            doc.set("_ingest._key",previous_key.unwrap_or_else(||Source::from_value(&Value::Null)))?;
            doc.set(&path,result?)?;
        },
        "fail"=>{let value=doc.render_source(cfg.get("message").ok_or_else(||EsError::illegal_argument("[message] required"))?,super::jtext::MediaType::Json)?;return Err(EsError::new(500,"fail_processor_exception",String::from_utf16_lossy(&crate::script::source_from_json(&value).string_units())).into());},
        "drop"=>return Err(ProcessorError::Dropped),
        "terminate"=>return Err(ProcessorError::Terminated),
        _=>return legacy(n,doc,cx),
    }Ok(())
}
fn condition(script:&Script,doc:&SourceDocument)->Result<bool> {
    let cs=script.compile_source(crate::script::ScriptContext::IngestConditional)?;
    let mut executor=cs.executor(None,None);executor.set_ctx(crate::script::source_from_json(&doc.script_context()));
    match executor.run().map_err(|e|cs.error(&e))? {crate::script::Val::Bool(v)=>Ok(v),_=>Err(EsError::illegal_argument("ingest condition must return a boolean"))}
}
fn source_trace_base(n: &ProcessorNode, condition: Option<bool>) -> BTreeMap<String, Source> {
    let mut entry = BTreeMap::from([
        ("processor_type".into(), Source::from_value(&json!(n.kind))),
        ("status".into(), Source::from_value(&json!("success"))),
    ]);
    if let Some(tag) = &n.tag { entry.insert("tag".into(), Source::from_value(&json!(tag))); }
    if let Some(description) = &n.description { entry.insert("description".into(), Source::from_value(&json!(description))); }
    if let (Some(script), Some(result)) = (&n.cond, condition) {
        let source = script.raw_script.as_ref().and_then(|s| s.get("source")).cloned()
            .unwrap_or_else(|| Source::from_value(&json!(script.source)));
        entry.insert("if".into(), Source::Object(BTreeMap::from([
            ("condition".into(), source), ("result".into(), Source::from_value(&json!(result))),
        ])));
    }
    entry
}
fn nodes(list:&[ProcessorNode],doc:&mut SourceDocument,cx:&PipelineContext,trace:&mut Option<Vec<Source>>)->PResult {
    for n in list {
        let timer=cx.count_stats.get().then(||n.stats.pre());
        let condition_result = n.cond.as_ref().map(|s|condition(s,doc)).transpose();
        let condition_passed = condition_result.as_ref().ok().copied().flatten();
        let result=match condition_result {
            Ok(Some(false))=>{
                if let Some(t)=timer{n.stats.post(t,false);}
                if let Some(trace) = trace.as_mut() {
                    let mut entry = source_trace_base(n, Some(false));
                    entry.insert("status".into(), Source::from_value(&json!("skipped")));
                    trace.push(Source::Object(entry));
                }
                continue;
            },
            Err(e)=>Err(e.into()),
            _=>process(n,doc,cx,trace),
        };
        if let Some(t)=timer{n.stats.post(t,matches!(&result,Err(ProcessorError::Failed(..))));}
        if let Some(trace)=trace.as_mut(){
            let mut entry=source_trace_base(n, condition_passed);
            match &result {
                Ok(())|Err(ProcessorError::Terminated)=>{entry.insert("doc".into(),doc.output());},
                Err(ProcessorError::Failed(e,_)) if n.ignore_failure=>{
                    entry.insert("status".into(),Source::from_value(&json!("error_ignored")));
                    entry.insert("ignored_error".into(),Source::from_value(&json!({"error":e.error_object(false)})));
                    entry.insert("doc".into(),doc.output());
                },
                Err(ProcessorError::Failed(e,_))=>{
                    entry.insert("status".into(),Source::from_value(&json!("error")));
                    entry.insert("error".into(),Source::from_value(&e.error_object(false)));
                },
                Err(ProcessorError::Dropped)=>{entry.insert("status".into(),Source::from_value(&json!("dropped")));}
            }
            trace.push(Source::Object(entry));
        }
        match result {
            Err(ProcessorError::Failed(e,origin)) if n.ignore_failure=>{let _=(e,origin);},
            Err(ProcessorError::Failed(e,origin)) if !n.on_failure.is_empty()=>{
                let origin=origin.unwrap_or_else(||(n.kind.clone(),n.tag.clone(),cx.current_pipeline()));
                failure_nodes(&n.on_failure,doc,cx,trace,&e,&origin)?;
            },
            Err(ProcessorError::Failed(e,origin))=>return Err(ProcessorError::Failed(e,origin.or_else(||Some((n.kind.clone(),n.tag.clone(),cx.current_pipeline()))))),
            other=>other?,
        }
        if doc.rerouted{return Err(ProcessorError::Terminated);}
    }Ok(())
}
fn failure_nodes(list:&[ProcessorNode],doc:&mut SourceDocument,cx:&PipelineContext,trace:&mut Option<Vec<Source>>,error:&EsError,origin:&(String,Option<String>,String))->PResult {
    let keys=["on_failure_message","on_failure_processor_type","on_failure_processor_tag","on_failure_pipeline"];
    let previous:Vec<_>=keys.iter().map(|key|doc.ingest.get(key).cloned()).collect();
    let values=[error.reason.clone(),origin.0.clone(),origin.1.clone().unwrap_or_default(),origin.2.clone()];
    for (key,value) in keys.iter().zip(values){doc.set(&format!("_ingest.{key}"),Source::from_value(&json!(value)))?;}
    *cx.fail_stack.borrow_mut()=None;
    let result=nodes(list,doc,cx,trace);
    for (key,value) in keys.iter().zip(previous){let path=format!("_ingest.{key}");if let Some(value)=value{doc.set(&path,value)?;}else{let _=doc.remove(&path);}}
    result
}
pub(super) fn run(p:&Pipeline,doc:&mut SourceDocument,cx:&PipelineContext,trace:&mut Option<Vec<Source>>)->PResult {
    if cx.stack.borrow().contains(&p.id){return Err(EsError::new(500,"graph_structure_exception",format!("Cycle detected for pipeline: {}",p.id)).into());}
    if cx.stack.borrow().len()>=100{return Err(EsError::illegal_state("Too many nested pipelines. Cannot have more than 100 nested pipelines").into());}
    let timer=cx.count_stats.get().then(||p.stats.pre());
    let previous=doc.ingest.get("pipeline").cloned();doc.set("_ingest.pipeline",Source::from_value(&json!(p.id)))?;
    cx.stack.borrow_mut().push(p.id.clone());cx.executed.borrow_mut().push(p.id.clone());
    let previous_pipeline=INGEST_PIPELINE.with(|value|value.replace(p.id.clone()));
    let result=match nodes(&p.processors,doc,cx,trace) {
        Err(ProcessorError::Failed(error,origin)) if !p.on_failure.is_empty()=>failure_nodes(&p.on_failure,doc,cx,trace,&error,&origin.unwrap_or_else(||("pipeline".into(),None,p.id.clone()))),
        Err(ProcessorError::Terminated)=>Ok(()),
        other=>other,
    };
    if matches!(&result,Err(ProcessorError::Failed(..)))&&cx.fail_stack.borrow().is_none(){*cx.fail_stack.borrow_mut()=Some(cx.stack.borrow().clone());}
    if let Some(previous)=previous{doc.set("_ingest.pipeline",previous)?;}else{let _=doc.remove("_ingest.pipeline");}
    cx.stack.borrow_mut().pop();INGEST_PIPELINE.with(|value|*value.borrow_mut()=previous_pipeline);
    if let Some(timer)=timer{p.stats.post(timer,matches!(&result,Err(ProcessorError::Failed(..))));}
    result
}

impl Node {
    pub fn simulate_pipeline_source(&self,id:Option<&str>,body:&Source,verbose:bool)->Result<Source> {
        let _scope=crate::enrich::NodeScope::enter(self);let store=self.ingest();
        let pipeline=match id {Some(id)=>store.pipeline(id).ok_or_else(||EsError::illegal_argument(format!("pipeline [{id}] does not exist")))?,None=>Arc::new(compile_pipeline_source("_simulate_pipeline",body.get("pipeline").ok_or_else(||EsError::illegal_argument("[pipeline] required property is missing"))?)?)};
        // ConfigurationUtils.readList distinguishes a missing/null property
        // from an explicitly empty list before SimulatePipelineRequest checks it.
        let missing_docs = || EsError::new(400, "parse_exception", "[docs] required property is missing").with("property_name", "docs");
        let docs = match body.get("docs") {
            None => return Err(missing_docs()),
            Some(Source::Scalar(raw)) if raw.get() == "null" => return Err(missing_docs()),
            Some(Source::Array(docs)) => docs,
            Some(_) => return Err(EsError::new(400, "parse_exception", "[docs] property isn't a list").with("property_name", "docs")),
        };
        if docs.is_empty() {
            return Err(EsError::illegal_argument("must specify at least one document in [docs]"));
        }
        let mut output=Vec::new();
        for input in docs {
            let source=input.get("_source").cloned().ok_or_else(||EsError::illegal_argument("[_source] required property is missing"))?;
            let index=input.get("_index").map(text).transpose()?.unwrap_or("_index".into());let id=input.get("_id").map(text).transpose()?.unwrap_or("_id".into());
            let mut doc=SourceDocument::new(source,&index,Some(&id),None,now_timestamp())?;
            doc.set("_version",input.get("_version").cloned().unwrap_or_else(||Source::from_value(&json!(-3))))?;
            let cx=PipelineContext::new(&store);let mut trace=verbose.then(Vec::new);
            let result=run(&pipeline,&mut doc,&cx,&mut trace);
            let item=if let Some(trace)=trace{Source::Object(BTreeMap::from([("processor_results".into(),Source::Array(trace))]))}else{match result{Ok(())|Err(ProcessorError::Terminated)=>Source::Object(BTreeMap::from([("doc".into(),doc.output())])),Err(ProcessorError::Dropped)=>Source::from_value(&Value::Null),Err(ProcessorError::Failed(e,_))=>Source::Object(BTreeMap::from([("error".into(),Source::from_value(&e.error_object(false)))]))}};
            output.push(item);
        }
        Ok(Source::Object(BTreeMap::from([("docs".into(),Source::Array(output))])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The native `set` with `copy_from` follows ES (and the legacy `SetP`):
    /// with `ignore_empty_value` a missing or empty source sets nothing (the
    /// apm-data `logs-apm@pipeline` step that sent every apm log to the
    /// failure store); without it a missing source fails; a present one is
    /// copied, and `override: false` keeps an existing target.
    #[test]
    fn set_copy_from_missing_source_with_ignore_empty_value_sets_nothing() {
        let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let run_one=|processor:Value,source:Value|->Result<Value>{
            let pipeline=compile_pipeline_source("set",&Source::from_value(&json!({"processors":[processor]}))).unwrap();
            let mut doc=SourceDocument::new(Source::from_value(&source),"logs-apm.error-testing",None,None,"now".into()).unwrap();
            run(&pipeline,&mut doc,&cx,&mut None).map_err(|e|EsError::illegal_argument(format!("{e:?}")))?;
            Ok(serde_json::from_str(&doc.source_value().text()).unwrap())
        };
        let apm=json!({"set":{"field":"event.dataset","copy_from":"data_stream.dataset","ignore_empty_value":true,"override":false}});
        assert_eq!(run_one(apm.clone(),json!({"@timestamp":"2017-06-22"})).unwrap(),json!({"@timestamp":"2017-06-22"}));
        assert_eq!(run_one(apm.clone(),json!({"data_stream":{"dataset":""}})).unwrap(),json!({"data_stream":{"dataset":""}}));
        assert_eq!(run_one(apm.clone(),json!({"data_stream":{"dataset":"apm.error"}})).unwrap(),json!({"data_stream":{"dataset":"apm.error"},"event":{"dataset":"apm.error"}}));
        assert_eq!(run_one(apm,json!({"data_stream":{"dataset":"apm.error"},"event":{"dataset":"kept"}})).unwrap(),json!({"data_stream":{"dataset":"apm.error"},"event":{"dataset":"kept"}}));
        assert!(run_one(json!({"set":{"field":"event.dataset","copy_from":"data_stream.dataset"}}),json!({"@timestamp":"2017-06-22"})).is_err());
    }
    #[test]
    fn unicode_domain_processors_preserve_unrelated_raw_values() {
        for (processor,source)in [
            (json!({"registered_domain":{"field":"domain_input","target_field":"parsed"}}),json!({"domain_input":"www.example.com"})),
            (json!({"network_direction":{"internal_networks":["private"]}}),json!({"source":{"ip":"192.168.1.2"},"destination":{"ip":"8.8.8.8"}})),
            (json!({"community_id":{}}),json!({"source":{"ip":"192.168.1.2","port":1234},"destination":{"ip":"8.8.8.8","port":53},"network":{"transport":"udp"}})),
            (json!({"attachment":{"field":"binary","remove_binary":true}}),json!({"binary":"aGVsbG8="})),
            (json!({"user_agent":{"field":"agent"}}),json!({"agent":"curl/8.0.1"})),
        ] {
            let pipeline=compile_pipeline_source("isolated",&Source::from_value(&json!({"processors":[processor]}))).unwrap();
            let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
            let mut normal=SourceDocument::new(Source::from_value(&source),"test",None,None,"now".into()).unwrap();let mut raw=normal.clone();
            raw.set("keep",Source::parse(r#""\ud800""#).unwrap()).unwrap();
            run(&pipeline,&mut normal,&cx,&mut None).unwrap();run(&pipeline,&mut raw,&cx,&mut None).unwrap();
            assert!(raw.remove("keep").unwrap().equivalent(&Source::parse(r#""\ud800""#).unwrap()));
            assert!(raw.source_value().equivalent(&normal.source_value()),"{processor}");
        }
    }
    #[test]
    fn uri_projection_retains_existing_unicode_parse_contract() {
        for input in ["https://user:pw@example.com:8443/a/b.php?x=1#frag","/rel/path?a=b","mailto:a@b.com","http://h:abc/","http://h/a b","bad uri","http://h.com/a.b/c","http://[fe80::1]/","https://example.com/a%20b?x=a+b","//registry_name/path","https://user%3Aname:p%40ss@host/","http://host/%ED%A0%80"] {
            let expected=super::super::net::uri_parts_map(input,true).map(Value::Object);
            let actual=super::super::source_uri::parts(&Source::from_value(&json!(input)),true).ok();
            assert_eq!(actual.is_some(),expected.is_some(),"{input}");
            if let(Some(actual),Some(expected))=(actual,expected){assert!(actual.equivalent(&Source::from_value(&expected)),"{input}: {} != {expected}",actual.text());}
        }
    }
    #[test]
    fn reroute_preserves_unrelated_raw_source_and_template_dependencies() {
        let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        for config in [json!({"destination":"other-{{target}}"}),json!({"dataset":"{{target}}"})] {
            let pipeline=compile_pipeline_source("reroute",&Source::from_value(&json!({"processors":[{"reroute":config}]}))).unwrap();
            let mut normal=SourceDocument::new(Source::parse(r#"{"target":"custom","event":{"dataset":"old"}}"#).unwrap(),"logs-old-default",None,None,"now".into()).unwrap();
            let mut raw=normal.clone();raw.set("keep",Source::parse(r#""\ud800""#).unwrap()).unwrap();
            run(&pipeline,&mut normal,&cx,&mut None).unwrap();run(&pipeline,&mut raw,&cx,&mut None).unwrap();
            assert!(raw.remove("keep").unwrap().equivalent(&Source::parse(r#""\ud800""#).unwrap()));
            assert!(raw.ctx.equivalent(&normal.ctx));assert!(raw.rerouted);
        }
    }
    #[test]
    fn fingerprint_hashes_java_utf8_without_mutating_source() {
        let pipeline=compile_pipeline_source("fingerprint",&Source::parse(r#"{"processors":[{"fingerprint":{"fields":["text"],"method":"SHA-1"}}]}"#).unwrap()).unwrap();
        let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut raw=SourceDocument::new(Source::parse(r#"{"text":["\ud800",{"value":"\udfff"}],"keep":"\ud800"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        let mut normal=SourceDocument::new(Source::parse(r#"{"text":["?",{"value":"?"}],"keep":"normal"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        let original=raw.source_value();run(&pipeline,&mut raw,&cx,&mut None).unwrap();run(&pipeline,&mut normal,&cx,&mut None).unwrap();
        assert!(raw.require("fingerprint").unwrap().equivalent(normal.require("fingerprint").unwrap()));
        assert!(raw.require("text").unwrap().equivalent(original.get("text").unwrap()));
        assert!(raw.require("keep").unwrap().equivalent(original.get("keep").unwrap()));
    }
    #[test]
    fn redaction_preserves_nonmatch_and_uses_java_utf8_when_rebuilding() {
        let pipeline=compile_pipeline_source("redact",&Source::parse(r#"{"processors":[{"redact":{"field":"text","patterns":["%{WORD:word}"],"trace_redact":true}}]}"#).unwrap()).unwrap();
        let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"text":"\ud800","keep":"\udfff"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert!(doc.require("text").unwrap().equivalent(&Source::parse(r#""\ud800""#).unwrap()));
        assert!(doc.get("_ingest._redact._is_redacted").unwrap().is_none());
        doc.set("text",Source::parse(r#""\ud800 secret""#).unwrap()).unwrap();run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert_eq!(doc.require("text").unwrap().text(),r#""? <word>""#);
        assert_eq!(doc.require("_ingest._redact._is_redacted").unwrap().text(),"true");
        assert!(doc.require("keep").unwrap().equivalent(&Source::parse(r#""\udfff""#).unwrap()));
    }
    #[test]
    fn date_templates_read_full_context_without_decoding_unrelated_values() {
        let pipeline=compile_pipeline_source("date",&Source::parse(r#"{"processors":[{"date":{"field":"date","formats":["yyyy-MM-dd"],"timezone":"{{zone}}"}}]}"#).unwrap()).unwrap();
        let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"date":"2020-01-02","zone":"UTC","keep":"\ud800"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert_eq!(doc.require("@timestamp").unwrap().text(),r#""2020-01-02T00:00:00.000Z""#);
        assert!(doc.require("keep").unwrap().equivalent(&Source::parse(r#""\ud800""#).unwrap()));
    }
    #[test]
    fn grok_capture_crosses_utf8_boundary_but_source_does_not() {
        let pipeline=compile_pipeline_source("grok",&Source::parse(r#"{"processors":[{"grok":{"field":"text","patterns":["%{GREEDYDATA:captured}"],"trace_match":true}}]}"#).unwrap()).unwrap();
        let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"text":"a\ud800b","keep":"\udfff"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert_eq!(doc.require("captured").unwrap().text(),r#""a?b""#);
        assert!(doc.require("text").unwrap().equivalent(&Source::parse(r#""a\ud800b""#).unwrap()));
        assert_eq!(doc.require("_ingest._grok_match_index").unwrap().text(),r#""0""#);
    }
    #[test]
    fn html_and_dot_expansion_preserve_raw_text_and_nested_values() {
        let pipeline=compile_pipeline_source("native",&Source::parse(r#"{"processors":[{"html_strip":{"field":"html"}},{"dot_expander":{"field":"*","path":"nested"}}]}"#).unwrap()).unwrap();
        let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"html":"<p>\ud800 &amp; x</p><b>\udfff</b>","nested":{"a.b":"\ud800","a":{"b":"\udfff"}},"keep":"\ud800"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert!(doc.require("html").unwrap().equivalent(&Source::parse(r#""\n\ud800 & x\n\udfff""#).unwrap()));
        assert!(doc.require("nested.a.b").unwrap().equivalent(&Source::parse(r#"["\udfff","\ud800"]"#).unwrap()));
        assert!(doc.require("keep").unwrap().equivalent(&Source::parse(r#""\ud800""#).unwrap()));
    }

    #[test]
    fn dissect_matches_utf8_boundary_without_mutating_source() {
        let raw=Source::parse(r#"{"processors":[{"dissect":{"field":"text","pattern":"%{a} %{+a} %{*key}=%{&key}","append_separator":":"}}]}"#).unwrap();
        let pipeline=compile_pipeline_source("dissect",&raw).unwrap();let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"text":"\ud800 \udfff dynamic=\ud800"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert!(doc.require("a").unwrap().equivalent(&Source::parse(r#""?:?""#).unwrap()));
        assert!(doc.require("dynamic").unwrap().equivalent(&Source::parse(r#""?""#).unwrap()));
    }
    #[test]
    fn key_value_processor_preserves_values_and_repeated_keys() {
        let raw=Source::parse(r#"{"processors":[{"kv":{"field":"text","field_split":";","value_split":"=","target_field":"out","strip_brackets":true,"trim_value":" "}}]}"#).unwrap();
        let pipeline=compile_pipeline_source("kv",&raw).unwrap();let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"text":"a=(\ud800);a=\udfff;b= x "}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert!(doc.require("out").unwrap().equivalent(&Source::parse(r#"{"a":["\ud800","\udfff"],"b":"x"}"#).unwrap()));
    }
    #[test]
    fn numeric_processors_do_not_decode_unrelated_source() {
        let raw=Source::parse(r#"{"processors":[{"convert":{"field":"n","type":"integer"}},{"bytes":{"field":"size"}},{"sort":{"field":"numbers"}},{"convert":{"field":"values","type":"auto"}}]}"#).unwrap();
        let pipeline=compile_pipeline_source("numeric",&raw).unwrap();let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"n":"12","size":"2kb","numbers":[3,1,2],"values":["12","\ud800"],"untouched":{"s":"\udfff"}}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        let expected=Source::parse(r#"{"n":12,"size":2048,"numbers":[1,2,3],"values":[12,"\ud800"],"untouched":{"s":"\udfff"}}"#).unwrap();
        assert!(doc.source_value().equivalent(&expected));
    }
    #[test]
    fn url_decode_preserves_unescaped_units_and_java_byte_replacement() {
        let pipeline=compile_pipeline_source("url",&Source::parse(r#"{"processors":[{"urldecode":{"field":"s"}}]}"#).unwrap()).unwrap();
        let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"s":["\ud800+%F0%9F%98%80","%ED%A0%80\udfff"],"other":"\ud800"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert!(doc.require("s").unwrap().equivalent(&Source::parse(r#"["\ud800 \ud83d\ude00","\ufffd\udfff"]"#).unwrap()));
    }
    #[test]
    fn csv_preserves_units_empty_values_and_partial_failure() {
        let raw=Source::parse(r#"{"processors":[{"csv":{"field":"csv","target_fields":["a","b","c"],"empty_value":"\udfff"}}]}"#).unwrap();
        let pipeline=compile_pipeline_source("csv",&raw).unwrap();let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"csv":"\"\ud800,x\",,z","other":"\ud800"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert!(doc.require("a").unwrap().equivalent(&Source::parse(r#""\ud800,x""#).unwrap()));
        assert!(doc.require("b").unwrap().equivalent(&Source::parse(r#""\udfff""#).unwrap()));
        doc.set("csv",Source::parse(r#""\ud800,\"unclosed""#).unwrap()).unwrap();
        assert!(run(&pipeline,&mut doc,&cx,&mut None).is_err());
        assert!(doc.require("a").unwrap().equivalent(&Source::parse(r#""\ud800""#).unwrap()));
    }
    #[test]
    fn raw_regex_configuration_and_keep_do_not_replace_units() {
        let raw=Source::parse(r#"{"processors":[{"gsub":{"field":"s","pattern":"\ud800","replacement":"\udfff"}},{"remove":{"keep":["s"]}}]}"#).unwrap();
        let pipeline=compile_pipeline_source("raw-regex",&raw).unwrap();let store=IngestService::in_memory();let cx=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"s":"a\ud800","removed":"\ud800"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&cx,&mut None).unwrap();
        assert!(doc.source_value().equivalent(&Source::parse(r#"{"s":"a\udfff"}"#).unwrap()));
        assert_eq!(doc.meta_str("_index").as_deref(),Some("test"));
    }
    #[test]
    fn templates_and_object_foreach_preserve_units() {
        let raw=Source::parse(r#"{"processors":[{"set":{"field":"{{name}}","value":"{{{s}}}","media_type":"text/plain"}},{"foreach":{"field":"map","processor":{"set":{"field":"_ingest._value","value":"{{{_ingest._value}}}x","media_type":"text/plain"}}}}]}"#).unwrap();
        let pipeline=compile_pipeline_source("templates",&raw).unwrap();let store=IngestService::in_memory();let context=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"name":"output","s":"\ud800","map":{"a":"\ud800"}}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&context,&mut None).unwrap();
        assert!(doc.require("output").unwrap().equivalent(&Source::parse(r#""\ud800""#).unwrap()));
        assert!(doc.require("map.a").unwrap().equivalent(&Source::parse(r#""\ud800x""#).unwrap()));
    }
    #[test]
    fn split_join_sort_and_replacement_keep_units() {
        let raw=Source::parse(r#"{"processors":[{"split":{"field":"s","separator":",","target_field":"a"}},{"sort":{"field":"a"}},{"join":{"field":"a","separator":":"}},{"gsub":{"field":"a","pattern":"(.)","replacement":"$1x"}}]}"#).unwrap();
        let pipeline=compile_pipeline_source("strings",&raw).unwrap();
        let store=IngestService::in_memory();let context=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"s":"\udfff,\ud800"}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&context,&mut None).unwrap();
        assert!(doc.require("a").unwrap().equivalent(&Source::parse(r#""\ud800x:x\udfffx""#).unwrap()));
    }
    #[test]
    fn scripts_and_structural_processors_keep_raw_units() {
        let raw=Source::parse(r#"{"processors":[{"script":{"source":"ctx.generated=params.s; ctx.upper=ctx.original","params":{"s":"\udfff"}}},{"uppercase":{"field":"upper"}},{"set":{"field":"copy","copy_from":"generated"}},{"rename":{"field":"copy","target_field":"renamed"}},{"append":{"field":"array","value":"\ud800","allow_duplicates":false}}]}"#).unwrap();
        let pipeline=compile_pipeline_source("lossless",&raw).unwrap();
        let store=IngestService::in_memory();let context=PipelineContext::new(&store);
        let mut doc=SourceDocument::new(Source::parse(r#"{"original":"a\ud800","array":["\ud800"]}"#).unwrap(),"test",None,None,"now".into()).unwrap();
        run(&pipeline,&mut doc,&context,&mut None).unwrap();
        assert!(doc.require("upper").unwrap().equivalent(&Source::parse(r#""A\ud800""#).unwrap()));
        assert!(doc.require("renamed").unwrap().equivalent(&Source::parse(r#""\udfff""#).unwrap()));
        assert!(doc.require("array").unwrap().equivalent(&Source::parse(r#"["\ud800"]"#).unwrap()));
        assert!(doc.get("copy").unwrap().is_none());
    }
}

/// A pipeline run's outcome; a document carries the first pipeline that
/// counts its produced bytes (stats), counted where the source is written.
enum SourceExec {Dropped(String),Doc(SourceDocument,Option<Arc<Pipeline>>)}
impl Node {
    fn execute_source_request_ix(&self, r: &IngestRequest, cx: &PipelineContext, pre: Option<SourceDocument>) -> std::result::Result<Option<SourceExec>, (EsError, String)> {
        let (mut def, mut fin) = self.index_pipelines(r.index);
        let mut current: Option<String> = match r.pipeline {
            Some("_none") => None,
            Some(p) => Some(p.to_string()),
            None => def.take(),
        };
        if current.is_none() && fin.is_none() && pre.is_none() {
            return Ok(None);
        }
        let mut doc = match pre {
            Some(d) => d,
            None => SourceDocument::new(Source::parse_leading(r.source).map_err(|e|(EsError::from(e),r.index.to_string()))?,r.index,r.id,r.routing,now_timestamp()).map_err(|e|(e,r.index.to_string()))?,
        };
        let mut seen = vec![r.index.to_string()];
        let mut first: Option<Arc<Pipeline>> = None;
        loop {
            if let Some(pid) = current.take() {
                let p = cx.pipeline(&pid).ok_or_else(|| (EsError::illegal_argument(format!("pipeline with id [{pid}] does not exist")), doc.meta_str("_index").unwrap_or_default()))?;
                if cx.count_stats.get() && first.is_none() {
                    p.ingested_bytes.fetch_add(r.source.len() as u64, std::sync::atomic::Ordering::Relaxed);
                    first = Some(p.clone());
                }
                match run(&p, &mut doc, cx,&mut None) {
                    Ok(()) => {}
                    Err(ProcessorError::Dropped) => return Ok(Some(SourceExec::Dropped(doc.meta_str("_index").unwrap_or_default()))),
                    Err(ProcessorError::Terminated) => {}
                    Err(ProcessorError::Failed(e, origin)) => {
                        *cx.fail_origin.borrow_mut() = origin;
                        return Err((e, doc.meta_str("_index").unwrap_or_default()));
                    }
                }
            }
            let now_index = doc.meta_str("_index").unwrap_or_default();
            if doc.rerouted {
                doc.rerouted = false;
                if now_index != *seen.last().unwrap() {
                    if seen.contains(&now_index) {
                        seen.push(now_index.clone());
                        return Err((EsError::new(500, "illegal_state_exception", format!("index cycle detected while processing pipeline [{}] for document [{}]: [{}]", cx.executed.borrow().last().cloned().unwrap_or_default(), r.id.unwrap_or(""), seen.join(", "))), now_index.clone()));
                    }
                    seen.push(now_index.clone());
                    *cx.fs_index.borrow_mut() = Some(now_index.clone());
                    let (d2, f2) = self.index_pipelines(&now_index);
                    fin = f2;
                    current = d2;
                    continue;
                }
            } else if now_index != *seen.last().unwrap() {
                // a plain `_index` change keeps the pipelines but picks up the new index's final pipeline
                fin = self.index_pipelines(&now_index).1;
            }
            break;
        }
        if let Some(fid) = fin {
            let p = cx.pipeline(&fid).ok_or_else(|| (EsError::illegal_argument(format!("pipeline with id [{fid}] does not exist")), doc.meta_str("_index").unwrap_or_default()))?;
            let before = doc.meta_str("_index");
            match run(&p, &mut doc, cx,&mut None) {
                Ok(()) | Err(ProcessorError::Terminated) => {}
                Err(ProcessorError::Dropped) => return Ok(Some(SourceExec::Dropped(doc.meta_str("_index").unwrap_or_default()))),
                Err(ProcessorError::Failed(e, origin)) => {
                    *cx.fail_origin.borrow_mut() = origin;
                    return Err((e, doc.meta_str("_index").unwrap_or_default()));
                }
            }
            let after = doc.meta_str("_index");
            if after != before {
                let msg = format!("final pipeline [{fid}] can't change the target index (from [{}] to [{}]) for document [{}]", before.unwrap_or_default(), after.clone().unwrap_or_default(), r.id.unwrap_or(""));
                return Err((EsError::new(500, "illegal_state_exception", msg), after.unwrap_or_default()));
            }
        }
        Ok(Some(SourceExec::Doc(doc, first)))
    }

}

impl Node {
    pub(super) fn run_ingest_source_ext(&self, r: IngestRequest) -> std::result::Result<IngestOutcome, IngestFailure> {
        let _enrich = crate::enrich::NodeScope::enter(self);
        if r.pipeline.is_none() && registry_is_empty(self) {
            return Ok(IngestOutcome::Unchanged);
        }
        let store = self.ingest();
        let cx = PipelineContext::new(&store);
        cx.count_stats.set(true);
        let t = store.total.pre();
        let res = self.execute_source_request_ix(&r, &cx, None).map_err(|(e,_)|e);
        store.total.post(t, res.is_err());
        let res = match res {
            Ok(res) => res,
            Err(error) => {
                let origin = cx.fail_origin.borrow_mut().take();
                let trace = cx.fail_stack.borrow_mut().take().filter(|s| !s.is_empty()).or_else(|| cx.executed.borrow().last().map(|p| vec![p.clone()])).unwrap_or_default();
                let fs_index = cx.fs_index.borrow_mut().take().unwrap_or_else(|| r.index.to_string());
                let context = crate::datastream::FailureContext {
                    pipeline: origin.as_ref().map(|o| o.2.clone()).filter(|p| !p.is_empty()).or_else(|| trace.last().cloned()),
                    processor_type: origin.as_ref().map(|o| o.0.clone()),
                    processor_tag: origin.as_ref().and_then(|o| o.1.clone()),
                    pipeline_trace: trace,
                };
                return Err(IngestFailure { error, index: fs_index, context });
            }
        };
        Ok(match res {
            None => IngestOutcome::Unchanged,
            Some(SourceExec::Dropped(index)) => IngestOutcome::Dropped { index },
            Some(SourceExec::Doc(d, first)) => {
                let (src, meta) = d.split_meta();
                let text = serde_json::to_string(&src).map_err(|e| IngestFailure { error: e.into(), index: r.index.to_string(), context: Default::default() })?;
                // The first pipeline's produced bytes: the source written
                // (counted for a document its metadata then fails, too).
                if let Some(p) = first {
                    p.produced_bytes.fetch_add(text.len() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                let meta = meta.map_err(|error|IngestFailure{error,index:r.index.to_string(),context:Default::default()})?;
                let index = meta.get("_index").map(doc::value_to_string).unwrap_or_else(|| r.index.to_string());
                let index = processors::resolve_index_expression(&index).map_err(|error| IngestFailure { error, index: r.index.to_string(), context: Default::default() })?;
                IngestOutcome::Changed {
                    index,
                    id: meta.get("_id").filter(|v| !v.is_null()).map(doc::value_to_string),
                    routing: meta.get("_routing").filter(|v| !v.is_null()).map(doc::value_to_string),
                    dynamic_templates: meta
                        .get("_dynamic_templates")
                        .and_then(Value::as_object)
                        .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect()),
                    source: text,
                }
            }
        })
    }

}

impl Node {
    pub fn simulate_ingest_source(&self,index:Option<&str>,pipeline:Option<&str>,body:&Source)->Result<Source> {
        let _scope=crate::enrich::NodeScope::enter(self);let store=self.ingest();let mut substitutions=HashMap::new();
        if let Some(configs)=body.get("pipeline_substitutions") {
            let Source::Object(configs)=configs else{return Err(EsError::new(400,"parse_exception","[pipeline_substitutions] must be an object"))};
            for (id,config) in configs {validate_no_system_properties(&super::source_config::metadata(config)?)?;substitutions.insert(id.clone(),Arc::new(compile_pipeline_source(id,config)?));}
        }
        let Some(Source::Array(docs))=body.get("docs") else{return Err(EsError::action_validation(&["docs is missing"]))};
        let mut output=Vec::new();
        for input in docs {
            let target=input.get("_index").map(text).transpose()?.or_else(||index.map(str::to_string)).unwrap_or_default();
            let id=input.get("_id").map(text).transpose()?;let routing=input.get("_routing").map(text).transpose()?;
            let source=input.get("_source").cloned().ok_or_else(||EsError::illegal_argument("[_source] required property is missing"))?;
            let document=SourceDocument::new(source,&target,id.as_deref(),routing.as_deref(),now_timestamp())?;
            let context=PipelineContext{substitutions:Some(&substitutions),..PipelineContext::new(&store)};
            let request=IngestRequest{index:&target,id:id.as_deref(),routing:routing.as_deref(),source:"",pipeline};
            let document=match self.execute_source_request_ix(&request,&context,Some(document)) {
                Ok(Some(SourceExec::Doc(doc,_)))=>doc,
                Ok(Some(SourceExec::Dropped(index)))=>{output.push(Source::from_value(&json!({"doc":{"_index":index,"_id":id.clone().unwrap_or_else(||"auto-generated".into()),"_version":-3,"result":"noop","_shards":{"total":0,"successful":0,"failed":0}}})));continue;},
                Err((error,index))=>{output.push(Source::from_value(&json!({"doc":{"_index":index,"_id":id.clone().unwrap_or_else(||"auto-generated".into()),"error":error_body(&error)}})));continue;},
                Ok(None)=>continue,
            };
            let target=document.meta_str("_index").unwrap_or(target);let source=document.source_value();
            let mut entry=BTreeMap::from([
                ("_source".into(),source.clone()),("_id".into(),Source::from_value(&json!(document.meta_str("_id")))),
                ("_index".into(),Source::from_value(&json!(target))),("_version".into(),Source::from_value(&json!(-3))),
                ("executed_pipelines".into(),Source::from_value(&json!(context.executed.borrow().clone()))),
            ]);
            // This is the indexed UTF8 value boundary, not stored-source output.
            // The original source above remains lossless in the response.
            if let Err(error)=self.validate_against_mapping(&target,&crate::script::source_from_json(&source).to_json()) {entry.insert("error".into(),Source::from_value(&error_body(&error)));}
            output.push(Source::Object(BTreeMap::from([("doc".into(),Source::Object(entry))])));
        }
        Ok(Source::Object(BTreeMap::from([("docs".into(),Source::Array(output))])))
    }
}
