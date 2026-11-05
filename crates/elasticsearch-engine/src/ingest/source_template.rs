//! Mustache rendering over Java UTF-16 values, without Unicode sentinels.
use super::{source_doc::SourceDocument,jtext::MediaType};
use crate::{source_json::Source,error::{EsError,Result},script::{source_from_json,source_to_json,Val}};
use std::collections::BTreeMap;

enum Token {Text(Vec<u16>),Var(String,bool),Section(String,bool,Vec<Token>)}
fn strict(units:&[u16])->Result<String>{String::from_utf16(units).map_err(|_|EsError::illegal_argument("template field names must be valid Unicode"))}
fn parse(units:&[u16])->Result<Vec<Token>> {
    let mut stack=vec![(String::new(),false,Vec::new())];let mut pos=0;
    while pos<units.len() {
        let Some(start)=units[pos..].windows(2).position(|w|w==[123,123]).map(|i|i+pos) else{stack.last_mut().unwrap().2.push(Token::Text(units[pos..].to_vec()));break};
        if start>pos{stack.last_mut().unwrap().2.push(Token::Text(units[pos..start].to_vec()));}
        let width=if units.get(start+2)==Some(&123){3}else{2};
        let Some(end)=units[start+width..].windows(width).position(|w|w.iter().all(|u|*u==125)).map(|i|i+start+width) else{stack.last_mut().unwrap().2.push(Token::Text(units[start..].to_vec()));break};
        let name=strict(&units[start+width..end])?;let name=name.trim();
        if width==3 {stack.last_mut().unwrap().2.push(Token::Var(name.into(),true));}
        else if let Some(n)=name.strip_prefix('#'){stack.push((n.trim().into(),false,Vec::new()));}
        else if let Some(n)=name.strip_prefix('^'){stack.push((n.trim().into(),true,Vec::new()));}
        else if name.starts_with('/') {if stack.len()>1{let(n,i,b)=stack.pop().unwrap();stack.last_mut().unwrap().2.push(Token::Section(n,i,b));}}
        else if !name.starts_with('!'){let(n,raw)=name.strip_prefix('&').map(|n|(n.trim(),true)).unwrap_or((name,false));stack.last_mut().unwrap().2.push(Token::Var(n.into(),raw));}
        pos=end+width;
    }
    while stack.len()>1{let(n,i,b)=stack.pop().unwrap();stack.last_mut().unwrap().2.push(Token::Section(n,i,b));}
    Ok(stack.pop().unwrap().2)
}
fn walk<'a>(value:&'a Source,path:&str)->Option<&'a Source>{
    if path.is_empty(){return Some(value)}
    let(head,tail)=path.split_once('.').unwrap_or((path,""));
    match value{Source::Object(m)=>m.get(head).and_then(|v|walk(v,tail)),Source::Array(a)=>head.parse::<usize>().ok().and_then(|i|a.get(i)).and_then(|v|walk(v,tail)),_=>None}
}
fn resolve(doc:&SourceDocument,name:&str,stack:&[Source])->Option<Source>{
    if matches!(name,"."|"this"){return stack.last().cloned()}
    let(head,tail)=name.split_once('.').unwrap_or((name,""));
    for ctx in stack.iter().rev(){if let Source::Object(m)=ctx{if let Some(v)=m.get(head){return walk(v,tail).cloned()}}}
    doc.get(name).ok().flatten().cloned()
}
fn stringify(value:&Source)->Vec<u16>{source_from_json(value).string_units()}
fn mustache(value:&Source)->Vec<u16>{
    match value {
        Source::Scalar(v) if v.get()=="null"=>Vec::new(),
        Source::Array(a)=>{let mut out=vec![123];for(i,v)in a.iter().enumerate(){if i>0{out.extend([44,32]);}out.extend(format!("{i}=").encode_utf16());out.extend(stringify(v));}out.push(125);out},
        _=>stringify(value),
    }
}
fn url(units:&[u16])->Vec<u16>{
    let mut bytes=Vec::new();for c in char::decode_utf16(units.iter().copied()){match c{Ok(c)=>{let mut b=[0;4];bytes.extend_from_slice(c.encode_utf8(&mut b).as_bytes());},Err(_)=>bytes.push(b'?')}}
    let mut out=String::new();for b in bytes{match b{b'A'..=b'Z'|b'a'..=b'z'|b'0'..=b'9'|b'.'|b'-'|b'*'|b'_'=>out.push(b as char),b' '=>out.push('+'),_=>out.push_str(&format!("%{b:02X}"))}}out.encode_utf16().collect()
}
fn escape(units:&[u16],media:MediaType)->Vec<u16>{
    match media{
        MediaType::Plain=>units.to_vec(),MediaType::Form=>url(units),
        MediaType::Json=>{let mut out=Vec::new();for &u in units{match u{34=>out.extend([92,34]),92=>out.extend([92,92]),8=>out.extend([92,98]),9=>out.extend([92,116]),10=>out.extend([92,110]),12=>out.extend([92,102]),13=>out.extend([92,114]),0..=31=>out.extend(format!("\\u{u:04x}").encode_utf16()),_=>out.push(u)}}out}
    }
}
fn render(doc:&SourceDocument,tokens:&[Token],stack:&mut Vec<Source>,media:MediaType,out:&mut Vec<u16>)->Result<()> {
    for token in tokens{match token{
        Token::Text(s)=>out.extend(s),
        Token::Var(name,raw)=>if let Some(value)=resolve(doc,name,stack){let units=mustache(&value);out.extend(if *raw{units}else{escape(&units,media)});},
        Token::Section(name,inverted,body)=>{
            let function=name.split_whitespace().next().unwrap_or("");
            if !inverted&&matches!(function,"toJson"|"join"|"url"){
                let units:Vec<_>=body.iter().filter_map(|t|if let Token::Text(s)=t{Some(s.as_slice())}else{None}).flatten().copied().collect();let path=strict(&units)?;let path=path.trim();let value=resolve(doc,path,stack);
                match function{
                    "toJson"=>if let Some(value)=value{out.extend(escape(&value.text().encode_utf16().collect::<Vec<_>>(),media));},
                    "join"=>if let Some(Source::Array(a))=value{let delimiter=name.split_once("delimiter=").map(|(_,d)|d.trim().trim_matches(|c|c=='\''||c=='"')).unwrap_or(",");let mut joined=Vec::new();for(i,v)in a.iter().enumerate(){if i>0{joined.extend(delimiter.encode_utf16());}joined.extend(stringify(v));}out.extend(escape(&joined,media));},
                    _=>out.extend(url(&value.as_ref().map(mustache).unwrap_or_else(||path.encode_utf16().collect()))),
                }continue;
            }
            let value=resolve(doc,name,stack);let truth=match &value{None=>false,Some(Source::Array(a))=>!a.is_empty(),Some(Source::Scalar(v))=>!matches!(v.get(),"null"|"false"|"\"\""),_=>true};
            if *inverted {if !truth{render(doc,body,stack,media,out)?;}}
            else if truth{match value{Some(Source::Array(a))=>for v in a{stack.push(v);render(doc,body,stack,media,out)?;stack.pop();},Some(v)if v.text()=="true"=>render(doc,body,stack,media,out)?,Some(v)=>{stack.push(v);render(doc,body,stack,media,out)?;stack.pop();},None=>{}}}
        }
    }}Ok(())
}
impl SourceDocument {
    pub fn render_source(&self,value:&Source,media:MediaType)->Result<Source>{
        match value{
            Source::Object(m)=>{let mut result=BTreeMap::new();for(k,v)in m{result.insert(self.render_path(k)?,self.render_source(v,media)?);}Ok(Source::Object(result))},
            Source::Array(a)=>Ok(Source::Array(a.iter().map(|v|self.render_source(v,media)).collect::<Result<_>>()?)),
            Source::Scalar(v)if v.get().starts_with('"')=>{let units=source_from_json(value).string_units();let tokens=parse(&units)?;let mut out=Vec::new();render(self,&tokens,&mut Vec::new(),media,&mut out)?;source_to_json(&Val::utf16(out))},
            _=>Ok(value.clone()),
        }
    }
    pub fn render_path(&self,path:&str)->Result<String>{let tokens=parse(&path.encode_utf16().collect::<Vec<_>>())?;let mut out=Vec::new();render(self,&tokens,&mut Vec::new(),MediaType::Json,&mut out)?;strict(&out)}
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn templates_keep_lone_values_in_sections_and_literals(){
        let doc=SourceDocument::new(Source::parse(r#"{"s":"\ud800","a":["\udfff","x"],"name":"out"}"#).unwrap(),"i",None,None,"now".into()).unwrap();
        let raw=Source::parse(r#"{"{{name}}":"\udfff{{s}}/{{#a}}{{.}}{{/a}}"}"#).unwrap();
        let out=doc.render_source(&raw,MediaType::Plain).unwrap();assert!(out.equivalent(&Source::parse(r#"{"out":"\udfff\ud800/\udfffx"}"#).unwrap()));
        assert!(doc.render_path("{{s}}").is_err());
    }
}
