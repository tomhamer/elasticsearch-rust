//! URI components are Java strings, including non-scalar units in paths.
use crate::source_json::Source;
use crate::error::{EsError,Result};
use std::collections::BTreeMap;
fn source(units:&[u16])->Result<Source>{crate::script::source_to_json(&crate::script::Val::utf16(units.to_vec()))}
fn decoded(units:&[u16],fallback:bool)->Result<Source>{
    if fallback{return source(units);}
    // URI decoding leaves '+' alone; form decoding does not. Escape literal
    // plus signs before using the shared Java UTF8 percent-byte decoder.
    let mut escaped=Vec::new();for &u in units{if u==43{escaped.extend([37,50,66]);}else{escaped.push(u);}}
    crate::script::ingest_url_decode(&source(&escaped)?)
}
fn split(units:&[u16],delimiter:u16)->(&[u16],Option<&[u16]>){match units.iter().position(|u|*u==delimiter){Some(i)=>(&units[..i],Some(&units[i+1..])),None=>(units,None)}}
fn parse(units:&[u16],keep:bool,fallback:bool)->Result<Source>{
    let bad=||EsError::illegal_argument("unable to parse URI");
    if !fallback {
        for &u in units{if u<=32||u==127||[34,60,62,92,94,96,123,124,125].contains(&u)||char::from_u32(u as u32).is_some_and(|c|u>=128&&c.is_whitespace()){return Err(bad());}}
        let mut i=0;while i<units.len(){if units[i]==37{if i+2>=units.len()||!units[i+1..i+3].iter().all(|u|*u<=127&&(*u as u8).is_ascii_hexdigit()){return Err(bad());}i+=3;}else{i+=1;}}
    }
    let mut out=BTreeMap::from([("domain".into(),Source::from_value(&serde_json::Value::Null)),("scheme".into(),Source::from_value(&serde_json::Value::Null))]);
    if keep{out.insert("original".into(),source(units)?);}
    let(main,fragment)=split(units,35);if let Some(fragment)=fragment{if !fallback&&fragment.contains(&35){return Err(bad());}out.insert("fragment".into(),decoded(fragment,fallback)?);}
    let mut rest=main;let mut scheme=None;
    if let Some(colon)=main.iter().position(|u|*u==58){let before=&main[..colon];if !before.contains(&47)&&!before.contains(&63){
        let name=String::from_utf16(before).map_err(|_|bad())?;
        if name.is_empty()||!name.as_bytes()[0].is_ascii_alphabetic()||!name.bytes().all(|c|c.is_ascii_alphanumeric()||b"+-.".contains(&c)){return Err(bad());}
        scheme=Some(name);rest=&main[colon+1..];if rest.is_empty()&&!fallback{return Err(bad());}
    }}
    if fallback&&!scheme.as_ref().is_some_and(|s|matches!(s.to_ascii_lowercase().as_str(),"http"|"https"|"ftp"|"file"|"jar"|"jrt"|"mailto")){return Err(bad());}
    if let Some(name)=&scheme{out.insert("scheme".into(),Source::from_value(&serde_json::json!(if fallback{name.to_ascii_lowercase()}else{name.clone()})));}
    if !fallback&&scheme.is_some()&&!rest.starts_with(&[47]){return Ok(Source::Object(out));}
    let(hier,query)=split(rest,63);if let Some(query)=query{out.insert("query".into(),decoded(query,fallback)?);}
    let path=if let Some(after)=hier.strip_prefix(&[47,47]){
        let(auth,path)=match after.iter().position(|u|*u==47){Some(i)=>(&after[..i],&after[i..]),None=>(after,&[][..])};
        let split_at=if fallback{auth.iter().rposition(|u|*u==64)}else{auth.iter().position(|u|*u==64)};
        let (userinfo,hostport)=match split_at{Some(i)=>(Some(&auth[..i]),&auth[i+1..]),None=>(None,auth)};
        if fallback {
            let(host,port)=if hostport.starts_with(&[91]){let end=hostport.iter().position(|u|*u==93).ok_or_else(bad)?;(&hostport[..=end],hostport[end+1..].strip_prefix(&[58]))}
                else{match hostport.iter().rposition(|u|*u==58){Some(i)=>(&hostport[..i],Some(&hostport[i+1..])),None=>(hostport,None)}};
            out.insert("domain".into(),source(host)?);
            if let Some(port)=port.filter(|p|!p.is_empty()){let port=String::from_utf16(port).map_err(|_|bad())?.parse::<i64>().map_err(|_|bad())?;out.insert("port".into(),Source::from_value(&serde_json::json!(port)));}
            if let Some(ui)=userinfo{put_user(&mut out,decoded(ui,true)?)?;}
        }else if !auth.is_empty(){
            if let Ok(hostport)=String::from_utf16(hostport){if let Some((_,host,port))=super::net::server_authority(&hostport){
                out.insert("domain".into(),Source::from_value(&serde_json::json!(host)));if port!=-1{out.insert("port".into(),Source::from_value(&serde_json::json!(port)));}
                if let Some(ui)=userinfo{put_user(&mut out,decoded(ui,false)?)?;}
            }}
            if auth.iter().any(|u|*u<128&&!(*u as u8).is_ascii_alphanumeric()&&!b"-_.!~*'()%$,;:@&=+[]".contains(&(*u as u8))){return Err(bad());}
        }
        path
    }else{if fallback{out.insert("domain".into(),source(&[])?);}hier};
    let path=decoded(path,fallback)?;let text=crate::script::source_from_json(&path).string_units();
    if let Some(dot)=text.iter().rposition(|u|*u==46){if text.iter().rposition(|u|*u==47).map(|slash|dot>slash).unwrap_or(true){out.insert("extension".into(),source(&text[dot+1..])?);}}
    out.insert("path".into(),path);Ok(Source::Object(out))
}
fn put_user(out:&mut BTreeMap<String,Source>,user:Source)->Result<()>{
    let units=crate::script::source_from_json(&user).string_units();let(name,password)=split(&units,58);
    out.insert("username".into(),source(name)?);if let Some(password)=password{out.insert("password".into(),source(password)?);}out.insert("user_info".into(),user);Ok(())
}
pub(super) fn parts(input:&Source,keep:bool)->Result<Source>{
    let value=crate::script::source_from_json(input);if !matches!(&value,crate::script::Val::Str(_)|crate::script::Val::Utf16(_)){return Err(EsError::illegal_argument("URI field must be a string"));}
    let units=value.string_units();parse(&units,keep,false).or_else(|_|parse(&units,keep,true))
}
#[cfg(test)]mod tests{use super::*;
    #[test]fn raw_uri_components_preserve_units_and_decode_percent_bytes(){
        let value=Source::parse(r#""https://user:\ud800@example.com/a.\udfff?x=\ud800+%E2%82%AC#\udfff""#).unwrap();
        let parsed=parts(&value,true).unwrap();
        assert!(parsed.get("original").unwrap().equivalent(&value));
        for(key,expected)in [("path",r#""/a.\udfff""#),("extension",r#""\udfff""#),("password",r#""\ud800""#),("query",r#""x=\ud800+€""#),("fragment",r#""\udfff""#)]{assert!(parsed.get(key).unwrap().equivalent(&Source::parse(expected).unwrap()),"{key}");}
    }
}
