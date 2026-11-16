//! Query-owned regex programs and explicit matcher scratch. No thread-local
//! cache outlives the request and no lazy DFA allocates outside admission.
use crate::{error::Result,memory::{breakers,Reservation}};
use regex_automata::{meta::Regex,Input};
use std::{collections::HashMap,sync::Arc};
const LABEL:&str="esql regex";
#[derive(Default)]
pub(crate) struct Cache {
    entries:HashMap<String,Option<Arc<Program>>>,
    metadata:Option<Reservation>,
}
impl Cache {
    pub(crate) fn get(&mut self,pattern:&str)->Result<Option<Arc<Program>>> {
        if let Some(re)=self.entries.get(pattern){return Ok(re.clone());}
        if self.entries.len()>=512 {self.entries.clear();self.metadata=None;}
        let memory=self.metadata.get_or_insert(breakers().reserve_request(0,LABEL)?);
        memory.grow((pattern.len() as u64).saturating_mul(2).saturating_add(256),LABEL)?;
        let compiled=Program::compile(pattern)?;
        self.entries.insert(pattern.to_string(),compiled.clone());
        Ok(compiled)
    }
}
pub(crate) struct Program { re:Regex, _memory:Reservation }
impl Program {
    pub(crate) fn compile(pattern:&str)->Result<Option<Arc<Self>>> {
        Ok(Self::compile_detailed(pattern)?.ok())
    }
    pub(crate) fn compile_detailed(pattern:&str)->Result<std::result::Result<Arc<Self>,String>> {
        let unicode_atoms=pattern.as_bytes().windows(2).filter(|p|p[0]==b'\\'&&matches!(p[1],b'p'|b'P'|b'w'|b'W'|b'd'|b'D'|b's'|b'S')).count() as u64;
        let parser=(pattern.len() as u64).saturating_mul(256).saturating_add(unicode_atoms.saturating_mul(8<<20)).saturating_add(8192);
        let mut budget=1024usize;
        loop {
            let _scratch=breakers().reserve_request(parser.saturating_add((budget as u64).saturating_mul(16)),LABEL)?;
            let config=Regex::config().nfa_size_limit(Some(budget)).hybrid(false).dfa(false).onepass(false).backtrack(false);
            match Regex::builder().configure(config).build(pattern) {
                Ok(re)=>{
                    let memory=breakers().reserve_request((re.memory_usage() as u64).saturating_mul(4).saturating_add(256),LABEL)?;
                    return Ok(Ok(Arc::new(Self{re,_memory:memory})));
                }
                Err(e) if e.size_limit().is_some() && budget<10*(1<<20)=>budget=(budget*2).min(10*(1<<20)),
                Err(error)=>return Ok(Err(error.syntax_error().map(ToString::to_string).unwrap_or_else(||error.to_string()))),
            }
        }
    }
    fn scratch(&self)->Result<Reservation> {
        breakers().reserve_request((self.re.memory_usage() as u64).saturating_mul(self.re.captures_len() as u64+1).saturating_mul(16).saturating_add(8192),LABEL)
    }
    pub(crate) fn is_match(&self,text:&str)->Result<bool> {
        let _scratch=self.scratch()?;
        let mut cache=self.re.create_cache();
        Ok(self.re.search_with(&mut cache,&Input::new(text)).is_some())
    }
    pub(crate) fn captures(&self,text:&str)->Result<Captures> {
        let memory=self.scratch()?;
        let mut cache=self.re.create_cache();
        let mut captures=self.re.create_captures();
        self.re.search_captures_with(&mut cache,&Input::new(text),&mut captures);
        Ok(Captures{captures,_memory:memory})
    }
    /// Java `Matcher.replaceAll(replacement)` (ES|QL `REPLACE`): `$n`
    /// (as many digits as name an existing group), `${name}` and `\x`
    /// escapes. `Err(exception)`: Java's exception for an invalid
    /// replacement (`java.lang.IllegalArgumentException: ..` /
    /// `java.lang.IndexOutOfBoundsException: No group n`).
    pub(crate) fn replace_java(&self,text:&str,replacement:&str)->Result<std::result::Result<String,String>> {
        let _scratch=self.scratch()?;
        let mut cache=self.re.create_cache();
        let mut caps=self.re.create_captures();
        let mut result=String::new();let mut end=0;
        let mut at=0usize;
        while at<=text.len() {
            self.re.search_captures_with(&mut cache,&Input::new(text).range(at..),&mut caps);
            let Some(m)=caps.get_match() else { break };
            result.push_str(&text[end..m.start()]);
            if let Err(e)=expand_java(&caps,text,replacement,&mut result) { return Ok(Err(e)); }
            end=m.end();
            // An empty match advances one character.
            at=if m.is_empty() { match text[m.end()..].chars().next() { Some(c)=>m.end()+c.len_utf8(), None=>break } } else { m.end() };
        }
        result.push_str(&text[end..]);Ok(Ok(result))
    }
    /// Literal replacement (every match replaced by `replacement` as is).
    /// Caller owns the output's pre-admitted scalar guard across handoff.
    pub(crate) fn replace(&self,text:&str,replacement:&str)->Result<String> {
        let _scratch=self.scratch()?;
        let mut cache=self.re.create_cache();
        let mut searcher=regex_automata::util::iter::Searcher::new(Input::new(text));
        let mut result=String::new();let mut end=0;
        while let Some(m)=searcher.advance(|input|Ok(self.re.search_with(&mut cache,input))) {
            result.push_str(&text[end..m.start()]);result.push_str(replacement);end=m.end();
        }
        result.push_str(&text[end..]);Ok(result)
    }
}

/// Append Java's expansion of `replacement` for one match (`caps`).
fn expand_java(caps:&regex_automata::util::captures::Captures,text:&str,replacement:&str,out:&mut String)->std::result::Result<(),String> {
    let group=|i:usize,out:&mut String| if let Some(sp)=caps.get_group(i) { out.push_str(&text[sp.start..sp.end]); };
    let groups=caps.group_len();
    let r:Vec<char>=replacement.chars().collect();
    let mut i=0;
    while i<r.len() {
        match r[i] {
            '\\'=>{
                i+=1;
                match r.get(i) { Some(c)=>{out.push(*c);i+=1;} None=>return Err("java.lang.IllegalArgumentException: character to be escaped is missing".into()) }
            }
            '$'=>{
                i+=1;
                match r.get(i) {
                    None=>return Err("java.lang.IllegalArgumentException: Illegal group reference: group index is missing".into()),
                    Some('{')=>{
                        i+=1;
                        let start=i;
                        while i<r.len() && r[i].is_ascii_alphanumeric() { i+=1; }
                        if i==start { return Err("java.lang.IllegalArgumentException: named capturing group has 0 length name".into()); }
                        if r.get(i)!=Some(&'}') { return Err("java.lang.IllegalArgumentException: named capturing group is missing trailing '}'".into()); }
                        let name:String=r[start..i].iter().collect();
                        i+=1;
                        if !r[start].is_ascii_alphabetic() { return Err("java.lang.IllegalArgumentException: capturing group name {".to_string()+&name+"} starts with digit character"); }
                        match caps.group_info().to_index(regex_automata::PatternID::ZERO,&name) {
                            Some(g)=>group(g,out),
                            None=>return Err(format!("java.lang.IllegalArgumentException: No group with name {{{name}}}")),
                        }
                    }
                    Some(c) if c.is_ascii_digit()=>{
                        let mut n=c.to_digit(10).unwrap() as usize;
                        if n>=groups { return Err(format!("java.lang.IndexOutOfBoundsException: No group {n}")); }
                        i+=1;
                        // Further digits while they still name a group.
                        while let Some(d)=r.get(i).and_then(|c|c.to_digit(10)) {
                            let next=n*10+d as usize;
                            if next>=groups { break; }
                            n=next;i+=1;
                        }
                        group(n,out);
                    }
                    Some(_)=>return Err("java.lang.IllegalArgumentException: Illegal group reference".into()),
                }
            }
            c=>{out.push(c);i+=1;}
        }
    }
    Ok(())
}

pub(crate) struct Captures {
    captures:regex_automata::util::captures::Captures,
    _memory:Reservation,
}
impl Captures {
    pub(crate) fn named<'a>(&self,text:&'a str,name:&str)->Option<&'a str> {
        self.captures.get_group_by_name(name).map(|span|&text[span.start..span.end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_matcher_preserves_unicode_empty_matches_and_literal_replacements() {
        let breakers=crate::memory::Breakers::private_for_tests(1<<30);
        crate::memory::scoped(Some(crate::memory::RequestScope::new("regex semantics",breakers)),|| {
            let mut cache=Cache::default();
            for pattern in ["a*","",r"\p{L}+","(?P<name>a+)"] {
                let compiled=cache.get(pattern).unwrap().unwrap();
                let reference=regex::Regex::new(pattern).unwrap();
                for text in ["", "ab", "💡aé"] {
                    assert_eq!(compiled.is_match(text).unwrap(),reference.is_match(text));
                    assert_eq!(compiled.replace(text,"$1").unwrap(),reference.replace_all(text,regex::NoExpand("$1")));
                }
            }
            assert!(cache.get("(").unwrap().is_none());
            assert!(breakers.request.reserved()>0);
            drop(cache);assert_eq!(breakers.request.reserved(),0);
        });
    }
}
