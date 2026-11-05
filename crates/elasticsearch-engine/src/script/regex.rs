//! `java.util.regex.Pattern` / `Matcher` over the `regex` crate, plus the
//! `script.painless.regex.enabled` switch.
//!
//! Differences from Java that cannot be expressed in the `regex` crate
//! (look-around, back-references, possessive quantifiers) surface as an
//! `invalid regex` error when the pattern is compiled.

use super::error::{illegal_arg, ScriptErr};
use super::runtime::Rt;
use super::value::Val;
use super::allocation::{Allocation,Scope};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};

type R<T> = Result<T, Box<ScriptErr>>;

// ---- `script.painless.regex.enabled` -------------------------------------

/// 0 = limited (default), 1 = true, 2 = false.
static MODE: AtomicU8 = AtomicU8::new(0);

/// Apply the `script.painless.regex.enabled` setting (`limited`, `true`, `false`).
pub fn set_regex_mode(v: &str) -> Result<(), String> {
    let m = match v {
        "limited" => 0,
        "true" => 1,
        "false" => 2,
        other => return Err(format!("Failed to parse value [{other}] as only [true], [false] or [limited] are allowed.")),
    };
    MODE.store(m, Ordering::Relaxed);
    Ok(())
}

pub fn regex_enabled() -> bool {
    MODE.load(Ordering::Relaxed) != 2
}

pub const DISABLED_MSG: &str = "Regexes are disabled. Set [script.painless.regex.enabled] to [true] in elasticsearch.yaml to allow them. Be careful though, regexes break out of Painless's protection against deep recursion and long loops.";

// ---- patterns -------------------------------------------------------------

pub const CANON_EQ: i32 = 128;
pub const CASE_INSENSITIVE: i32 = 2;
pub const COMMENTS: i32 = 4;
pub const DOTALL: i32 = 32;
pub const LITERAL: i32 = 16;
pub const MULTILINE: i32 = 8;
pub const UNICODE_CASE: i32 = 64;
pub const UNICODE_CHARACTER_CLASS: i32 = 256;
pub const UNIX_LINES: i32 = 1;

pub fn flag_const(name: &str) -> Option<i32> {
    Some(match name {
        "CANON_EQ" => CANON_EQ,
        "CASE_INSENSITIVE" => CASE_INSENSITIVE,
        "COMMENTS" => COMMENTS,
        "DOTALL" => DOTALL,
        "LITERAL" => LITERAL,
        "MULTILINE" => MULTILINE,
        "UNICODE_CASE" => UNICODE_CASE,
        "UNICODE_CHARACTER_CLASS" => UNICODE_CHARACTER_CLASS,
        "UNIX_LINES" => UNIX_LINES,
        _ => return None,
    })
}

/// Flags of a regex literal suffix (`/a/i`).
pub fn literal_flags(s: &str) -> Result<i32, String> {
    let mut f = 0;
    for c in s.chars() {
        f |= match c {
            'c' => CANON_EQ,
            'i' => CASE_INSENSITIVE,
            'l' => LITERAL,
            'm' => MULTILINE,
            's' => DOTALL,
            'U' => UNICODE_CHARACTER_CLASS,
            'u' => UNICODE_CASE,
            'x' => COMMENTS,
            other => return Err(format!("invalid regular expression: unknown flag [{other}]")),
        };
    }
    Ok(f)
}

/// Explicit caches keep per-search allocations out of the regex crate's
/// hidden thread-local pool. Byte syntax and leftmost-first semantics match
/// regex::bytes::Regex; PikeVM avoids unaccounted lazy DFA growth.
#[derive(Debug)]
pub(super) struct CompiledRegex { re:regex_automata::meta::Regex, native:Option<regex::bytes::Regex> }
impl CompiledRegex {
    fn captures_len(&self)->usize {self.re.captures_len()}
    fn capture_names(&self)->impl Iterator<Item=Option<&str>> {self.re.group_info().pattern_names(regex_automata::PatternID::ZERO)}
    pub(super) fn memory_usage(&self)->usize {self.re.memory_usage()}
}
fn parser_scratch_bytes(pattern:&str)->u64 {
    // Unicode character classes expand independently of their short spelling.
    // Charge their HIR/range construction separately from the NFA size cap.
    let unicode_atoms=pattern.as_bytes().windows(2).filter(|p|p[0]==b'\\'&&matches!(p[1],b'p'|b'P'|b'w'|b'W'|b'd'|b'D'|b's'|b'S')).count() as u64;
    (pattern.len() as u64).saturating_mul(256).saturating_add(unicode_atoms.saturating_mul(8<<20)).saturating_add(8192)
}
fn check_translation_depth(pattern:&str)->R<()> {
    if Scope::current().is_none(){return Ok(())}
    let mut depth=0usize;let mut escaped=false;let mut amp=false;
    let mut recursion=[0usize;129];
    for c in pattern.chars() {
        if escaped{escaped=false;amp=false;continue;}
        if c=='\\'{escaped=true;amp=false;continue;}
        if c=='[' {
            depth+=1;
            if depth>=recursion.len(){return Err(translation_depth_error());}
            recursion[depth]=recursion[depth-1]+1;
        }
        // class_parts recursively translates the right side of every &&,
        // including chains whose visible square-bracket depth is only one.
        if c=='&'&&amp&&depth>0 {recursion[depth]+=1;amp=false;} else {amp=c=='&';}
        if recursion[depth]>128{return Err(translation_depth_error());}
        if c==']'{depth=depth.saturating_sub(1);amp=false;}
    }
    Ok(())
}
fn translation_depth_error()->Box<ScriptErr> {
    ScriptErr::breaker(crate::error::EsError::new(429,"circuit_breaking_exception","regex character class nesting exceeds metric translation limit"))
}
fn compile_regex(pattern:&str)->R<Arc<CompiledRegex>> {
    let scratch=Allocation::new();
    let mut budget=1024usize;
    loop {
        if scratch.enabled() {scratch.reserve((budget as u64).saturating_mul(16).saturating_add(parser_scratch_bytes(pattern)))?;}
        let config=regex_automata::meta::Regex::config()
            .utf8_empty(false).nfa_size_limit(Some(budget))
            .hybrid(false).dfa(false).onepass(false).backtrack(false);
        let result=regex_automata::meta::Regex::builder().configure(config)
            .syntax(regex_automata::util::syntax::Config::new().utf8(false)).build(pattern);
        match result {
            Ok(re)=>{
                let native=if Scope::current().is_none() {Some(regex::bytes::Regex::new(pattern).map_err(|e|ScriptErr::new("pattern_syntax_exception",e.to_string()))?)}else{None};
                let re=Arc::new(CompiledRegex{re,native});
                Scope::adopt_regex(&re)?;
                return Ok(re);
            },
            Err(error) if error.size_limit().is_some() && budget<10*(1<<20)=>{budget=(budget*2).min(10*(1<<20));},
            Err(error)=>return Err(ScriptErr::new("pattern_syntax_exception",error.syntax_error().map(ToString::to_string).unwrap_or_else(||error.to_string()))),
        }
    }
}

#[derive(Debug)]
pub struct PatternData {
    pub source: String,
    source_units: Vec<u16>,
    pub flags: i32,
    re: Arc<CompiledRegex>,
    full: OnceLock<Arc<CompiledRegex>>,
    start: OnceLock<Arc<CompiledRegex>>,
    translated: String,
}

fn translate(src: &str, flags: i32) -> String {
    let mut body = if flags & LITERAL != 0 { regex::escape(src) } else { java_to_rust(src) };
    let mut pre = String::new();
    if flags & CASE_INSENSITIVE != 0 {
        pre.push('i');
    }
    if flags & MULTILINE != 0 {
        pre.push('m');
    }
    if flags & DOTALL != 0 {
        pre.push('s');
    }
    if flags & COMMENTS != 0 {
        pre.push('x');
    }
    if !pre.is_empty() {
        body = format!("(?{pre}){body}");
    }
    body
}

/// Rewrite the Java-only escapes the `regex` crate spells differently.
fn java_to_rust(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut it = src.chars().peekable();
    let mut in_quote = false;
    while let Some(c) = it.next() {
        if in_quote {
            if c == '\\' && it.peek() == Some(&'E') {
                it.next();
                in_quote = false;
            } else {
                out.push_str(&regex::escape(&c.to_string()));
            }
            continue;
        }
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Q') => in_quote = true,
            Some('Z') => out.push_str("(?:\\n?\\z)"),
            Some('h') => out.push_str("[ \\t\\xA0]"),
            Some('R') => out.push_str("(?:\\r\\n|[\\n\\r\\x0B\\f])"),
            Some('e') => out.push_str("\\x1B"),
            Some(n) => {
                out.push('\\');
                out.push(n);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn flag_letters(f: i32) -> String {
    let mut s = String::new();
    for (bit, c) in [(CANON_EQ, 'c'), (CASE_INSENSITIVE, 'i'), (LITERAL, 'l'), (MULTILINE, 'm'), (DOTALL, 's'), (UNICODE_CHARACTER_CLASS, 'U'), (UNICODE_CASE, 'u'), (COMMENTS, 'x')] {
        if f & bit != 0 {
            s.push(c);
        }
    }
    s
}

pub fn compile_pattern(src: &str, flags: i32) -> R<PatternData> {
    let _text=Allocation::new();
    _text.reserve((src.len() as u64).saturating_mul(1024).saturating_add(8192))?;
    let java=translate(src,flags);
    check_translation_depth(&java)?;
    let translated=super::regex_text::byte_pattern(&java);
    let re = compile_regex(&translated).map_err(|e| {
        if e.breaker.is_some() {return e;}
        let msg = e.msg.to_lowercase();
        let desc = if msg.contains("unclosed character class") {
            "Unclosed character class".to_string()
        } else if msg.contains("unclosed group") {
            "Unclosed group".to_string()
        } else if msg.contains("unopened group") || msg.contains("unmatched") {
            "Unmatched closing ')'".to_string()
        } else if msg.contains("repetition operator missing expression") {
            let c = src.chars().find(|c| matches!(c, '*' | '+' | '?')).unwrap_or('*');
            format!("Dangling meta character '{c}'")
        } else if msg.contains("look-around") || msg.contains("backreference") {
            "unsupported construct".to_string()
        } else {
            msg.lines().last().unwrap_or("invalid regular expression").trim().to_string()
        };
        let mut s = ScriptErr::new("illegal_argument_exception", format!("invalid regular expression: could not compile regex constant [{src}] with flags [{}]: {desc}", flag_letters(flags)));
        s.compile = true;
        s
    })?;
    Ok(PatternData { source: src.to_string(), source_units:src.encode_utf16().collect(), flags, re, full: OnceLock::new(), start: OnceLock::new(), translated })
}

/// Raw Java pattern text is decoded by code point; unpaired units are exact
/// WTF-8 byte atoms, rather than replacement characters.
pub fn compile_pattern_units(units:&[u16],flags:i32)->R<PatternData> {
    let _text=Allocation::new();
    _text.reserve((units.len() as u64).saturating_mul(1024).saturating_add(8192))?;
    if let Ok(text)=String::from_utf16(units){return compile_pattern(&text,flags);}
    let mut text=String::new();let mut i=0;let mut quoted=flags&LITERAL!=0;
    while i<units.len() {
        if flags&LITERAL==0&&units[i]==92 {
            if !quoted&&units.get(i+1)==Some(&81){quoted=true;i+=2;continue;}
            if quoted&&units.get(i+1)==Some(&69){quoted=false;i+=2;continue;}
        }
        let unit=units[i];
        if (0xd800..=0xdbff).contains(&unit)&&units.get(i+1).is_some_and(|v|(0xdc00..=0xdfff).contains(v)) {
            let cp=0x10000+(((unit as u32-0xd800)<<10)|(units[i+1] as u32-0xdc00));let c=char::from_u32(cp).unwrap();
            if quoted{text.push_str(&regex::escape(&c.to_string()));}else{text.push(c);}i+=2;continue;
        }
        if (0xd800..=0xdfff).contains(&unit){text.push_str(&format!(r"\u{unit:04X}"));}
        else{let c=char::from_u32(unit as u32).unwrap();if quoted{text.push_str(&regex::escape(&c.to_string()));}else{text.push(c);}}
        i+=1;
    }
    let mut pattern=compile_pattern(&text,flags&!LITERAL)?;
    pattern.source=String::from_utf16_lossy(units); // diagnostic display only
    pattern.source_units=units.to_vec();pattern.flags=flags;
    Ok(pattern)
}

impl PatternData {
    fn full(&self) -> R<&Arc<CompiledRegex>> {
        if self.full.get().is_none() {
            let re=compile_regex(&format!("\\A(?:{})\\z",self.translated))?;
            let _=self.full.set(re);
        }
        Ok(self.full.get().unwrap())
    }
    fn start(&self) -> R<&Arc<CompiledRegex>> {
        if self.start.get().is_none() {
            let re=compile_regex(&format!("\\A(?:{})",self.translated))?;
            let _=self.start.set(re);
        }
        Ok(self.start.get().unwrap())
    }
    pub(super) fn memory_bytes(&self)->u64 {
        self.source.capacity() as u64+(self.source_units.capacity()*2) as u64+self.translated.capacity() as u64
            +256
    }
}

// ---- matchers -------------------------------------------------------------
// Capture offsets and cursor state are UTF-16 units, never Rust byte offsets.
type Caps = Vec<Option<(usize, usize)>>;
#[derive(Debug)]
pub struct MatcherState {
    pub pat: Arc<PatternData>,
    pub input: Rc<[u16]>,
    pos: usize,
    caps: Option<Caps>,
    append: usize,
    region_start: usize,
    region_end: usize,
    allocation: Allocation,
}
pub fn new_matcher(pat:&Arc<PatternData>, input:&str)->R<Val> {
    let scratch=Allocation::new();scratch.reserve((input.len() as u64).saturating_mul(4))?;
    new_matcher_units(pat,&input.encode_utf16().collect::<Vec<_>>())
}
fn matcher_state(pat:&Arc<PatternData>,input:&[u16])->R<MatcherState> {
    Scope::adopt(&Val::Pattern(pat.clone()))?;
    let allocation=Allocation::new();
    allocation.reserve((input.len() as u64).saturating_mul(2).saturating_add((pat.re.captures_len() as u64).saturating_mul(128)).saturating_add(256))?;
    Ok(MatcherState{pat:pat.clone(),input:Rc::from(input),pos:0,caps:None,append:0,region_start:0,region_end:input.len(),allocation})
}
pub fn new_matcher_units(pat:&Arc<PatternData>,input:&[u16])->R<Val> {
    Ok(Val::Matcher(Rc::new(RefCell::new(matcher_state(pat,input)?))))
}
fn capture(re:&Arc<CompiledRegex>,input:&[u16],from:usize)->R<Option<Caps>> {
    if from>input.len(){return Ok(None);}
    if Scope::current().is_none() {
        if let Some(native)=&re.native {
            let text=super::regex_text::Text::new(input,from);
            return Ok(native.captures_at(&text.bytes,text.byte(from)).map(|c|(0..native.captures_len()).map(|i|c.get(i).map(|m|(text.unit(m.start()),text.unit(m.end())))).collect()));
        }
    }
    Scope::adopt_regex(re)?;
    let scratch=Allocation::new();
    scratch.reserve((input.len() as u64).saturating_mul(64).saturating_add((re.captures_len() as u64).saturating_mul(128)).saturating_add(1024))?;
    scratch.reserve((input.len() as u64).saturating_mul(64).saturating_add((re.memory_usage() as u64).saturating_mul(re.captures_len() as u64+1).saturating_mul(16)).saturating_add(8192))?;
    let text=super::regex_text::Text::new(input,from);
    let mut cache=re.re.create_cache();
    let mut caps=re.re.create_captures();
    re.re.search_captures_with(&mut cache,&regex_automata::Input::new(&text.bytes).range(text.byte(from)..),&mut caps);
    Ok(caps.is_match().then(||(0..re.captures_len()).map(|i|caps.get_group(i).map(|m|(text.unit(m.start),text.unit(m.end)))).collect()))
}
fn capture_region(re:&Arc<CompiledRegex>,m:&MatcherState,from:usize)->R<Option<Caps>> {
    if from>m.region_end{return Ok(None);}
    let Some(mut caps)=capture(re,&m.input[m.region_start..m.region_end],from.saturating_sub(m.region_start))? else {return Ok(None)};
    for range in caps.iter_mut().flatten(){range.0+=m.region_start;range.1+=m.region_start;}
    Ok(Some(caps))
}
fn append_units(out:&mut Vec<u16>,units:&[u16],allocation:&Allocation)->R<()> {
    let needed=out.len().checked_add(units.len()).ok_or_else(||illegal_arg("replacement too large"))?;
    let capacity=if needed>out.capacity(){needed.max(out.capacity().saturating_mul(2)).max(8)}else{out.capacity()};
    allocation.reserve((capacity as u64).saturating_mul(2).saturating_add(256))?;
    if capacity>out.capacity(){out.try_reserve_exact(capacity-out.len()).map_err(|_|ScriptErr::breaker(crate::error::EsError::new(429,"circuit_breaking_exception","regex output allocation failed")))?;}
    out.extend_from_slice(units);Ok(())
}
fn no_match()->Box<ScriptErr> {ScriptErr::new("illegal_state_exception","No match found")}
fn group_range(m:&MatcherState,idx:usize)->R<Option<(usize,usize)>> {
    let caps=m.caps.as_ref().ok_or_else(no_match)?;
    caps.get(idx).copied().ok_or_else(||ScriptErr::new("index_out_of_bounds_exception",format!("No group {idx}")))
}
fn group_by_name(m:&MatcherState,name:&str)->R<usize> {
    m.pat.re.capture_names().position(|n|n==Some(name)).ok_or_else(||illegal_arg(format!("No group with name <{name}>")))
}
fn expand(m:&MatcherState,caps:&Caps,template:&[u16])->R<(Vec<u16>,Allocation)> {
    let allocation=Allocation::new();
    let mut out=Vec::new();let mut i=0;
    while i<template.len() {
        let c=template[i];i+=1;
        if c==92 {
            let Some(&next)=template.get(i) else{return Err(illegal_arg("character to be escaped is missing"))};
            append_units(&mut out,&[next],&allocation)?;i+=1;
        } else if c==36 {
            let idx=if template.get(i)==Some(&123) {
                i+=1;let start=i;while i<template.len()&&template[i]!=125{i+=1;}
                if i==template.len(){return Err(illegal_arg("named capturing group is missing trailing '}'"));}
                let name=String::from_utf16(&template[start..i]).map_err(|_|illegal_arg("invalid group name"))?;i+=1;
                group_by_name(m,&name)?
            } else {
                let Some(&digit)=template.get(i) else{return Err(illegal_arg("Illegal group reference"))};
                if !(48..=57).contains(&digit){return Err(illegal_arg("Illegal group reference"));}
                let mut n=(digit-48) as usize;i+=1;
                while let Some(&digit)=template.get(i) {
                    if !(48..=57).contains(&digit){break;}
                    let next=n*10+(digit-48) as usize;
                    if next>=caps.len(){break;} n=next;i+=1;
                }
                if n>=caps.len(){return Err(ScriptErr::new("index_out_of_bounds_exception",format!("No group {n}")));}n
            };
            if let Some((start,end))=caps[idx]{append_units(&mut out,&m.input[start..end],&allocation)?;}
        } else {append_units(&mut out,&[c],&allocation)?;}
    }
    Ok((out,allocation))
}
fn replace(rt:&mut Rt,m:&Rc<RefCell<MatcherState>>,replacement:&Val,all:bool)->R<Val> {
    let (pat,input)={let mut b=m.borrow_mut();b.pos=0;b.caps=None;b.append=0;b.region_start=0;b.region_end=b.input.len();(b.pat.clone(),b.input.clone())};
    let _replacement_copy=Allocation::text_copy(replacement)?;
    let output_allocation=Allocation::new();
    let template=matches!(replacement,Val::Str(_)|Val::Utf16(_)).then(||replacement.string_units());
    let mut out=Vec::new();let mut last=0;let mut from=0;
    while let Some(caps)=capture(&pat.re,&input,from)? {
        let (start,end)=caps[0].unwrap();
        append_units(&mut out,&input[last..start],&output_allocation)?;
        m.borrow_mut().caps=Some(caps.clone());
        if let Some(template)=&template {let (expanded,_guard)=expand(&m.borrow(),&caps,template)?;append_units(&mut out,&expanded,&output_allocation)?;}
        else {let value=rt.call_fn(replacement,&[Val::Matcher(m.clone())])?;let _copy=Allocation::text_copy(&value)?;append_units(&mut out,&value.string_units(),&output_allocation)?;}
        last=end;from=if start==end{end+1}else{end};
        m.borrow_mut().pos=from;
        if !all{break;}
    }
    append_units(&mut out,&input[last..],&output_allocation)?;
    if all||last==0&&m.borrow().caps.is_none(){let mut b=m.borrow_mut();b.caps=None;b.pos=b.input.len()+1;}
    let result=Val::utf16(out);Scope::adopt(&result)?;Ok(result)
}
pub(crate) fn replace_template_units(pattern:&Arc<PatternData>,input:&[u16],template:&[u16],all:bool)->R<Vec<u16>> {
    let mut matcher=matcher_state(pattern,input)?;
    let output_allocation=Allocation::new();
    let mut output=Vec::new();let mut from=0;let mut last=0;
    while let Some(caps)=capture(&pattern.re,input,from)? {
        let (start,end)=caps[0].unwrap();append_units(&mut output,&input[last..start],&output_allocation)?;
        matcher.caps=Some(caps.clone());let (expanded,_guard)=expand(&matcher,&caps,template)?;append_units(&mut output,&expanded,&output_allocation)?;
        last=end;from=if start==end{end+1}else{end};if !all{break;}
    }
    append_units(&mut output,&input[last..],&output_allocation)?;Ok(output)
}
pub fn matcher_method(rt:&mut Rt,m:&Rc<RefCell<MatcherState>>,name:&str,args:&[Val])->R<Option<Val>> {
    let _args=args.iter().map(Allocation::text_copy).collect::<R<Vec<_>>>()?;
    Ok(Some(match (name,args.len()) {
        ("matches",0)|("lookingAt",0)=>{
            let mut b=m.borrow_mut();let pat=b.pat.clone();
            b.caps=capture_region(if name=="matches"{pat.full()?}else{pat.start()?},&b,b.region_start)?;
            if let Some((start,end))=b.caps.as_ref().and_then(|c|c[0]){b.pos=if start==end{end+1}else{end};}
            Val::Bool(b.caps.is_some())
        },
        ("find",0)|("find",1)=>{
            let mut b=m.borrow_mut();
            if let Some(arg)=args.first(){let from=arg.to_i64()?;if from<0||from as usize>b.input.len(){return Err(ScriptErr::new("index_out_of_bounds_exception","Illegal start index"));}b.pos=from as usize;b.append=0;b.region_start=0;b.region_end=b.input.len();}
            b.caps=capture_region(&b.pat.re,&b,b.pos)?;
            if let Some(caps)=&b.caps {let (start,end)=caps[0].unwrap();b.pos=if start==end{end+1}else{end};Val::Bool(true)} else {b.pos=b.region_end+1;Val::Bool(false)}
        },
        ("reset",0)=>{let mut b=m.borrow_mut();b.pos=0;b.caps=None;b.append=0;b.region_start=0;b.region_end=b.input.len();Val::Matcher(m.clone())},
        ("region",2)=>{let start=args[0].to_i64()?;let end=args[1].to_i64()?;let mut b=m.borrow_mut();if start<0||end<start||end as usize>b.input.len(){return Err(ScriptErr::new("index_out_of_bounds_exception","Invalid region"));}b.region_start=start as usize;b.region_end=end as usize;b.pos=start as usize;b.caps=None;b.append=0;Val::Matcher(m.clone())},
        ("regionStart",0)=>Val::Int(m.borrow().region_start as i32),
        ("regionEnd",0)=>Val::Int(m.borrow().region_end as i32),
        ("usePattern",1)=>{let Val::Pattern(pattern)=&args[0]else{return Err(illegal_arg("Pattern required"))};let mut b=m.borrow_mut();b.allocation.reserve((b.input.len() as u64).saturating_mul(2).saturating_add((pattern.re.captures_len() as u64).saturating_mul(128)).saturating_add(256))?;b.pat=pattern.clone();b.caps=None;Val::Matcher(m.clone())},
        ("group",0)=>group_val(&m.borrow(),0)?,
        ("group",1)=>{
            if matches!(args[0],Val::Str(_)|Val::Utf16(_)){return Err(ScriptErr::new("wrong_method_type_exception","cannot convert MethodHandle(Matcher,int)String to (Object,String)Object"));}
            group_val(&m.borrow(),args[0].to_i64()? as usize)?
        },
        ("namedGroup",1)=>{let b=m.borrow();let name=String::from_utf16(&args[0].string_units()).map_err(|_|illegal_arg("invalid group name"))?;group_val(&b,group_by_name(&b,&name)?)?},
        ("quoteReplacement",1)=>Val::utf16(quote_replacement(&args[0].string_units())),
        ("groupCount",0)=>Val::Int(m.borrow().pat.re.captures_len() as i32-1),
        ("start"|"end",0)|("start"|"end",1)=>{
            let b=m.borrow();let idx=match args.first(){Some(Val::Str(s))=>group_by_name(&b,s)?,Some(v)=>v.to_i64()? as usize,None=>0};
            Val::Int(match group_range(&b,idx)?{Some((s,e))=>if name=="start"{s as i32}else{e as i32},None=>-1})
        },
        ("hitEnd",0)=>Val::Bool(m.borrow().pos>=m.borrow().input.len()),
        ("replaceAll",1)|("replaceFirst",1)=>replace(rt,m,&args[0],name=="replaceAll")?,
        ("pattern",0)=>Val::Pattern(m.borrow().pat.clone()),
        ("toString",0)=>Val::string(Val::Matcher(m.clone()).to_java_string()),
        _=>return Ok(None),
    }))
}
fn group_val(m:&MatcherState,idx:usize)->R<Val>{
    let allocation=Allocation::new();
    let result=match group_range(m,idx)?{Some((s,e))=>{allocation.reserve(((e-s) as u64).saturating_mul(4).saturating_add(256))?;Val::utf16(m.input[s..e].to_vec())},None=>Val::Null};
    Scope::adopt(&result)?;Ok(result)
}
pub fn pattern_method(p:&Arc<PatternData>,name:&str,args:&[Val])->R<Option<Val>> {
    let _args=args.iter().map(Allocation::text_copy).collect::<R<Vec<_>>>()?;
    Ok(Some(match (name,args.len()) {
        ("matcher",1)=>{let _copy=Allocation::text_copy(&args[0])?;new_matcher_units(p,&args[0].string_units())?},
        ("pattern"|"toString",0)=>{
            let guard=Allocation::new();guard.reserve((p.source_units.len() as u64).saturating_mul(4).saturating_add(256))?;
            let result=Val::utf16(p.source_units.clone());Scope::adopt(&result)?;result
        },
        ("flags",0)=>Val::Int(p.flags),
        ("splitAsStream",1)|("split",1)|("split",2)=>{
            let units=args[0].string_units();
            let allocation=super::allocation::Allocation::values(units.len().saturating_add(1).saturating_mul(4))?;
            let parts=split_units(p,&units,if args.len()==2{args[1].to_i64()?}else{0})?;
            let values:Vec<Val>=parts.into_iter().map(Val::utf16).collect();
            for value in &values {Scope::adopt(value)?;}
            if name=="splitAsStream" {Val::stream_from_values(values,allocation)}
            else {Val::new_array(super::ast::Ty::String,values).allocated(allocation)}
        },
        _=>return Ok(None),
    }))
}
pub fn split_units(p:&PatternData,input:&[u16],limit:i64)->R<Vec<Vec<u16>>> {
    let allocation=Allocation::values(input.len().saturating_add(1).saturating_mul(4))?;
    allocation.reserve((input.len() as u64).saturating_add(1).saturating_mul(256))?;
    let mut out=Vec::new();let mut last=0;let mut from=0;
    while let Some(caps)=capture(&p.re,input,from)? {
        let (start,end)=caps[0].unwrap();
        if limit>0&&out.len() as i64==limit-1{break;}
        from=if start==end{end+1}else{end};
        if end==0{continue;}
        out.push(input[last..start].to_vec());last=end;
    }
    out.push(input[last..].to_vec());
    if limit==0 {while out.len()>1&&out.last().is_some_and(Vec::is_empty){out.pop();}if out.len()==1&&out[0].is_empty()&&!input.is_empty(){out.clear();}}
    Ok(out)
}
pub fn pattern_static(method:&str,args:&[Val])->R<Option<Val>> {
    let _args=args.iter().map(Allocation::text_copy).collect::<R<Vec<_>>>()?;
    let result=match (method,args.len()) {
        ("compile",1)=>Val::Pattern(Arc::new(compile_pattern_units(&args[0].string_units(),0)?)),
        ("compile",2)=>Val::Pattern(Arc::new(compile_pattern_units(&args[0].string_units(),args[1].to_i64()? as i32)?)),
        ("matches",2)=>{let p=compile_pattern_units(&args[0].string_units(),0)?;Val::Bool(capture(p.full()?,&args[1].string_units(),0)?.is_some())},
        ("quote",1)=>{let mut units=vec![92,81];let input=args[0].string_units();let mut i=0;while i<input.len(){if input[i]==92&&input.get(i+1)==Some(&69){units.extend([92,69,92,92,69,92,81]);i+=2;}else{units.push(input[i]);i+=1;}}units.extend([92,69]);Val::utf16(units)},
        _=>return Ok(None),
    };
    Scope::adopt(&result)?;
    Ok(Some(result))
}
pub fn regex_op(find:bool,text:&Val,pat:&Val)->R<Val> {
    let _text=Allocation::text_copy(text)?;
    let Val::Pattern(p)=pat else{return Err(super::error::class_cast(format!("Cannot cast {} to java.util.regex.Pattern",pat.type_name())))};
    if matches!(text,Val::Null){return Err(super::error::npe("Cannot invoke CharSequence.length because input is null"));}
    Ok(Val::Bool(capture(if find{&p.re}else{p.full()?},&text.string_units(),0)?.is_some()))
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    #[test]
    fn metric_explicit_regex_cache_matches_native_captures_and_releases() {
        let _budget = rustic_core::admission::private_shared(256 << 20);
        let breakers=crate::memory::Breakers::private_for_tests(1 << 30);
        // The pattern and the expected captures are made in a scope of their
        // own breakers: outside one they would charge the process's, which
        // the rest of the suite (running in parallel) may have filled.
        let setup=crate::memory::Breakers::private_for_tests(1 << 30);
        for source in ["(.)", "(?P<letter>a+)|b", "[a-z]+", "a*", "(?:ab){1,3}"] {
            let input=[97u16,98,97,97,0xd800,98];
            let (pattern,expected)=crate::memory::scoped(Some(crate::memory::RequestScope::new("metric regex expected",setup)),|| {
                let pattern=Arc::new(compile_pattern(source,0).unwrap());
                let expected=(0..=input.len()).map(|from|capture(&pattern.re,&input,from).unwrap()).collect::<Vec<_>>();
                (pattern,expected)
            });
            let request=crate::memory::RequestScope::new("metric regex cache",breakers);
            crate::memory::scoped(Some(request),|| {
                let scope=Scope::enter();
                for _ in 0..100 {
                    for (from,expected) in expected.iter().enumerate() {assert_eq!(&capture(&pattern.re,&input,from).unwrap(),expected,"{source}");}
                }
                assert!(breakers.request.reserved()>0);
                drop(scope);assert_eq!(breakers.request.reserved(),0);
            });
        }
    }

    #[test]
    fn metric_shallow_intersection_chain_rejects_before_recursive_translation() {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        let request=crate::memory::RequestScope::new("regex intersection recursion",breakers);
        crate::memory::scoped(Some(request),|| {
            let scope=Scope::enter();
            let source=format!("[a{}]","&&a".repeat(1000));
            assert!(check_translation_depth(&source).unwrap_err().breaker.is_some());
            assert!(compile_pattern(&source,0).unwrap_err().breaker.is_some());
            drop(scope);assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn metric_regex_compilation_breaks_before_large_automaton() {
        let breakers=crate::memory::Breakers::private_for_tests(1 << 30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),
            _=>None,
        });
        let request=crate::memory::RequestScope::new("metric regex compile",breakers);
        crate::memory::scoped(Some(request),|| {
            let scope=Scope::enter();
            assert!(compile_pattern("a{10000000}",0).unwrap_err().breaker.is_some());
            drop(scope);assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn split_scalar_alias_stays_charged_after_array_drop() {
        let _budget = rustic_core::admission::private_shared(256 << 20);
        // Compiled in a scope of its own breakers: outside one it would
        // charge the process's, which the parallel suite may have filled.
        let setup=crate::memory::Breakers::private_for_tests(1 << 30);
        let pattern=crate::memory::scoped(Some(crate::memory::RequestScope::new("split alias pattern",setup)),|| Arc::new(compile_pattern(",",0).unwrap()));
        let breakers=crate::memory::Breakers::private_for_tests(1 << 30);
        let request=crate::memory::RequestScope::new("split alias",breakers);
        crate::memory::scoped(Some(request),|| {
            let scope=Scope::enter();
            let input=Val::str(&"x".repeat(8192));
            let result=pattern_method(&pattern,"split",&[input]).unwrap().unwrap();
            let Val::Array{items,..}=&result else {panic!("split array expected")};
            let escaped=items.borrow()[0].clone();
            drop(result);
            Scope::collect_all_dead();
            let held=breakers.request.reserved();
            assert!(held>=8192,"split scalar owner must outlive temporary array");
            drop(escaped);Scope::collect_all_dead();
            assert!(breakers.request.reserved()<held);
            drop(scope);assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn metric_regex_buffers_fail_before_copy_or_expansion() {
        let pattern=Arc::new(compile_pattern("(a)",0).unwrap());
        let input=vec![97u16;40000];
        let breakers=crate::memory::Breakers::leak_for_tests(1 << 30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),
            _=>None,
        });
        let request=crate::memory::RequestScope::new("metric regex buffers",breakers);
        crate::memory::scoped(Some(request),|| {
            let scope=Scope::enter();
            assert!(new_matcher_units(&pattern,&input).unwrap_err().breaker.is_some());
            assert!(split_units(&pattern,&input,0).unwrap_err().breaker.is_some());
            assert!(capture(&pattern.re,&input,0).unwrap_err().breaker.is_some());
            let matcher=matcher_state(&pattern,&[97]).unwrap();
            let caps=vec![Some((0,1)),Some((0,1))];
            let template=[36u16,49].repeat(40000);
            assert!(expand(&matcher,&caps,&template).unwrap_err().breaker.is_some());
            drop(matcher);
            drop(scope);
            assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn captures_and_replacements_preserve_utf16_units() {
        let pat=Arc::new(compile_pattern("(.)",0).unwrap());
        let input:Rc<[u16]>=Rc::from([0xd800,65,0xd83d,0xde00,0xdfff]);
        let caps=capture(&pat.re,&input,0).unwrap().unwrap();
        assert_eq!(caps[0],Some((0,1)));
        let mut matcher=matcher_state(&pat,&input).unwrap();matcher.caps=Some(caps.clone());
        assert_eq!(group_val(&matcher,1).unwrap().string_units(),vec![0xd800]);
        assert_eq!(expand(&matcher,&caps,&[0xdfff,36,49]).unwrap().0,vec![0xdfff,0xd800]);
        assert_eq!(capture(&matcher.pat.re,&matcher.input,3).unwrap().unwrap()[0],Some((3,4)));
    }
}

pub(crate) fn quote_replacement(input:&[u16])->Vec<u16>{let mut out=Vec::new();for &unit in input {if unit==92||unit==36{out.push(92);}out.push(unit);}out}
