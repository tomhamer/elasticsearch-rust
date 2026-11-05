//! Java string casing over UTF-16, including conditional locale mappings.
use super::{error::{class_cast, npe, SResult}, value::Val};
use super::locale_case_data as data;
use super::locale_word_data as word;

fn combining(cp:u32)->u8 {
    data::COMBINING.binary_search_by_key(&cp,|p|p.0).map(|i|data::COMBINING[i].1).unwrap_or(0)
}
fn soft_dotted(cp:u32)->bool { data::SOFT_DOTTED.binary_search_by_key(&cp,|p|p.0).is_ok() }
fn word_boundaries(cps:&[u32])->Vec<bool> {
    let mut boundaries=vec![false;cps.len()+1];
    boundaries[0]=true;
    let mut start=0;
    while start<cps.len() {
        let (mut result,mut lookahead,mut state,mut i)=(start+1,0,1usize,start);
        while i<cps.len() && cps[i]!=0xffff && state!=0 {
            let entry=word::RANGES.partition_point(|&(first,_,_)|first<=cps[i])-1;
            let category=word::RANGES[entry].2;
            if category>=0 {state=word::STATES[state*word::CATEGORIES+category as usize];}
            if word::LOOKAHEAD[state] {
                if word::ENDS[state] {result=lookahead;} else {lookahead=i+1;}
            } else if word::ENDS[state] {result=i+1;}
            i+=1;
        }
        if i==cps.len() && lookahead==cps.len() {result=lookahead;}
        boundaries[result]=true;
        start=result;
    }
    boundaries
}
fn final_sigma(cps:&[u32],at:usize,boundaries:&[bool])->bool {
    let cased=|cp:u32| {
        let i=word::CASED.partition_point(|&(first,_)|first<=cp);
        i>0 && cp<=word::CASED[i-1].1
    };
    let mut before=at;
    while before>0 && !boundaries[before] {
        before-=1;
        if cased(cps[before]) {
            let mut after=at+1;
            while after<cps.len() && !boundaries[after] {
                if cased(cps[after]) {return false;}
                after+=1;
            }
            return true;
        }
    }
    false
}

fn after(cps:&[u32],at:usize,predicate:impl Fn(u32)->bool)->bool {
    for &cp in cps[..at].iter().rev() {
        if predicate(cp) {return true;}
        if matches!(combining(cp),0|230) {return false;}
    }
    false
}
fn before_dot(cps:&[u32],at:usize)->bool {
    for &cp in &cps[at+1..] {
        if cp==0x307 {return true;}
        if matches!(combining(cp),0|230) {return false;}
    }
    false
}
fn more_above(cps:&[u32],at:usize)->bool {
    for &cp in &cps[at+1..] {
        match combining(cp) {230=>return true,0=>return false,_=>{}}
    }
    false
}

pub(super) fn map(units:&[u16],upper:bool,args:&[Val])->SResult<Val> {
    let language=match args.first() {
        None=>String::new(),
        Some(Val::Obj(o))=>match &**o {
            super::jmisc::Obj::Locale {language,..}=>String::from_utf16(language).unwrap_or_default(),
            _=>return Err(class_cast("Expected java.util.Locale")),
        },
        Some(Val::Time(t))=>match &**t {
            super::jtime::JTime::Locale(locale)=>locale.language().to_owned(),
            _=>return Err(class_cast("Expected java.util.Locale")),
        },
        Some(Val::Null)=>return Err(npe("locale is null")),
        Some(_)=>return Err(class_cast("Expected java.util.Locale")),
    };
    let cps:Vec<u32>=char::decode_utf16(units.iter().copied()).map(|c|match c {Ok(c)=>c as u32,Err(e)=>e.unpaired_surrogate() as u32}).collect();
    let boundaries=if !upper && cps.contains(&0x3a3) {word_boundaries(&cps)} else {Vec::new()};
    let turkic=matches!(language.as_str(),"tr"|"az");
    let lithuanian=language=="lt";
    let mut out=Vec::with_capacity(units.len());
    for (i,&cp) in cps.iter().enumerate() {
        if !upper && cp==0x3a3 && final_sigma(&cps,i,&boundaries) {out.push(0x3c2);continue;}
        if turkic {
            match (upper,cp) {
                (true,0x69)=>{out.push(0x130);continue;}
                (false,0x130)=>{out.push(0x69);continue;}
                (false,0x49) if !before_dot(&cps,i)=>{out.push(0x131);continue;}
                (false,0x307) if after(&cps,i,|cp|cp==0x49)=>continue,
                _=>{},
            }
        }
        if lithuanian {
            if upper && cp==0x307 && after(&cps,i,soft_dotted) {continue;}
            if !upper {
                let accent=match cp {0xcc=>Some(0x300),0xcd=>Some(0x301),0x128=>Some(0x303),_=>None};
                if let Some(accent)=accent {out.extend([0x69,0x307,accent]);continue;}
                if matches!(cp,0x49|0x4a|0x12e) && more_above(&cps,i) {
                    out.extend([match cp {0x49=>0x69,0x4a=>0x6a,_=>0x12f},0x307]);continue;
                }
            }
        }
        let table=if upper {data::UPPER} else {data::LOWER};
        if let Ok(i)=table.binary_search_by_key(&cp,|p|p.0) {out.extend_from_slice(table[i].1);}
        else if cp<=0xffff {out.push(cp as u16);}
        else {let cp=cp-0x10000;out.extend([0xd800+(cp>>10) as u16,0xdc00+(cp&0x3ff) as u16]);}
    }
    Ok(Val::utf16(out))
}

#[cfg(test)]
mod tests {
    #[test]
    fn final_sigma_matches_bundled_jdk_word_boundaries() {
        for line in include_str!("test_data/locale_boundaries.tsv").lines() {
            let (input,expected)=line.split_once('\t').unwrap();
            let decode=|s:&str|s.split(',').map(|x|u16::from_str_radix(x,16).unwrap()).collect::<Vec<_>>();
            let input=decode(input);
            assert_eq!(super::map(&input,false,&[]).unwrap().string_units(),decode(expected),"{input:?}");
        }
    }
}
