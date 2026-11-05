//! WTF-8 is an internal byte alphabet, not a Unicode string or a sentinel.
//! Every Java code unit has an exact boundary map. A scan starting inside a
//! surrogate pair encodes the two halves separately, as Java Matcher.find(int)
//! may start at the low half. Normal scans keep pairs as one code point.
pub(super) struct Text {
    pub bytes: Vec<u8>,
    boundaries: Vec<usize>,
}
impl Text {
    pub fn new(units: &[u16], split: usize) -> Self {
        let mut bytes=Vec::new();
        let mut boundaries=vec![0];
        let mut i=0;
        while i<units.len() {
            let first=units[i];
            let paired=(0xd800..=0xdbff).contains(&first) && i+1<units.len() && i+1!=split && (0xdc00..=0xdfff).contains(&units[i+1]);
            let (cp,width)=if paired {(0x10000+(((first as u32-0xd800)<<10)|(units[i+1] as u32-0xdc00)),2)} else {(first as u32,1)};
            let start=bytes.len();
            if cp<0x80 {bytes.push(cp as u8);} else if cp<0x800 {bytes.extend([(0xc0|(cp>>6)) as u8,(0x80|(cp&63)) as u8]);}
            else if cp<0x10000 {bytes.extend([(0xe0|(cp>>12)) as u8,(0x80|((cp>>6)&63)) as u8,(0x80|(cp&63)) as u8]);}
            else {bytes.extend([(0xf0|(cp>>18)) as u8,(0x80|((cp>>12)&63)) as u8,(0x80|((cp>>6)&63)) as u8,(0x80|(cp&63)) as u8]);}
            boundaries.resize(bytes.len()+1,i);
            boundaries[start]=i;
            i+=width;
            boundaries[bytes.len()]=i;
        }
        Self {bytes,boundaries}
    }
    pub fn unit(&self, byte:usize)->usize {self.boundaries[byte]}
    pub fn byte(&self, unit:usize)->usize {self.boundaries.iter().position(|u|*u==unit).unwrap_or(self.bytes.len())}
}

const SURROGATE:&str=r"(?-u:\xED[\xA0-\xBF][\x80-\xBF])";

fn hex_escape(chars:&[char],i:usize)->Option<(u32,usize)> {
    if chars.get(i)!=Some(&'\\'){return None;}
    let kind=*chars.get(i+1)?;
    if !matches!(kind,'u'|'x'){return None;}
    if chars.get(i+2)==Some(&'{') {
        let end=chars[i+3..].iter().position(|c|*c=='}')?+i+3;
        let value=u32::from_str_radix(&chars[i+3..end].iter().collect::<String>(),16).ok()?;
        return Some((value,end+1));
    }
    let count=if kind=='u'{4}else{2};let end=i+2+count;
    let value=u32::from_str_radix(&chars.get(i+2..end)?.iter().collect::<String>(),16).ok()?;
    Some((value,end))
}
fn surrogate_atom(first:u16,last:u16)->String {
    let mut alternatives=Vec::new();
    for middle in (first>>6)..=(last>>6) {
        let lo=if middle==first>>6{first&63}else{0};
        let hi=if middle==last>>6{last&63}else{63};
        let tail=if lo==hi{format!(r"\x{:02X}",0x80|lo)}else{format!(r"[\x{:02X}-\x{:02X}]",0x80|lo,0x80|hi)};
        alternatives.push(format!(r"\xED\x{:02X}{tail}",0x80|(middle&63)));
    }
    format!("(?-u:{})",alternatives.join("|"))
}
fn class_parts(class:&[char])->(String,[bool;2048]) {
    let negated=class.get(1)==Some(&'^');let begin=if negated{2}else{1};
    let end=class.len().saturating_sub(1);
    let mut unicode=String::new();let mut units=[false;2048];let mut i=begin;
    while i<end {
        if class[i]=='[' {
            let finish=class_end(class,i);
            let (part,members)=class_parts(&class[i..finish]);
            unicode.push_str(&part);for (a,b) in units.iter_mut().zip(members){*a|=b;}
            i=finish;continue;
        }
        if class[i]=='&'&&class.get(i+1)==Some(&'&') {
            let mut right=vec!['['];right.extend_from_slice(&class[i+2..end]);right.push(']');
            let (part,members)=class_parts(&right);
            for (a,b) in units.iter_mut().zip(members){*a&=b;}
            unicode=format!("{}&&{part}",if unicode.is_empty(){"[a&&b]".into()}else{format!("[{unicode}]")});break;
        }
        let first=hex_escape(class,i).or_else(||(class[i]!='\\').then_some((class[i] as u32,i+1)));
        if let Some((first,next))=first {
            let range=if class.get(next)==Some(&'-')&&next+1<end {hex_escape(class,next+1).or_else(||(class[next+1]!='\\').then_some((class[next+1] as u32,next+2)))}else{None};
            let (last,finish)=range.unwrap_or((first,next));
            if first<=last&&(first<=0xdfff&&last>=0xd800) {
                for unit in first.max(0xd800)..=last.min(0xdfff){units[(unit-0xd800) as usize]=true;}
                for (lo,hi) in [(first,last.min(0xd7ff)),(first.max(0xe000),last)] {
                    if lo<=hi {unicode.push_str(&if lo==hi{format!(r"\x{{{lo:X}}}")}else{format!(r"\x{{{lo:X}}}-\x{{{hi:X}}}")});}
                }
                i=finish;continue;
            }
            unicode.extend(class[i..finish].iter());i=finish;continue;
        }
        if class[i]=='\\'&&i+1<end {
            let code=class[i+1];
            if matches!(code,'p'|'P')&&class.get(i+2)==Some(&'{') {
                if let Some(offset)=class[i+3..end].iter().position(|c|*c=='}') {
                    let finish=i+4+offset;let category:String=class[i+3..finish-1].iter().collect();
                    let surrogate=matches!(category.as_str(),"Cs"|"Surrogate"|"gc=Cs"|"C"|"Other");
                    if surrogate==(code=='p'){units.fill(true);}
                    if matches!(category.as_str(),"Cs"|"Surrogate"|"gc=Cs") {
                        if code=='P'{unicode.push_str(r"\x{0}-\x{D7FF}\x{E000}-\x{10FFFF}");}
                    }else{unicode.extend(class[i..finish].iter());}
                    i=finish;continue;
                }
            }
            if matches!(code,'D'|'W'|'S'){units.fill(true);}
            unicode.push(class[i]);unicode.push(code);i+=2;continue;
        }
        unicode.push(class[i]);i+=1;
    }
    if negated{for u in &mut units{*u=!*u;}}
    let unicode=if unicode.is_empty(){if negated{r"[\x{0}-\x{D7FF}\x{E000}-\x{10FFFF}]".into()}else{r"[a&&b]".into()}}else{format!("[{}{unicode}]",if negated{"^"}else{""})};
    (unicode,units)
}
fn class_end(chars:&[char],start:usize)->usize {
    let mut depth=1;let mut i=start+1;let mut first=true;
    if chars.get(i)==Some(&'^'){i+=1;}
    while i<chars.len(){match chars[i]{'\\'=>{i+=2;first=false;continue;},'['=>depth+=1,']' if !first=>{depth-=1;if depth==0{return i+1;}},_=>{}}first=false;i+=1;}chars.len()
}
fn class_pattern(class:&[char])->String {
    let(unicode,units)=class_parts(class);
    let mut alternatives=vec![unicode];
    let mut start=0;
    while start<units.len(){if !units[start]{start+=1;continue;}let mut end=start;while end+1<units.len()&&units[end+1]{end+=1;}alternatives.push(surrogate_atom(0xd800+start as u16,0xd800+end as u16));start=end+1;}
    format!("(?:{})",alternatives.join("|"))
}

/// Add the Java surrogate category to code-point atoms which Unicode regex
/// necessarily excludes. Literal text remains UTF-8; no user character is
/// reserved or remapped to stand for a surrogate.
pub(super) fn byte_pattern(src:&str)->String {
    let chars:Vec<char>=src.chars().collect();
    let mut out=String::new();let mut i=0;let mut dot_all=false;let mut groups=Vec::new();
    while i<chars.len() {
        let c=chars[i];
        if c=='(' {
            if chars.get(i+1)==Some(&'?') {
                let mut end=i+2;while end<chars.len()&&(chars[end].is_ascii_alphabetic()||chars[end]=='-'){end+=1;}
                if end>i+2&&matches!(chars.get(end),Some(':'|')')) {
                    let previous=dot_all;let mut enabled=true;
                    for flag in &chars[i+2..end]{if *flag=='-'{enabled=false;}else if *flag=='s'{dot_all=enabled;}}
                    if chars[end]==':'{groups.push(previous);}
                    out.extend(chars[i..=end].iter());i=end+1;continue;
                }
            }
            groups.push(dot_all);
        }else if c==')'{if let Some(previous)=groups.pop(){dot_all=previous;}}
        if let Some((unit,next))=hex_escape(&chars,i) {
            if (0xd800..=0xdbff).contains(&unit) {
                if let Some((low,end))=hex_escape(&chars,next) {if (0xdc00..=0xdfff).contains(&low){let cp=0x10000+((unit-0xd800)<<10)+(low-0xdc00);out.push_str(&format!(r"\x{{{cp:X}}}"));i=end;continue;}}
            }
            if (0xd800..=0xdfff).contains(&unit){out.push_str(&surrogate_atom(unit as u16,unit as u16));i=next;continue;}
        }
        if c=='\\' && i+1<chars.len() {
            let next=chars[i+1];
            if matches!(next,'D'|'W'|'S') {out.push_str(&format!("(?:\\{next}|{SURROGATE})"));i+=2;continue;}
            if matches!(next,'p'|'P') && chars.get(i+2)==Some(&'{') {
                if let Some(offset)=chars[i+3..].iter().position(|c|*c=='}') {
                    let end=i+4+offset;let mut class=vec!['['];class.extend_from_slice(&chars[i..end]);class.push(']');
                    out.push_str(&class_pattern(&class));i=end;continue;
                }
            }
            out.push(c);out.push(next);i+=2;continue;
        }
        if c=='[' {
            let start=i;i=class_end(&chars,start);
            out.push_str(&class_pattern(&chars[start..i]));
            continue;
        }
        if c=='.' {let atom=if dot_all{r"(?s:.)"}else{r"[^\n\r\x{85}\x{2028}\x{2029}]"};out.push_str(&format!("(?:{atom}|{SURROGATE})"));} else {out.push(c);}
        i+=1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn surrogate_classes_are_exact_and_do_not_alias_private_use() {
        let input=Text::new(&[0xd800,0xdbff,0xdc00,0xdfff,0xe000],2);
        for (pattern,expected) in [(r"[\uD800-\uDBFF]",vec![0,1]),(r"[^\uD800-\uDBFF]",vec![2,3,4]),(r"[\p{Cs}]",vec![0,1,2,3]),(r"[^\p{Cs}]",vec![4])] {
            let re=regex::bytes::Regex::new(&byte_pattern(pattern)).unwrap();
            let found:Vec<_>=re.find_iter(&input.bytes).map(|m|input.unit(m.start())).collect();
            assert_eq!(found,expected,"{pattern}");
        }
    }
    #[test]
    fn class_intersection_handles_surrogates_and_unicode_independently() {
        let input=Text::new(&[0xd800,65,0xdfff,66],0);
        for(pattern,expected)in [(r"[\p{Cs}&&[^\uD800]]",vec![2]),(r"[A-Z&&[^B]]",vec![1]),(r"[A[\uD800]]",vec![0,1])] {
            let re=regex::bytes::Regex::new(&byte_pattern(pattern)).unwrap();
            let actual:Vec<_>=re.find_iter(&input.bytes).map(|m|input.unit(m.start())).collect();assert_eq!(actual,expected,"{pattern}");
        }
    }
    #[test]
    fn wtf8_has_exact_units_and_no_reserved_characters() {
        let units=[0xd800,65,0xd83d,0xde00,0xe000,0xdfff];
        let text=Text::new(&units,0);
        let re=regex::bytes::Regex::new(&byte_pattern("(.)")).unwrap();
        let ranges:Vec<_>=re.find_iter(&text.bytes).map(|m|(text.unit(m.start()),text.unit(m.end()))).collect();
        assert_eq!(ranges,vec![(0,1),(1,2),(2,4),(4,5),(5,6)]);
        let split=Text::new(&units,3);
        let m=re.find_at(&split.bytes,split.byte(3)).unwrap();
        assert_eq!((split.unit(m.start()),split.unit(m.end())),(3,4));
    }
}
