//! Allocation-free encoding walks, used twice (count, then fixed destination).
//! Smile's bounded name dictionary borrows keys; YAML folding walks UTF-8
//! offsets rather than constructing character vectors or temporary strings.
use super::{Wire, printable};
use serde_json::Value;
use std::io::{self,Write};

struct Out<'a> { w:&'a mut dyn Write, column:usize }
impl Out<'_> {
    fn bytes(&mut self,b:&[u8])->io::Result<()> {
        self.w.write_all(b)?;
        self.column=if let Some(n)=b.iter().rposition(|b|*b==b'\n'){b.len()-n-1}else{self.column.checked_add(b.len()).ok_or_else(||io::Error::other("wire column overflow"))?};
        Ok(())
    }
    fn text(&mut self,s:&str)->io::Result<()>{self.bytes(s.as_bytes())}
    fn byte(&mut self,b:u8)->io::Result<()>{self.bytes(&[b])}
    fn pad(&mut self,mut n:usize)->io::Result<()>{let spaces=[b' ';64];while n>0{let k=n.min(spaces.len());self.bytes(&spaces[..k])?;n-=k;}Ok(())}
}
fn quoted(text:&str,fold:Option<(usize,usize)>,o:&mut Out<'_>)->io::Result<()> {
    o.byte(b'"')?;
    let mut column=fold.map_or(0,|(c,_)|c)+1;
    let total=text.chars().count();
    let(mut start_byte,mut start_char)=(0,0);
    for (end,(offset,ch)) in text.char_indices().map(|(i,c)|(i,Some(c))).chain(std::iter::once((text.len(),None))).enumerate(){
        let special=ch.is_none_or(|c|matches!(c,'"'|'\\'|'\u{85}'|'\u{2028}'|'\u{2029}'|'\u{FEFF}')||!(' '..='~').contains(&c));
        if special {
            if start_char<end{let part=&text[start_byte..offset];o.text(part)?;column+=part.chars().count();start_byte=offset;start_char=end;}
            if let Some(c)=ch {
                let literal=match c {'\0'=>Some("\\0"),'\u{7}'=>Some("\\a"),'\u{8}'=>Some("\\b"),'\t'=>Some("\\t"),'\n'=>Some("\\n"),'\u{b}'=>Some("\\v"),'\u{c}'=>Some("\\f"),'\r'=>Some("\\r"),'\u{1b}'=>Some("\\e"),'"'=>Some("\\\""),'\\'=>Some("\\\\"),'\u{85}'=>Some("\\N"),'\u{a0}'=>Some("\\_"),'\u{2028}'=>Some("\\L"),'\u{2029}'=>Some("\\P"),_=>None};
                if let Some(literal)=literal{o.text(literal)?;column+=literal.len();}
                else if printable(c){let mut bytes=[0;4];o.text(c.encode_utf8(&mut bytes))?;column+=1;}
                else {let mut bytes=[0;12];let mut cursor=io::Cursor::new(bytes.as_mut_slice());if c as u32<=255{write!(&mut cursor,"\\x{:02x}",c as u32)?;}else{write!(&mut cursor,"\\u{:04x}",c as u32)?;}let len=cursor.position() as usize;o.bytes(&bytes[..len])?;column+=len;}
                start_char=end+1;start_byte=offset+c.len_utf8();
            }
        }
        if let Some((_,indent))=fold {
            if end>0 && end+1<total && (ch==Some(' ')||start_char>=end) && column+end.saturating_sub(start_char)>80 {
                if start_char<end{o.text(&text[start_byte..offset])?;start_char=end;start_byte=offset;}
                o.text("\\\n")?;o.pad(indent)?;column=indent;
                if text[start_byte..].starts_with(' '){o.byte(b'\\')?;column+=1;}
            }
        }
    }
    o.byte(b'"')
}
fn key(k:&str,o:&mut Out<'_>)->io::Result<()> {
    let reserved=["true","false","yes","no","on","off","y","n","null","~",""] .iter().any(|s|k.eq_ignore_ascii_case(s));
    let first=k.chars().next().unwrap_or(' ');
    let special="-?:,[]{}#&*!|>'\"%@`".contains(first)||first.is_ascii_digit()||first.is_whitespace()||k.contains(": ")||k.contains(" #")||k.ends_with(':')||k.ends_with(' ');
    if reserved||special{quoted(k,None,o)}else{o.text(k)}
}
fn scalar(v:&Value,o:&mut Out<'_>)->io::Result<()> {
    match v {Value::Null=>o.text("null"),Value::Bool(true)=>o.text("true"),Value::Bool(false)=>o.text("false"),Value::Number(n)=>write!(o.w,"{n}"),Value::String(s)=>quoted(s,None,o),_=>unreachable!()}
}
fn block(v:&Value,indent:usize,o:&mut Out<'_>)->io::Result<()> {
    match v {
        Value::Object(m)=>for(k,v)in m{o.pad(indent)?;key(k,o)?;o.byte(b':')?;child(v,indent,o,false)?;},
        Value::Array(a)=>for v in a{o.pad(indent)?;o.byte(b'-')?;child(v,indent,o,true)?;},_=>{}
    }Ok(())
}
fn child(v:&Value,indent:usize,o:&mut Out<'_>,in_seq:bool)->io::Result<()> {
    match v {
        Value::Object(m)if m.is_empty()=>o.text(" {}\n"),Value::Array(a)if a.is_empty()=>o.text(" []\n"),
        Value::Object(m)=>{if in_seq{for(i,(k,v))in m.iter().enumerate(){if i==0{o.byte(b' ')?;}else{o.pad(indent+2)?;}key(k,o)?;o.byte(b':')?;child(v,indent+2,o,false)?;}}else{o.byte(b'\n')?;block(v,indent+2,o)?;}Ok(())},
        Value::Array(a)=>{if in_seq{for(i,v)in a.iter().enumerate(){if i==0{o.byte(b' ')?;}else{o.pad(indent+2)?;}o.byte(b'-')?;child(v,indent+2,o,true)?;}}else{o.byte(b'\n')?;block(v,indent,o)?;}Ok(())},
        Value::String(s)=>{o.byte(b' ')?;quoted(s,Some((o.column,indent+2)),o)?;o.byte(b'\n')},
        v=>{o.byte(b' ')?;scalar(v,o)?;o.byte(b'\n')}
    }
}
fn yaml(v:&Value,o:&mut Out<'_>)->io::Result<()> {
    o.text("---")?;
    match v {Value::Object(m)if m.is_empty()=>o.text(" {}\n"),Value::Array(a)if a.is_empty()=>o.text(" []\n"),Value::Object(_)|Value::Array(_)=>{o.byte(b'\n')?;block(v,0,o)},v=>{o.byte(b' ')?;scalar(v,o)?;o.byte(b'\n')}}
}
fn head(major:u8,n:u64,o:&mut Out<'_>)->io::Result<()> {
    let m=major<<5;
    if n<24{o.byte(m|n as u8)}else if n<256{o.bytes(&[m|24,n as u8])}else if n<65536{o.byte(m|25)?;o.bytes(&(n as u16).to_be_bytes())}else if n<0x1_0000_0000{o.byte(m|26)?;o.bytes(&(n as u32).to_be_bytes())}else{o.byte(m|27)?;o.bytes(&n.to_be_bytes())}
}
fn cbor(v:&Value,o:&mut Out<'_>)->io::Result<()> {
    match v {Value::Null=>o.byte(0xf6),Value::Bool(b)=>o.byte(if *b{0xf5}else{0xf4}),Value::Number(n)=>{if let Some(i)=n.as_i64(){if i>=0{head(0,i as u64,o)}else{head(1,(-1-i) as u64,o)}}else if let Some(u)=n.as_u64(){head(0,u,o)}else{o.byte(0xfb)?;o.bytes(&n.as_f64().unwrap_or(0.).to_be_bytes())}},Value::String(s)=>{head(3,s.len() as u64,o)?;o.text(s)},Value::Array(a)=>{o.byte(0x9f)?;for v in a{cbor(v,o)?;}o.byte(0xff)},Value::Object(m)=>{o.byte(0xbf)?;for(k,v)in m{head(3,k.len() as u64,o)?;o.text(k)?;cbor(v,o)?;}o.byte(0xff)}}
}
fn vint(mut n:u64,o:&mut Out<'_>)->io::Result<()> {let last=(n&63) as u8|128;n>>=6;let mut groups=[0u8;9];let mut used=0;while n>0{groups[used]=(n&127) as u8;used+=1;n>>=7;}for b in groups[..used].iter().rev(){o.byte(*b)?;}o.byte(last)}
fn smile_str(s:&str,o:&mut Out<'_>)->io::Result<()> {
    let b=s.as_bytes();if b.is_empty(){return o.byte(0x20);}let ascii=b.is_ascii();
    if ascii&&b.len()<=32{o.byte(0x40+(b.len()-1) as u8)?;}else if ascii&&b.len()<=64{o.byte(0x60+(b.len()-33) as u8)?;}else if !ascii&&b.len()<=33{o.byte(0xc0+(b.len()-2) as u8)?;}else if !ascii&&b.len()<=65{o.byte(0xe0+(b.len()-34) as u8)?;}else{o.byte(if ascii{0xe0}else{0xe4})?;o.bytes(b)?;return o.byte(0xfc);}o.bytes(b)
}
struct Names<'a>{values:[Option<&'a str>;1024],len:usize}
fn name<'a>(k:&'a str,o:&mut Out<'_>,names:&mut Names<'a>)->io::Result<()> {
    let b=k.as_bytes();if b.is_empty(){return o.byte(0x20);}if b.len()<=64 {
        if let Some(i)=names.values[..names.len].iter().position(|v|*v==Some(k)){if i<64{return o.byte(0x40+i as u8);}o.byte(0x30+(i>>8) as u8)?;return o.byte((i&255)as u8);}
        if names.len==1024{names.len=0;}names.values[names.len]=Some(k);names.len+=1;
        if b.is_ascii(){o.byte(0x80+(b.len()-1)as u8)?;}else if b.len()>=2&&b.len()<=57{o.byte(0xc0+(b.len()-2)as u8)?;}else{o.byte(0x34)?;o.bytes(b)?;return o.byte(0xfc);}o.bytes(b)
    }else{o.byte(0x34)?;o.bytes(b)?;o.byte(0xfc)}
}
fn smile<'a>(v:&'a Value,o:&mut Out<'_>,names:&mut Names<'a>)->io::Result<()> {
    match v {Value::Null=>o.byte(0x21),Value::Bool(b)=>o.byte(if *b{0x23}else{0x22}),Value::Number(n)=>{if let Some(i)=n.as_i64(){let z=((i<<1)^(i>>63)) as u64;if(-16..=15).contains(&i){o.byte(0xc0+z as u8)}else{o.byte(if i>=i32::MIN as i64&&i<=i32::MAX as i64{0x24}else{0x25})?;vint(z,o)}}else if let Some(u)=n.as_u64(){o.byte(0x25)?;vint(u,o)}else{let bits=n.as_f64().unwrap_or(0.).to_bits();o.byte(0x29)?;o.byte((bits>>63) as u8)?;for shift in (0..9).rev(){o.byte(((bits>>(shift*7))&127)as u8)?;}Ok(())}},Value::String(s)=>smile_str(s,o),Value::Array(a)=>{o.byte(0xf8)?;for v in a{smile(v,o,names)?;}o.byte(0xf9)},Value::Object(m)=>{o.byte(0xfa)?;for(k,v)in m{name(k,o,names)?;smile(v,o,names)?;}o.byte(0xfb)}}
}
pub(super) fn encode(w:Wire,v:&Value,sink:&mut dyn Write)->io::Result<()> {
    let mut out=Out{w:sink,column:0};
    match w {Wire::Yaml=>yaml(v,&mut out),Wire::Cbor=>cbor(v,&mut out),Wire::Smile=>{out.bytes(&[0x3a,0x29,0x0a,0x05])?;smile(v,&mut out,&mut Names{values:[None;1024],len:0})}}
}
