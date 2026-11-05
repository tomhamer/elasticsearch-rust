//! Immutable recovery telemetry: publishing a snapshot shares its admitted
//! details. A writer forks before mutation; no unowned Vec/string escapes.
use super::FileDetail;
use elasticsearch_engine::rustic_core::admission::{self, Reservation};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use std::{fmt, sync::Arc};

struct Data { values: Vec<FileDetail>, memory: Reservation }
#[derive(Clone, Default)]
pub struct FileDetails(Option<Arc<Data>>);
impl std::ops::Deref for FileDetails {
    type Target = [FileDetail];
    fn deref(&self) -> &[FileDetail] { self.0.as_ref().map_or(&[], |d| d.values.as_slice()) }
}
impl fmt::Debug for FileDetails { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { self.deref_slice().fmt(f) } }
impl PartialEq for FileDetails { fn eq(&self,other:&Self)->bool { self.deref_slice()==other.deref_slice() } }
impl Eq for FileDetails {}
impl Serialize for FileDetails {
    fn serialize<S:Serializer>(&self,s:S)->Result<S::Ok,S::Error> { self.deref_slice().serialize(s) }
}
impl Data {
    fn new()->Result<Self,String> {
        // Includes the Arc's two counters and its payload; allocated only for
        // nonempty telemetry. The reservation is retained after the values.
        let memory=admission::shared().reserve((std::mem::size_of::<Self>()+2*std::mem::size_of::<usize>()) as u64,"recovery details owner").map_err(|e|e.to_string())?;
        Ok(Self{values:Vec::new(),memory})
    }
    fn slot(&mut self)->Result<(),String> {
        if self.values.len()==self.values.capacity() {
            let old=self.values.capacity();
            let next=old.checked_mul(2).unwrap_or(usize::MAX).max(4);
            let bytes=next.checked_mul(std::mem::size_of::<FileDetail>()).ok_or("recovery details size overflow")?;
            self.memory.grow(bytes as u64,"recovery details replacement").map_err(|e|e.to_string())?;
            self.values.try_reserve_exact(next-self.values.len()).map_err(|e|e.to_string())?;
            self.memory.shrink((old*std::mem::size_of::<FileDetail>()) as u64);
        }
        Ok(())
    }
    fn push(&mut self,name:&str,length:u64,recovered:u64,reused:bool)->Result<(),String> {
        self.slot()?;
        let name=copy_text(name,&mut self.memory)?;
        self.values.push(FileDetail{name,length,recovered,reused});Ok(())
    }
}
fn copy_text(s:&str,memory:&mut Reservation)->Result<String,String> {
    memory.grow(s.len() as u64,"recovery detail name").map_err(|e|e.to_string())?;
    let mut out=String::new();out.try_reserve_exact(s.len()).map_err(|e|e.to_string())?;out.push_str(s);Ok(out)
}
impl FileDetails {
    fn deref_slice(&self)->&[FileDetail] { self }
    fn writable(&mut self)->Result<&mut Data,String> {
        if self.0.as_mut().is_some_and(|v|Arc::get_mut(v).is_some()) {
            return Ok(Arc::get_mut(self.0.as_mut().unwrap()).unwrap());
        }
        let mut replacement=Data::new()?;
        for f in self.iter(){replacement.push(&f.name,f.length,f.recovered,f.reused)?;}
        self.0=Some(Arc::new(replacement));
        Ok(Arc::get_mut(self.0.as_mut().unwrap()).unwrap())
    }
    pub fn push(&mut self,name:&str,length:u64,recovered:u64,reused:bool)->Result<(),String> {
        self.writable()?.push(name,length,recovered,reused)
    }
    pub fn set_recovered(&mut self,index:usize,recovered:u64)->Result<(),String> {
        if index>=self.len(){return Err("invalid recovery detail index".into());}
        self.writable()?.values[index].recovered=recovered;Ok(())
    }
}
struct Text<'a>(&'a mut Reservation);
impl<'de> DeserializeSeed<'de> for Text<'_> {
    type Value=String;
    fn deserialize<D:Deserializer<'de>>(self,d:D)->Result<String,D::Error> {
        struct V<'a>(&'a mut Reservation);
        impl<'de> Visitor<'de> for V<'_> {
            type Value=String;
            fn expecting(&self,f:&mut fmt::Formatter)->fmt::Result{f.write_str("a recovery filename")}
            fn visit_str<E:de::Error>(self,s:&str)->Result<String,E>{copy_text(s,self.0).map_err(|e|E::custom(elasticsearch_transport::decode_admission_error(e)))}
        }
        d.deserialize_str(V(self.0))
    }
}
struct Detail<'a>(&'a mut Reservation);
impl<'de> DeserializeSeed<'de> for Detail<'_> {
    type Value=FileDetail;
    fn deserialize<D:Deserializer<'de>>(self,d:D)->Result<FileDetail,D::Error> {
        #[derive(Deserialize)] #[serde(field_identifier,rename_all="snake_case")]
        enum Field {Name,Length,Recovered,Reused,#[serde(other)]Other}
        struct V<'a>(&'a mut Reservation);
        impl<'de> Visitor<'de> for V<'_> {
            type Value=FileDetail;
            fn expecting(&self,f:&mut fmt::Formatter)->fmt::Result{f.write_str("a recovery file detail")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut a:A)->Result<FileDetail,A::Error>{
                Ok(FileDetail{
                    name:a.next_element_seed(Text(self.0))?.ok_or_else(||de::Error::missing_field("name"))?,
                    length:a.next_element()?.ok_or_else(||de::Error::missing_field("length"))?,
                    recovered:a.next_element()?.ok_or_else(||de::Error::missing_field("recovered"))?,
                    reused:a.next_element()?.ok_or_else(||de::Error::missing_field("reused"))?,
                })
            }
            fn visit_map<A:MapAccess<'de>>(self,mut a:A)->Result<FileDetail,A::Error>{
                let(mut name,mut length,mut recovered,mut reused)=(None,None,None,None);
                while let Some(k)=a.next_key::<Field>()? {match k {
                    Field::Name=>{if name.is_some(){return Err(de::Error::duplicate_field("name"));}name=Some(a.next_value_seed(Text(self.0))?);},
                    Field::Length=>{if length.is_some(){return Err(de::Error::duplicate_field("length"));}length=Some(a.next_value()?);},
                    Field::Recovered=>{if recovered.is_some(){return Err(de::Error::duplicate_field("recovered"));}recovered=Some(a.next_value()?);},
                    Field::Reused=>{if reused.is_some(){return Err(de::Error::duplicate_field("reused"));}reused=Some(a.next_value()?);},
                    Field::Other=>{let _:de::IgnoredAny=a.next_value()?;}
                }}
                Ok(FileDetail{name:name.ok_or_else(||de::Error::missing_field("name"))?,length:length.ok_or_else(||de::Error::missing_field("length"))?,recovered:recovered.ok_or_else(||de::Error::missing_field("recovered"))?,reused:reused.ok_or_else(||de::Error::missing_field("reused"))?})
            }
        }
        d.deserialize_struct("FileDetail",&["name","length","recovered","reused"],V(self.0))
    }
}
// The element seed is only invoked when the sequence has another item, so
// an empty sequence needs no owner and EOF never requires spare capacity.
struct Element<'a>(&'a mut Option<Data>);
impl<'de> DeserializeSeed<'de> for Element<'_> {
    type Value=();
    fn deserialize<D:Deserializer<'de>>(self,d:D)->Result<(),D::Error>{
        if self.0.is_none(){*self.0=Some(Data::new().map_err(|e|de::Error::custom(elasticsearch_transport::decode_admission_error(e)))?);}
        let data=self.0.as_mut().unwrap();
        data.slot().map_err(|e|de::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
        let value=Detail(&mut data.memory).deserialize(d)?;
        data.values.push(value);Ok(())
    }
}
impl<'de> Deserialize<'de> for FileDetails {
    fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=FileDetails;
            fn expecting(&self,f:&mut fmt::Formatter)->fmt::Result{f.write_str("recovery file details")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut a:A)->Result<FileDetails,A::Error>{
                let mut data=None;
                while a.next_element_seed(Element(&mut data))?.is_some(){}
                Ok(FileDetails(data.map(Arc::new)))
            }
        }
        d.deserialize_seq(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshots_share_and_numeric_updates_fork_without_changing_old_progress() {
        let mut details=FileDetails::default();
        details.push("core-1/a",90,10,false).unwrap();
        let snapshot=details.clone();
        assert!(Arc::ptr_eq(details.0.as_ref().unwrap(),snapshot.0.as_ref().unwrap()));
        details.set_recovered(0,50).unwrap();
        assert_eq!(snapshot[0].recovered,10);
        assert_eq!(details[0].recovered,50);
        assert!(!Arc::ptr_eq(details.0.as_ref().unwrap(),snapshot.0.as_ref().unwrap()));
        assert_eq!(snapshot[0].name,"core-1/a");
    }
    #[test]
    fn sequence_wire_matches_legacy_details_and_empty_is_allocation_free() {
        let old=vec![FileDetail{name:"core-1/é".into(),length:90,recovered:10,reused:false}];
        let encoded=bincode::serialize(&old).unwrap();
        let decoded:FileDetails=bincode::deserialize(&encoded).unwrap();
        assert_eq!(&*decoded,old.as_slice());
        assert_eq!(bincode::serialize(&decoded).unwrap(),encoded);
        let json=serde_json::to_vec(&old).unwrap();
        let decoded:FileDetails=serde_json::from_slice(&json).unwrap();
        assert_eq!(&*decoded,old.as_slice());
        assert_eq!(serde_json::to_vec(&decoded).unwrap(),json);
        let empty:FileDetails=serde_json::from_str("[]").unwrap();
        assert!(empty.0.is_none());
    }
}
