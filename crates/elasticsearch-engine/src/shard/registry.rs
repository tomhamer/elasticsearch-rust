//! Resident keyed metadata. Keys are copied and admitted before sequencing;
//! callers reserve table capacity including every uninstalled batch first.
use std::collections::HashMap;
use rustic_core::admission::Reservation;
use crate::error::{EsError, Result};

fn overflow()->EsError { EsError::new(429,"circuit_breaking_exception","write registry capacity overflow") }
pub(super) struct Key { text:String, memory:Reservation }
impl Key {
    pub(super) fn prepare(text:&str)->Result<Self> {
        let memory=rustic_core::admission::shared().reserve(text.len() as u64,"resident write key")?;
        let mut owned=String::new();
        owned.try_reserve_exact(text.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        owned.push_str(text);
        Ok(Self{text:owned,memory})
    }
}
struct Entry<V> { value:V, _key:Reservation }
/// An entry taken out by [`Registry::extract`], freed when dropped.
pub(super) struct Removed<V>(#[allow(dead_code)] String, #[allow(dead_code)] Entry<V>);
pub(super) struct Registry<V> { entries:HashMap<String,Entry<V>,ahash::RandomState>, memory:Option<Reservation> }
impl<V> Default for Registry<V> { fn default()->Self {Self{entries:HashMap::default(),memory:None}} }
impl<V> Registry<V> {
    pub(super) fn len(&self)->usize {self.entries.len()}
    pub(super) fn get(&self,key:&str)->Option<&V> {self.entries.get(key).map(|e|&e.value)}
    pub(super) fn contains_key(&self,key:&str)->bool {self.entries.contains_key(key)}
    pub(super) fn reserve_capacity(&mut self,required:usize)->Result<()> {
        if required<=self.entries.capacity(){return Ok(())}
        let buckets=required.checked_next_power_of_two().and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?;
        let bytes=buckets.checked_mul(std::mem::size_of::<(String,Entry<V>)>()+1).and_then(|n|n.checked_add(32)).ok_or_else(overflow)?;
        let memory=rustic_core::admission::shared().reserve(bytes as u64,"resident write registry table")?;
        self.entries.try_reserve(required-self.entries.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        self.memory=Some(memory);
        Ok(())
    }
    pub(super) fn insert(&mut self,key:Key,value:V)->Option<V> {
        if let Some(existing)=self.entries.get_mut(key.text.as_str()) {
            // HashMap preserves its original key allocation on replacement;
            // preserve that exact key's guard too.
            return Some(std::mem::replace(&mut existing.value,value));
        }
        assert!(self.entries.len()<self.entries.capacity(),"write table slots must be admitted before sequencing");
        let Key{text,memory}=key;
        self.entries.insert(text,Entry{value,_key:memory});
        None
    }
    pub(super) fn replace_existing(&mut self,key:&str,value:V) {
        self.entries.get_mut(key).expect("sequenced key remains pinned until rollback").value=value;
    }
    pub(super) fn remove(&mut self,key:&str)->Option<V> { self.entries.remove(key).map(|entry|entry.value) }
    pub(super) fn retain(&mut self,mut keep:impl FnMut(&String,&V)->bool) { self.entries.retain(|key,entry|keep(key,&entry.value)); }
    /// Take out the entries `keep` rejects, for the caller to free when
    /// convenient (their keys and memory guards).
    pub(super) fn extract(&mut self,mut keep:impl FnMut(&String,&V)->bool)->Vec<Removed<V>> {
        self.entries.extract_if(|key,entry|!keep(key,&entry.value)).map(|(k,e)|Removed(k,e)).collect()
    }
}
impl<V> std::ops::Index<&str> for Registry<V> {type Output=V;fn index(&self,key:&str)->&V {self.get(key).expect("registry key")}}

/// Own the temporary key container through all per-op moves/refusals.
pub(super) struct Keys { keys:std::vec::IntoIter<Key>, _memory:Reservation }
impl Keys {
    pub(super) fn prepare<'a>(values:impl ExactSizeIterator<Item=&'a str>)->Result<Self> {
        let count=values.len();
        let bytes=count.checked_mul(std::mem::size_of::<Key>()).ok_or_else(overflow)?;
        let memory=rustic_core::admission::shared().reserve(bytes as u64,"prepared registry key slots")?;
        let mut keys=Vec::new();keys.try_reserve_exact(count).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        for value in values {keys.push(Key::prepare(value)?);}
        Ok(Self{keys:keys.into_iter(),_memory:memory})
    }
}
impl Iterator for Keys {type Item=Key;fn next(&mut self)->Option<Key>{self.keys.next()}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replacement_preserves_original_key_owner_and_retained_table() {
        let owner=rustic_core::admission::Owner::new();
        let _scope=owner.enter();
        let mut registry=Registry::default();
        registry.reserve_capacity(2).unwrap();
        let table=owner.bytes().current;
        registry.insert(Key::prepare("original-key").unwrap(),1);
        let retained=owner.bytes().current;
        assert!(retained>table);
        assert_eq!(registry.insert(Key::prepare("original-key").unwrap(),2),Some(1));
        assert_eq!(owner.bytes().current,retained);
        assert_eq!(registry.remove("original-key"),Some(2));
        assert_eq!(owner.bytes().current,table,"empty table still retains backing capacity");
        drop(registry);
        assert_eq!(owner.bytes().current,0);
    }
    #[test]
    fn refused_capacity_does_not_mutate_existing_registry() {
        let mut registry=Registry::default();
        registry.reserve_capacity(1).unwrap();
        registry.insert(Key::prepare("key").unwrap(),7);
        assert!(registry.reserve_capacity(usize::MAX).is_err());
        assert_eq!(registry.get("key"),Some(&7));
        assert_eq!(registry.len(),1);
    }
}

/// A buffered delete shares one immutable key across its writer and refresh
/// snapshots. The B-tree node allowance is acquired before sequencing too.
#[derive(Clone)]
pub(super) struct DeleteKey(std::sync::Arc<rustic_core::admission::Allocation<String>>);
impl DeleteKey {
    pub(super) fn prepare(text:&str)->Result<Self> {
        let bytes=text.len().checked_add(std::mem::size_of::<String>()+2*std::mem::size_of::<usize>())
            .and_then(|n|n.checked_add(512)).ok_or_else(overflow)?;
        let memory=rustic_core::admission::shared().reserve(bytes as u64,"buffered delete key and node")?;
        let mut key=String::new();key.try_reserve_exact(text.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        key.push_str(text);
        Ok(Self(std::sync::Arc::new(rustic_core::admission::Allocation::new(key,memory))))
    }
}
impl std::ops::Deref for DeleteKey {type Target=str;fn deref(&self)->&str{self.0.as_str()}}
impl std::borrow::Borrow<str> for DeleteKey {fn borrow(&self)->&str{self}}
impl PartialEq for DeleteKey {fn eq(&self,other:&Self)->bool{**self==**other}}
impl Eq for DeleteKey {}
impl PartialOrd for DeleteKey {fn partial_cmp(&self,other:&Self)->Option<std::cmp::Ordering>{Some(self.cmp(other))}}
impl Ord for DeleteKey {fn cmp(&self,other:&Self)->std::cmp::Ordering{self.as_ref().cmp(other.as_ref())}}
impl AsRef<str> for DeleteKey {fn as_ref(&self)->&str{self}}
pub(super) type Deletes=std::collections::BTreeMap<DeleteKey,i64>;

/// Prepared container remains admitted while individual keys move into writers.
pub(super) struct DeleteKeys { keys:std::vec::IntoIter<DeleteKey>, _memory:Reservation }
impl DeleteKeys {
 pub(super) fn prepare<'a>(values:impl ExactSizeIterator<Item=&'a str>)->Result<Self> {
  let count=values.len();
  let bytes=count.checked_mul(std::mem::size_of::<DeleteKey>()).ok_or_else(overflow)?;
  let memory=rustic_core::admission::shared().reserve(bytes as u64,"prepared delete key slots")?;
  let mut keys=Vec::new();keys.try_reserve_exact(count).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
  for value in values { keys.push(DeleteKey::prepare(value)?); }
  Ok(Self{keys:keys.into_iter(),_memory:memory})
 }
}
impl Iterator for DeleteKeys {type Item=DeleteKey;fn next(&mut self)->Option<DeleteKey>{self.keys.next()}}
pub(super) fn delete_window(source:&Deletes)->Result<rustic_core::admission::Allocation<Deletes>> {
 let bytes=source.len().checked_mul(512).ok_or_else(overflow)?;
 let memory=rustic_core::admission::shared().reserve(bytes as u64,"sorted delete window nodes")?;
 Ok(rustic_core::admission::Allocation::new(source.clone(),memory))
}
pub(super) fn delete_snapshot(source:&Deletes)->Result<rustic_core::admission::Allocation<Vec<(DeleteKey,i64)>>> {
 let bytes=source.len().checked_mul(std::mem::size_of::<(DeleteKey,i64)>()).ok_or_else(overflow)?;
 let memory=rustic_core::admission::shared().reserve(bytes as u64,"delete resolver snapshot")?;
 let mut entries=Vec::new();entries.try_reserve_exact(source.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
 entries.extend(source.iter().map(|(key,seq)|(key.clone(),*seq)));
 Ok(rustic_core::admission::Allocation::new(entries,memory))
}

#[cfg(test)]
mod delete_tests {
 use super::*;
 #[test]
 fn delete_windows_own_nodes_independently_and_share_admitted_key() {
  let owner=rustic_core::admission::Owner::new();let _scope=owner.enter();
  let key=DeleteKey::prepare("kept").unwrap();let key_bytes=owner.bytes().current;
  let mut source=Deletes::new();source.insert(key.clone(),7);
  let window=delete_window(&source).unwrap();
  assert_eq!(owner.bytes().current,key_bytes+512);
  let snapshot=delete_snapshot(&window).unwrap();
  assert_eq!(snapshot[0].0.as_ref(),"kept");
  drop(source);drop(key);drop(window);
  assert_eq!(owner.bytes().current,key_bytes+std::mem::size_of::<(DeleteKey,i64)>() as u64);
  drop(snapshot);assert_eq!(owner.bytes().current,0);
 }
}
