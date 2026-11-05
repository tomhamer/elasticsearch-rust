//! Ownership-carrying translog/replication containers. No unowned consuming
//! iterator or mutable Vec escape: allocation ownership follows moved values.
use std::sync::Arc;
use crate::{error::{EsError, Result}, memory::{Breakers, Reservation}};

pub struct AdmittedBatch<T> {
    values: Vec<T>,
    owners: Vec<Arc<Reservation>>,
    slots: Reservation,
}
impl<T: std::fmt::Debug> std::fmt::Debug for AdmittedBatch<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.values.fmt(f) }
}
impl<T> std::ops::Deref for AdmittedBatch<T> {
    type Target = [T];
    fn deref(&self) -> &[T] { &self.values }
}
impl<T: serde::Serialize> serde::Serialize for AdmittedBatch<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.values, serializer)
    }
}
impl<T: PartialEq> PartialEq<Vec<T>> for AdmittedBatch<T> {
    fn eq(&self, other: &Vec<T>) -> bool { self.values == *other }
}
impl<T: PartialEq> PartialEq<[T]> for AdmittedBatch<T> {
    fn eq(&self, other: &[T]) -> bool { self.values == other }
}
impl<T: PartialEq> PartialEq for AdmittedBatch<T> {
    fn eq(&self, other: &Self) -> bool { self.values == other.values }
}
impl<T: Eq> Eq for AdmittedBatch<T> {}
impl<'a, T> IntoIterator for &'a AdmittedBatch<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter { self.values.iter() }
}
fn refused(reason: impl std::fmt::Display) -> EsError {
    EsError::new(429, "circuit_breaking_exception", format!("admitted batch allocation refused: {reason}"))
}
fn reserve_vec<T>(values: &mut Vec<T>, required: usize, guard: &mut Reservation) -> Result<()> {
    if required <= values.capacity() { return Ok(()); }
    let capacity = required.max(values.capacity().saturating_mul(2)).max(4);
    let bytes = capacity.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| refused("capacity overflow"))? as u64;
    // Charge the entire replacement while the old allocation remains alive.
    // Conservatively retain prior slot charges after replacement; no owner
    // currently retaining an older allocation can be released prematurely.
    guard.grow(bytes, "admitted batch replacement slots")?;
    if let Err(error) = values.try_reserve_exact(capacity - values.len()) {
        guard.shrink(bytes);
        return Err(refused(error));
    }
    Ok(())
}
impl<T> AdmittedBatch<T> {
    /// Remove values without allocating or prematurely releasing a shared owner.
    pub fn retain(&mut self, predicate: impl FnMut(&T) -> bool) { self.values.retain(predicate); }
    /// Synchronous terminal consumer. Payload may be moved into a downstream
    /// subsystem only if that subsystem supplies its own retained admission.
    /// This batch's owners survive every callback and unwind/error cleanup.
    pub fn try_for_each(self, mut consume: impl FnMut(T) -> Result<()>) -> Result<()> {
        let Self { values, owners, slots } = self;
        let _ownership = (owners, slots);
        for value in values { consume(value)?; }
        Ok(())
    }
    pub fn capacity(&self) -> usize { self.values.capacity() }
    /// Transfer a vector produced under `memory` without allocating or copying.
    /// The producer must reserve its full vector capacity and all owned payload
    /// capacities *before* constructing them. This cannot retroactively admit
    /// an existing uncharged allocation. The transferred guard is mandatory and
    /// stays live through every subsequent move/conversion of this batch.
    pub fn from_reserved(values: Vec<T>, memory: Reservation) -> Self {
        assert!(memory.bytes() >= values.capacity().saturating_mul(std::mem::size_of::<T>()) as u64,
            "admitted batch reservation does not cover vector capacity");
        Self { values, owners: Vec::new(), slots: memory }
    }
    pub fn new(breakers: &'static Breakers) -> Result<Self> {
        Ok(Self { values: Vec::new(), owners: Vec::new(), slots: breakers.reserve_request(0, "admitted batch slots")? })
    }
    /// Must succeed before a tailer marks or removes its next pending op.
    pub fn reserve_one(&mut self) -> Result<()> {
        let values = self.values.len().checked_add(1).ok_or_else(|| refused("length overflow"))?;
        let owners = self.owners.len().checked_add(1).ok_or_else(|| refused("owner length overflow"))?;
        reserve_vec(&mut self.values, values, &mut self.slots)?;
        reserve_vec(&mut self.owners, owners, &mut self.slots)
    }
    /// Attach data already produced under this reservation. Refusal returns
    /// both unchanged, so a consumer can retain/requeue rather than lose data.
    pub fn push_reserved(&mut self, value: T, memory: Reservation) -> std::result::Result<(), (EsError, T, Reservation)> {
        if let Err(error) = self.reserve_one().and_then(|_| self.slots.grow((std::mem::size_of::<Reservation>() + 2 * std::mem::size_of::<usize>()) as u64, "admitted batch owner")) {
            return Err((error, value, memory));
        }
        self.values.push(value);
        self.owners.push(Arc::new(memory));
        Ok(())
    }
    /// No allocation: reserve_one must have succeeded before consuming input.
    pub fn push_shared_reserved(&mut self, value: T, owner: Arc<Reservation>) {
        assert!(self.values.len() < self.values.capacity() && self.owners.len() < self.owners.capacity());
        self.values.push(value);
        self.owners.push(owner);
    }
    /// Convert by moving existing allocations. This is not a deep-copy API.
    /// Destination allocation refusal returns the input batch intact.
    pub fn try_map<U>(mut self, breakers: &'static Breakers, mut map: impl FnMut(T) -> U)
        -> std::result::Result<AdmittedBatch<U>, (EsError, Self)> {
        let mut out = match AdmittedBatch::new(breakers) { Ok(v) => v, Err(e) => return Err((e, self)) };
        let owner_count = match self.owners.len().checked_add(1) { Some(n) => n, None => return Err((refused("owner length overflow"), self)) };
        if let Err(error) = reserve_vec(&mut out.values, self.values.len(), &mut out.slots)
            .and_then(|_| reserve_vec(&mut self.owners, owner_count, &mut self.slots))
            .and_then(|_| self.slots.grow((std::mem::size_of::<Reservation>() + 2 * std::mem::size_of::<usize>()) as u64, "admitted batch conversion owner")) {
            return Err((error, self));
        }
        let Self { values, mut owners, slots } = self;
        owners.push(Arc::new(slots));
        out.owners = owners;
        for value in values { out.values.push(map(value)); }
        Ok(out)
    }
    /// Append without dropping either input on refusal; callers can requeue
    /// the returned batch. The old allocation charges remain with the result.
    pub fn try_append(&mut self, mut other: Self) -> std::result::Result<(), (EsError, Self)> {
        let required = match self.values.len().checked_add(other.values.len()) { Some(n) => n, None => return Err((refused("append overflow"), other)) };
        let owners = match self.owners.len().checked_add(other.owners.len()).and_then(|n|n.checked_add(1)) { Some(n) => n, None => return Err((refused("owner append overflow"), other)) };
        if let Err(error) = reserve_vec(&mut self.values, required, &mut self.slots)
            .and_then(|_| reserve_vec(&mut self.owners, owners, &mut self.slots))
            .and_then(|_| self.slots.grow((std::mem::size_of::<Reservation>() + 2 * std::mem::size_of::<usize>()) as u64, "admitted appended owner")) {
            return Err((error, other));
        }
        self.values.append(&mut other.values);
        self.owners.append(&mut other.owners);
        self.owners.push(Arc::new(other.slots));
        Ok(())
    }
    pub fn try_stable_sort_by_key<K: Ord>(&mut self, breakers: &'static Breakers, key: impl FnMut(&T)->K)->Result<()> {
        let bytes=self.values.len().checked_mul(std::mem::size_of::<T>()).ok_or_else(||refused("sort workspace overflow"))?;
        let _memory=breakers.reserve_request(bytes as u64,"admitted stable sort workspace")?;
        self.values.sort_by_key(key);Ok(())
    }
    pub fn dedup_by_key<K: PartialEq>(&mut self, key: impl FnMut(&mut T)->K) { self.values.dedup_by_key(key); }
    pub fn sort_by_key<K: Ord>(&mut self, f: impl FnMut(&T) -> K) { self.values.sort_unstable_by_key(f); }
}
