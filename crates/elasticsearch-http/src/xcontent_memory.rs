//! Fallible storage used by the x-content producers. Every capacity change
//! is admitted before the allocator is called; ownership follows the buffer.
use elasticsearch_engine::{error::EsError, rustic_core::admission::{self, Reservation}};
use std::ops::{Deref, DerefMut};

pub(super) struct Buffer<T> {
    values: Vec<T>,
    memory: Option<Reservation>,
    pub(super) write_error: Option<EsError>,
}
impl<T> Buffer<T> {
    pub(super) fn new() -> Self { Self { values: Vec::new(), memory: None, write_error: None } }
    pub(super) fn reserve(&mut self, additional: usize) -> Result<(), EsError> {
        let needed = self.values.len().checked_add(additional).ok_or_else(overflow)?;
        if needed <= self.values.capacity() && self.memory.is_some() { return Ok(()); }
        let capacity = needed.checked_next_power_of_two().ok_or_else(overflow)?.max(8);
        let bytes = capacity.checked_mul(std::mem::size_of::<T>()).and_then(|n| n.checked_add(512)).ok_or_else(overflow)?;
        // Admit old+new overlap explicitly. Reallocation may temporarily
        // retain both buffers, so charging only the capacity delta is unsafe.
        let memory = match &self.memory {
            Some(owner) => owner.reserve_related(bytes as u64, "x-content replacement buffer")?,
            None => admission::shared().reserve(bytes as u64, "x-content buffer")?,
        };
        let mut replacement = Vec::new();
        replacement.try_reserve_exact(capacity).map_err(|_| overflow())?;
        replacement.append(&mut self.values);
        self.values = replacement;
        self.memory = Some(memory);
        Ok(())
    }
    pub(super) fn push(&mut self, value: T) -> Result<(), EsError> { self.reserve(1)?; self.values.push(value); Ok(()) }
    pub(super) fn extend_from_slice(&mut self, values: &[T]) -> Result<(), EsError> where T: Clone {
        self.reserve(values.len())?; self.values.extend_from_slice(values); Ok(())
    }
    pub(super) fn pop(&mut self) -> Option<T> { self.values.pop() }
    pub(super) fn clear(&mut self) { self.values.clear(); }
    /// Move both parts directly into an owner-bearing result. If further
    /// fallible work is required, perform it first: a later-declared guard
    /// otherwise drops before the earlier-declared Vec on an error return.
    pub(super) fn into_parts(self) -> (Vec<T>, Option<Reservation>) { (self.values, self.memory) }
}
impl<T> Deref for Buffer<T> { type Target = [T]; fn deref(&self) -> &[T] { &self.values } }
impl<T> DerefMut for Buffer<T> { fn deref_mut(&mut self) -> &mut [T] { &mut self.values } }
impl AsRef<[u8]> for Buffer<u8> { fn as_ref(&self) -> &[u8] { &self.values } }
impl std::io::Write for Buffer<u8> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Err(error) = self.extend_from_slice(bytes) {
            self.write_error = Some(error);
            return Err(std::io::Error::other("x-content memory admission"));
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}
pub(super) fn overflow() -> EsError { EsError::from(elasticsearch_engine::rustic_core::RusticError::Memory("x-content allocation capacity exceeded".into())) }

/// JSON writes are sticky-fallible: existing scalar formatting keeps its
/// parser signature, but no denied write allocates and finish returns 429.
pub(super) struct ByteBuffer(Buffer<u8>);
impl ByteBuffer {
    pub(super) fn new() -> Self { Self(Buffer::new()) }
    pub(super) fn push(&mut self, value: u8) { self.extend_from_slice(&[value]); }
    pub(super) fn extend_from_slice(&mut self, values: &[u8]) {
        if self.0.write_error.is_some() { return; }
        if let Err(error) = self.0.extend_from_slice(values) { self.0.write_error = Some(error); }
    }
    pub(super) fn check(&mut self) -> Result<(), EsError> {
        match self.0.write_error.take() { Some(error) => Err(error), None => Ok(()) }
    }
    pub(super) fn finish(mut self) -> Result<bytes::Bytes, EsError> { self.check()?; self.0.reserve(0)?; Ok(bytes::Bytes::from_owner(self.0)) }
}
impl Deref for ByteBuffer { type Target = [u8]; fn deref(&self) -> &[u8] { &self.0 } }
impl std::io::Write for ByteBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.extend_from_slice(bytes);
        if self.0.write_error.is_some() { Err(std::io::Error::other("x-content memory admission")) } else { Ok(bytes.len()) }
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[derive(Clone)]
pub(super) struct Text(std::sync::Arc<admission::Allocation<String>>);
impl Text {
    pub(super) fn utf8(bytes: &[u8]) -> Result<Self, EsError> {
        let length = bytes.utf8_chunks().try_fold(0usize, |n, chunk| n.checked_add(chunk.valid().len()).and_then(|n| n.checked_add(if chunk.invalid().is_empty() { 0 } else { 3 }))).ok_or_else(overflow)?;
        let memory = admission::shared().reserve((length as u64).saturating_add(128), "x-content decoded string")?;
        let mut text = String::new();
        text.try_reserve_exact(length).map_err(|_| overflow())?;
        for chunk in bytes.utf8_chunks() { text.push_str(chunk.valid()); if !chunk.invalid().is_empty() { text.push('\u{fffd}'); } }
        Ok(Self(std::sync::Arc::new(admission::Allocation::new(text, memory))))
    }
    pub(super) fn formatted(arguments: std::fmt::Arguments<'_>) -> Result<Self, EsError> {
        struct Counter(usize);
        impl std::fmt::Write for Counter { fn write_str(&mut self, s: &str) -> std::fmt::Result { self.0 = self.0.checked_add(s.len()).ok_or(std::fmt::Error)?; Ok(()) } }
        let mut count = Counter(0);
        std::fmt::write(&mut count, arguments).map_err(|_| overflow())?;
        let memory = admission::shared().reserve((count.0 as u64).saturating_add(128), "x-content formatted string")?;
        let mut text = String::new();
        text.try_reserve_exact(count.0).map_err(|_| overflow())?;
        struct Fixed<'a> { text: &'a mut String, length: usize }
        impl std::fmt::Write for Fixed<'_> {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                if s.len() > self.length.saturating_sub(self.text.len()) { return Err(std::fmt::Error); }
                self.text.push_str(s); Ok(())
            }
        }
        std::fmt::write(&mut Fixed { text: &mut text, length: count.0 }, arguments).map_err(|_| overflow())?;
        Ok(Self(std::sync::Arc::new(admission::Allocation::new(text, memory))))
    }
}
impl Deref for Text { type Target = str; fn deref(&self) -> &str { &self.0 } }
impl std::fmt::Display for Text { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self) } }
