//! Optional caller-owned admission for collected encoded blob bodies.
//! Socket/TLS buffers belong to the HTTP client; this owns the application
//! accumulation buffer, including old/new replacement overlap.
use super::*;
pub type ReadAdmission = Arc<dyn Fn(usize) -> Result<Box<dyn Send + Sync>> + Send + Sync>;
pub(crate) struct Buffer {
    bytes: Vec<u8>,
    owner: Option<Box<dyn Send + Sync>>,
    admission: ReadAdmission,
}
impl AsRef<[u8]> for Buffer { fn as_ref(&self) -> &[u8] { &self.bytes } }
impl Buffer {
    pub(crate) fn new(admission: ReadAdmission) -> Self { Self { bytes: Vec::new(), owner: None, admission } }
    pub(crate) fn extend(&mut self, bytes: &[u8]) -> Result<()> {
        let need = self.bytes.len().checked_add(bytes.len()).ok_or_else(|| SnapshotError::new("circuit_breaking_exception",429,"blob size overflow"))?;
        if need > self.bytes.capacity() {
            let capacity = need.max(self.bytes.capacity().saturating_mul(2));
            let owner = (self.admission)(capacity)?;
            let mut next = Vec::new();
            next.try_reserve_exact(capacity).map_err(|e| SnapshotError::new("circuit_breaking_exception",429,format!("blob allocation refused: {e}")))?;
            next.extend_from_slice(&self.bytes);
            let old = std::mem::replace(&mut self.bytes, next);
            drop(old); // old allocation dies before its reservation
            self.owner = Some(owner);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    pub(crate) fn finish(self) -> Bytes { Bytes::from_owner(self) }
}
pub(crate) async fn response(mut response: reqwest::Response, admission: ReadAdmission) -> Result<Bytes> {
    let mut out = Buffer::new(admission);
    while let Some(chunk) = response.chunk().await.map_err(|e| SnapshotError::io("reading admitted blob",e))? { out.extend(&chunk)?; }
    Ok(out.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Held { bytes: usize, used: Arc<AtomicUsize> }
    impl Drop for Held { fn drop(&mut self) { self.used.fetch_sub(self.bytes, Ordering::SeqCst); } }
    #[test]
    fn replacement_is_admitted_before_growth_and_owner_follows_bytes_clones() {
        let used = Arc::new(AtomicUsize::new(0));
        let limit = Arc::new(AtomicUsize::new(6));
        let admission: ReadAdmission = {
            let used = used.clone(); let limit = limit.clone();
            Arc::new(move |bytes| {
                let old = used.load(Ordering::SeqCst);
                if old + bytes > limit.load(Ordering::SeqCst) { return Err(SnapshotError::new("circuit_breaking_exception",429,"test budget")); }
                used.fetch_add(bytes, Ordering::SeqCst);
                Ok(Box::new(Held { bytes, used: used.clone() }) as Box<dyn Send + Sync>)
            })
        };
        let mut out = Buffer::new(admission);
        out.extend(b"abcd").unwrap();
        assert_eq!(used.load(Ordering::SeqCst),4);
        assert_eq!(out.extend(b"e").unwrap_err().status,429);
        assert_eq!(out.as_ref(),b"abcd");
        assert_eq!(used.load(Ordering::SeqCst),4);
        limit.store(12,Ordering::SeqCst);
        out.extend(b"e").unwrap();
        assert_eq!(used.load(Ordering::SeqCst),8);
        let bytes=out.finish(); let slice=bytes.slice(1..3); drop(bytes);
        assert_eq!(&slice[..],b"bc");
        assert_eq!(used.load(Ordering::SeqCst),8);
        drop(slice);
        assert_eq!(used.load(Ordering::SeqCst),0);
    }
}
