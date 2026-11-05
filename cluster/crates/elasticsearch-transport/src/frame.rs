//! Wire framing.
//!
//! Every message is one frame:
//!
//! ```text
//! u32 BE   length of everything below
//! u8       protocol version (1)
//! u8       flags (FLAG_*)
//! u64 BE   request id (a response echoes its request's id)
//! u16 BE   action name length (0 on responses), then the UTF-8 action name
//! ...      payload (lz4 block with u32 LE uncompressed-size prefix when
//!          FLAG_COMPRESSED, else raw)
//! ```
//!
//! Payloads are opaque bytes (bincode for the typed helpers). Compression is
//! decided per frame: only above the threshold, and only if it actually
//! shrinks the payload.

use crate::error::TransportError;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use tokio::io::{AsyncRead, AsyncReadExt};

/// Injected by the embedding node; the transport crate has no engine dependency.
/// The returned owner is retained by every clone/slice of the payload bytes.
pub type FrameAdmission = std::sync::Arc<dyn Fn(usize) -> Result<Box<dyn Send + Sync>, crate::RemoteError> + Send + Sync>;

struct AdmittedBytes {
    bytes: Bytes,
    _reservation: Option<Box<dyn Send + Sync>>,
}
impl AsRef<[u8]> for AdmittedBytes {
    fn as_ref(&self) -> &[u8] { &self.bytes }
}

fn reserve(admission: Option<&FrameAdmission>, bytes: usize) -> std::io::Result<Option<Box<dyn Send + Sync>>> {
    admission.map(|admit| admit(bytes)).transpose().map_err(std::io::Error::other)
}
fn own(bytes: Bytes, reservation: Option<Box<dyn Send + Sync>>) -> Bytes {
    Bytes::from_owner(AdmittedBytes { bytes, _reservation: reservation })
}

/// Size without an output allocation, admit, then serialize into a fixed slice.
/// A stateful serializer cannot grow past its admitted sizing pass.
pub(crate) fn serialize_admitted<T: serde::Serialize + ?Sized>(value: &T, admission: Option<&FrameAdmission>) -> Result<Bytes, TransportError> {
    let size = bincode::serialized_size(value).map_err(|e| TransportError::Serialization(e.to_string()))?;
    let size = usize::try_from(size).map_err(|_| TransportError::Serialization("serialized size overflow".into()))?;
    if size > u32::MAX as usize {
        return Err(TransportError::FrameTooLarge { size, max: u32::MAX as usize });
    }
    let reservation = admission.map(|admit| admit(size)).transpose().map_err(TransportError::Remote)?;
    let mut data = Vec::new();
    data.try_reserve_exact(size).map_err(|e| TransportError::Serialization(format!("serialization allocation failed: {e}")))?;
    data.resize(size, 0);
    let mut writer = std::io::Cursor::new(data.as_mut_slice());
    bincode::serialize_into(&mut writer, value).map_err(|e| TransportError::Serialization(e.to_string()))?;
    if writer.position() != size as u64 {
        return Err(TransportError::Serialization("serialized size changed between passes".into()));
    }
    Ok(own(Bytes::from(data), reservation))
}

pub(crate) const PROTOCOL_VERSION: u8 = 1;

pub(crate) const FLAG_REQUEST: u8 = 0x01;
pub(crate) const FLAG_ERROR: u8 = 0x02;
pub(crate) const FLAG_HANDSHAKE: u8 = 0x04;
pub(crate) const FLAG_PING: u8 = 0x08;
pub(crate) const FLAG_COMPRESSED: u8 = 0x10;

/// Bytes between the length prefix and the action name.
const FIXED_HEADER: usize = 1 + 1 + 8 + 2;

/// Internal action names (ES uses the same for its handshake).
pub(crate) const HANDSHAKE_ACTION: &str = "internal:tcp/handshake";
pub(crate) const PING_ACTION: &str = "internal:tcp/ping";

/// A decoded frame (payload already decompressed).
#[derive(Debug)]
pub(crate) struct Frame {
    pub flags: u8,
    pub request_id: u64,
    pub action: String,
    pub payload: Bytes,
}

impl Frame {
    pub fn is_request(&self) -> bool {
        self.flags & FLAG_REQUEST != 0
    }
    pub fn has(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }
}

/// Actions whose payloads are raw indexing data (ES
/// `RawIndexingDataTransportRequest`): the ones `transport.compress:
/// indexing_data` (the default) compresses.
pub fn indexing_data(action: &str) -> bool {
    action.starts_with("indices:data/write/") || action == "internal:index/seq_no/resync" || action.starts_with("internal:index/shard/recovery/") && action.ends_with("batch")
}

/// Encodes one frame, compressing the payload when `compress_threshold` is
/// `Some(t)` and the payload is at least `t` bytes.
pub(crate) fn encode(
    flags: u8,
    request_id: u64,
    action: &str,
    payload: &[u8],
    compress_threshold: Option<usize>,
    max_frame_size: usize,
) -> Result<Bytes, TransportError> {
    encode_admitted(flags, request_id, action, payload, compress_threshold, max_frame_size, None)
}

pub(crate) fn encode_admitted(flags: u8, request_id: u64, action: &str, payload: &[u8], compress_threshold: Option<usize>, max_frame_size: usize, admission: Option<&FrameAdmission>) -> Result<Bytes, TransportError> {
    if action.len() > u16::MAX as usize {
        return Err(TransportError::Serialization("action name too long".into()));
    }
    // Reject impossible uncompressed lengths before lz4's u32 size prefix or
    // maximum-output-size arithmetic. Compressed wire size is checked below.
    if payload.len() > u32::MAX as usize {
        return Err(TransportError::FrameTooLarge { size: payload.len(), max: u32::MAX as usize });
    }
    let mut compression_reservation = None;
    let compressed = match compress_threshold {
        Some(t) if payload.len() >= t => {
            let capacity = payload.len().checked_mul(110).and_then(|n| (n / 100).checked_add(24))
                .ok_or_else(|| TransportError::Serialization("compression size overflow".into()))?;
            // lz4_flex 0.11's largest temporary dictionary is 4096 u32 entries.
            compression_reservation = admission.map(|admit| admit(capacity + 16 * 1024)).transpose().map_err(TransportError::Remote)?;
            let mut c = Vec::new();
            c.try_reserve_exact(capacity).map_err(|e| TransportError::Serialization(format!("compression allocation failed: {e}")))?;
            c.resize(capacity, 0);
            c[..4].copy_from_slice(&(payload.len() as u32).to_le_bytes());
            let written = lz4_flex::block::compress_into(payload, &mut c[4..])
                .map_err(|e| TransportError::Serialization(e.to_string()))?;
            c.truncate(written + 4); // no shrink/reallocation while both buffers live
            (c.len() < payload.len()).then_some(c)
        }
        _ => None,
    };
    let (flags, body): (u8, &[u8]) = match &compressed {
        Some(c) => (flags | FLAG_COMPRESSED, c),
        None => (flags & !FLAG_COMPRESSED, payload),
    };
    if action.len() > u16::MAX as usize {
        return Err(TransportError::Serialization(format!("action name too long: [{action}]")));
    }
    let len = FIXED_HEADER + action.len() + body.len();
    if len > max_frame_size || len > u32::MAX as usize {
        return Err(TransportError::FrameTooLarge { size: len, max: max_frame_size });
    }
    let reservation = admission.map(|admit| admit(4 + len)).transpose().map_err(TransportError::Remote)?;
    let mut buf = BytesMut::with_capacity(4 + len);
    buf.put_u32(len as u32);
    buf.put_u8(PROTOCOL_VERSION);
    buf.put_u8(flags);
    buf.put_u64(request_id);
    buf.put_u16(action.len() as u16);
    buf.put_slice(action.as_bytes());
    buf.put_slice(body);
    let result = own(buf.freeze(), reservation);
    drop(compressed);
    drop(compression_reservation);
    Ok(result)
}

/// Reads one frame. `Ok(None)` is a clean EOF between frames; any protocol
/// violation (oversized frame, bad version, corrupt lz4) is an error, after
/// which the connection must be closed.
#[cfg(test)]
pub(crate) async fn read_frame<R: AsyncRead + Unpin>(rd: &mut R, max_frame_size: usize) -> std::io::Result<Option<(Frame, usize)>> {
    read_frame_admitted(rd, max_frame_size, None).await
}

pub(crate) async fn read_frame_admitted<R: AsyncRead + Unpin>(rd: &mut R, max_frame_size: usize, admission: Option<&FrameAdmission>) -> std::io::Result<Option<(Frame, usize)>> {
    let len = match rd.read_u32().await {
        Ok(n) => n as usize,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    };
    if len > max_frame_size {
        return Err(invalid(format!("frame of [{len}] bytes exceeds the maximum frame size of [{max_frame_size}] bytes")));
    }
    if len < FIXED_HEADER {
        return Err(invalid(format!("frame of [{len}] bytes is shorter than the header")));
    }
    let reservation = reserve(admission, len)?;
    let mut buf = BytesMut::zeroed(len);
    rd.read_exact(&mut buf).await?;
    let mut buf = own(buf.freeze(), reservation);
    let version = buf.get_u8();
    if version != PROTOCOL_VERSION {
        return Err(invalid(format!("unsupported transport protocol version [{version}]")));
    }
    let flags = buf.get_u8();
    let request_id = buf.get_u64();
    let action_len = buf.get_u16() as usize;
    if buf.remaining() < action_len {
        return Err(invalid("truncated action name".into()));
    }
    let action = String::from_utf8(buf.split_to(action_len).to_vec()).map_err(|_| invalid("action name is not UTF-8".into()))?;
    let payload = if flags & FLAG_COMPRESSED != 0 {
        // Guard against decompression bombs before allocating.
        if buf.len() < 4 {
            return Err(invalid("truncated compressed payload".into()));
        }
        let raw = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        if raw > max_frame_size {
            return Err(invalid(format!("decompressed payload of [{raw}] bytes exceeds the maximum frame size of [{max_frame_size}] bytes")));
        }
        // The encoded owner remains charged while decompression allocates.
        let decoded = reserve(admission, raw)?;
        own(Bytes::from(lz4_flex::decompress_size_prepended(&buf).map_err(|e| invalid(format!("corrupt lz4 payload: {e}")))?), decoded)
    } else {
        buf
    };
    Ok(Some((Frame { flags, request_id, action, payload }, 4 + len)))
}

fn invalid(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn roundtrip_plain_and_compressed() {
        let payload = vec![7u8; 100_000];
        for threshold in [None, Some(10)] {
            let f = encode(FLAG_REQUEST, 42, "a:b/c", &payload, threshold, 1 << 20).unwrap();
            if threshold.is_some() {
                assert!(f.len() < 10_000, "should compress");
            }
            let mut rd = &f[..];
            let (frame, n) = read_frame(&mut rd, 1 << 20).await.unwrap().unwrap();
            assert_eq!(n, f.len());
            assert_eq!(frame.request_id, 42);
            assert_eq!(frame.action, "a:b/c");
            assert!(frame.is_request());
            assert_eq!(&frame.payload[..], &payload[..]);
        }
    }

    fn accounting(limit: usize) -> (FrameAdmission, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        struct Held(Arc<AtomicUsize>, usize);
        impl Drop for Held { fn drop(&mut self) { self.0.fetch_sub(self.1, Ordering::SeqCst); } }
        let used = Arc::new(AtomicUsize::new(0));
        let gauge = used.clone();
        let admit: FrameAdmission = Arc::new(move |bytes| {
            gauge.fetch_update(Ordering::SeqCst, Ordering::SeqCst,
                |current| current.checked_add(bytes).filter(|next| *next <= limit))
                .map_err(|_| crate::RemoteError::new("circuit_breaking_exception", "frame budget", 429))?;
            Ok(Box::new(Held(gauge.clone(), bytes)))
        });
        (admit, used)
    }

    #[test]
    fn typed_serialization_is_admitted_and_retained() {
        use std::sync::atomic::Ordering;
        let value = vec![7u8; 700];
        let (small, used) = accounting(707);
        assert!(matches!(serialize_admitted(&value, Some(&small)), Err(TransportError::Remote(_))));
        assert_eq!(used.load(Ordering::SeqCst), 0);
        let (admit, used) = accounting(708);
        let payload = serialize_admitted(&value, Some(&admit)).unwrap();
        assert_eq!(&payload[..], &bincode::serialize(&value).unwrap());
        let retained = payload.clone();
        drop(payload);
        assert_eq!(used.load(Ordering::SeqCst), 708);
        drop(retained);
        assert_eq!(used.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn changing_serializer_cannot_grow_past_its_admission() {
        struct Changing(std::cell::Cell<bool>);
        impl serde::Serialize for Changing {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let grown = self.0.replace(true);
                serializer.serialize_bytes(if grown { &[1, 2, 3, 4] } else { &[1] })
            }
        }
        let (admit, used) = accounting(9);
        assert!(matches!(serialize_admitted(&Changing(std::cell::Cell::new(false)), Some(&admit)), Err(TransportError::Serialization(_))));
        assert_eq!(used.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn outbound_charge_follows_queued_frame_slices() {
        use std::sync::atomic::Ordering;
        let (admit, used) = accounting(800);
        let frame = encode_admitted(0, 1, "", &[7; 700], None, 10000, Some(&admit)).unwrap();
        let slice = frame.slice(4..);
        drop(frame);
        assert_eq!(used.load(Ordering::SeqCst), 716);
        assert!(matches!(encode_admitted(0, 2, "", &[8; 100], None, 10000, Some(&admit)), Err(TransportError::Remote(_))));
        drop(slice);
        assert_eq!(used.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn outbound_compression_admits_workspace_and_frame_overlap() {
        use std::sync::atomic::Ordering;
        let payload = [7; 5000];
        let workspace = 5000 * 110 / 100 + 24 + 16 * 1024;
        let (admit, used) = accounting(workspace);
        // Workspace fits, but it must remain charged until the framed copy exists.
        assert!(matches!(encode_admitted(0, 1, "", &payload, Some(1), 10000, Some(&admit)), Err(TransportError::Remote(_))));
        assert_eq!(used.load(Ordering::SeqCst), 0);
        let (admit, used) = accounting(workspace + 10000);
        let frame = encode_admitted(0, 1, "", &payload, Some(1), 10000, Some(&admit)).unwrap();
        assert_eq!(used.load(Ordering::SeqCst), frame.len());
        assert!(frame.len() < 5000);
        drop(frame);
        assert_eq!(used.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn admission_precedes_read_and_survives_payload_clones() {
        use std::sync::atomic::Ordering;
        let (admit, used) = accounting(1000);
        let f = encode(0, 1, "", &[1; 700], None, 10000).unwrap();
        let (frame, _) = read_frame_admitted(&mut &f[..], 10000, Some(&admit)).await.unwrap().unwrap();
        let retained = frame.payload.slice(1..);
        drop(frame);
        assert_eq!(used.load(Ordering::SeqCst), f.len() - 4);
        // The second frame is rejected from its header alone, before reading
        // or allocating its absent body; a retained slice owns the first charge.
        let mut header = &f[..4];
        let error = read_frame_admitted(&mut header, 10000, Some(&admit)).await.unwrap_err();
        assert_eq!(error.get_ref().unwrap().downcast_ref::<crate::RemoteError>().unwrap().status, 429);
        drop(retained);
        assert_eq!(used.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn decompression_reserves_both_buffers_and_releases_on_failure() {
        use std::sync::atomic::Ordering;
        let f = encode(0, 1, "", &[7; 5000], Some(1), 10000).unwrap();
        let (admit, used) = accounting(5000); // decoded alone fits, combined does not
        let error = read_frame_admitted(&mut &f[..], 10000, Some(&admit)).await.unwrap_err();
        assert!(error.get_ref().unwrap().downcast_ref::<crate::RemoteError>().is_some());
        assert_eq!(used.load(Ordering::SeqCst), 0);
        let (admit, used) = accounting(10000);
        let (frame, _) = read_frame_admitted(&mut &f[..], 10000, Some(&admit)).await.unwrap().unwrap();
        assert_eq!(used.load(Ordering::SeqCst), 5000);
        drop(frame);
        assert_eq!(used.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn cancelled_partial_frame_releases_admission() {
        use std::sync::atomic::Ordering;
        let (admit, used) = accounting(1000);
        let (mut wr, mut rd) = tokio::io::duplex(32);
        use tokio::io::AsyncWriteExt;
        wr.write_all(&500u32.to_be_bytes()).await.unwrap();
        let mut read = Box::pin(read_frame_admitted(&mut rd, 1000, Some(&admit)));
        assert!(futures::poll!(&mut read).is_pending());
        assert_eq!(used.load(Ordering::SeqCst), 500);
        // Drop the future itself (not just its Pin reference).
        drop(read);
        assert_eq!(used.load(Ordering::SeqCst), 0);
        drop(wr);
    }

    #[tokio::test]
    async fn oversized_frames_rejected() {
        let payload = vec![1u8; 2000];
        assert!(matches!(encode(0, 1, "", &payload, None, 1000), Err(TransportError::FrameTooLarge { .. })));
        let f = encode(0, 1, "", &payload, None, 1 << 20).unwrap();
        let mut rd = &f[..];
        assert!(read_frame(&mut rd, 1000).await.is_err());
        // compressed frame that is small on the wire but large decompressed
        let f = encode(0, 1, "", &payload, Some(1), 1 << 20).unwrap();
        assert!(f.len() < 1000);
        let mut rd = &f[..];
        assert!(read_frame(&mut rd, 1000).await.is_err());
        // clean EOF
        let mut empty: &[u8] = &[];
        assert!(read_frame(&mut empty, 1000).await.unwrap().is_none());
    }
}
