//! Owned forwarding copies. No payload or nested DFS/application-plan copy is
//! made until its allocation has been admitted; the owner spans the send.
use super::{AppSearchPlan, Dfs, RestReq};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::memory::{Breakers, Reservation};
use elasticsearch_http::{DfsField, DfsStats, DfsTerm};

pub(super) struct Request {
    // Declaration order is intentional: data is destroyed before admission.
    pub(super) value: RestReq,
    pub(super) memory: (),
}

fn allocation(error: impl std::fmt::Display) -> EsError {
    EsError::new(429, "circuit_breaking_exception", error.to_string())
}
pub(super) fn slots<T>(memory: &mut Reservation, count: usize) -> Result<Vec<T>, EsError> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| allocation("forwarding capacity overflow"))?;
    memory.grow(bytes as u64, "forwarding slots")?;
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(allocation)?;
    Ok(out)
}
pub(super) fn text(memory: &mut Reservation, value: &str) -> Result<String, EsError> {
    memory.grow(value.len() as u64, "forwarding text")?;
    let mut out = String::new();
    out.try_reserve_exact(value.len()).map_err(allocation)?;
    out.push_str(value);
    Ok(out)
}
pub(super) fn bytes(memory: &mut Reservation, value: &[u8]) -> Result<Vec<u8>, EsError> {
    let mut out = slots::<u8>(memory, value.len())?;
    out.extend_from_slice(value);
    Ok(out)
}

/// A local HTTP request can outlive the dispatch future (the router may move
/// its body into a worker). Put the reservation in the bytes owner, rather
/// than on the awaiting caller's stack.
pub(super) fn http_body(value: &[u8], breakers: &'static Breakers) -> Result<axum::body::Bytes, EsError> {
    struct Owned { value: Vec<u8>, _memory: Reservation }
    impl AsRef<[u8]> for Owned { fn as_ref(&self) -> &[u8] { &self.value } }
    let mut memory = breakers.reserve_request((std::mem::size_of::<Owned>() + 4 * std::mem::size_of::<usize>()) as u64, "local HTTP body owner")?;
    let value = bytes(&mut memory, value)?;
    Ok(axum::body::Bytes::from_owner(Owned { value, _memory: memory.into_result_owner() }))
}

/// Encode into a fixed admitted buffer. The callback is invoked twice; if its
/// output changes, the fixed destination rejects growth or a size mismatch.
/// It must only borrow input and write to the supplied sink.
pub(super) fn encoded_body(encode: impl Fn(&mut dyn std::io::Write) -> std::io::Result<()>, breakers: &'static Breakers) -> Result<axum::body::Bytes, EsError> {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> { self.0=self.0.checked_add(bytes.len()).ok_or_else(||std::io::Error::other("request body size overflow"))?; Ok(bytes.len()) }
        fn flush(&mut self)->std::io::Result<()> { Ok(()) }
    }
    struct Owned { bytes: Vec<u8>, _memory: Reservation }
    impl AsRef<[u8]> for Owned { fn as_ref(&self)->&[u8] { &self.bytes } }
    let mut count=Count(0); encode(&mut count).map_err(allocation)?;
    let mut memory=breakers.reserve_request((std::mem::size_of::<Owned>()+4*std::mem::size_of::<usize>()) as u64,"encoded request owner")?;
    let mut bytes=slots::<u8>(&mut memory,count.0)?; bytes.resize(count.0,0);
    let mut remaining=bytes.as_mut_slice(); encode(&mut remaining).map_err(allocation)?;
    if !remaining.is_empty() { return Err(allocation("request body changed between count and encoding")); }
    Ok(axum::body::Bytes::from_owner(Owned{bytes,_memory:memory.into_result_owner()}))
}

/// Count the immutable value first, then encode into a fixed-size admitted
/// destination. No growable serde writer can allocate behind the reservation.
pub(super) fn json(memory: &mut Reservation, value: &serde_json::Value) -> Result<Vec<u8>, EsError> {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_add(bytes.len()).ok_or_else(|| std::io::Error::other("local request JSON size overflow"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let mut count = Count(0);
    serde_json::to_writer(&mut count, value).map_err(allocation)?;
    let mut out = slots::<u8>(memory, count.0)?;
    out.resize(count.0, 0);
    serde_json::to_writer(out.as_mut_slice(), value).map_err(allocation)?;
    Ok(out)
}

impl Request {
    /// Re-enter the coordinator after authorization without unaccounted
    /// HeaderMap/String/body copies. Invalid textual headers remain omitted,
    /// matching the previous HTTP boundary behavior.
    pub(super) fn authorized(method: &str, uri: &str, headers: &axum::http::HeaderMap, body: &[u8], breakers: &'static Breakers) -> Result<Self, EsError> {
        let mut memory = breakers.reserve_request(0, "authorized coordinator request")?;
        let method = text(&mut memory, method)?;
        let uri = text(&mut memory, uri)?;
        let mut copied = slots(&mut memory, headers.len())?;
        for (name, value) in headers {
            if let Ok(value) = value.to_str() { copied.push((text(&mut memory, name.as_str())?, text(&mut memory, value)?)); }
        }
        let body = bytes(&mut memory, body)?;
        Ok(Self { value: RestReq::from_reserved(method,uri,copied,body,0,Dfs::None,None,memory)?, memory:() })
    }

    /// Construct an engine callback request without cloning credentials or
    /// serializing its body before admission. The caller retains `memory`
    /// through the complete synchronous dispatch, including cancellation.
    ///
    /// `nested`: the caller's thread waits for the answer while holding
    /// an executor thread (see `elasticsearch_http::pools::sub_requests_nested`):
    /// the request and everything derived from it (its shard requests,
    /// also those other tasks or nodes run) run off the bounded executors.
    pub(super) fn cluster_call(method: &str, uri: &str, credentials: &[(String, String)], body: Option<&serde_json::Value>, nested: bool, breakers: &'static Breakers) -> Result<Self, EsError> {
        let mut memory = breakers.reserve_request(0, "local cluster request")?;
        let method = text(&mut memory, method)?;
        let uri = text(&mut memory, uri)?;
        let count = credentials.len().checked_add(2 + usize::from(nested)).ok_or_else(|| allocation("local cluster header capacity overflow"))?;
        let mut headers = slots(&mut memory, count)?;
        for (name, value) in credentials { headers.push((text(&mut memory, name)?, text(&mut memory, value)?)); }
        headers.push((text(&mut memory, super::RANK_PLAN_HEADER)?, text(&mut memory, "apply")?));
        if nested {
            headers.push((text(&mut memory, super::NESTED_HEADER)?, text(&mut memory, "1")?));
        }
        let (content_type, body) = match body {
            Some(serde_json::Value::String(raw)) => ("application/x-ndjson", bytes(&mut memory, raw.as_bytes())?),
            Some(value) => ("application/json", json(&mut memory, value)?),
            None => ("application/json", Vec::new()),
        };
        headers.push((text(&mut memory, "content-type")?, text(&mut memory, content_type)?));
        Ok(Self { value: RestReq::from_reserved(method,uri,headers,body,0,Dfs::None,None,memory)?, memory:() })
    }

    pub(super) fn forwarding(req: &RestReq, nested: bool, breakers: &'static Breakers) -> Result<Self, EsError> {
        let hops = req.hops.checked_add(1).ok_or_else(|| allocation("forwarding hop overflow"))?;
        let mut memory = breakers.reserve_request(0, "forwarding request")?;
        let method = text(&mut memory, &req.method)?;
        let uri = text(&mut memory, &req.uri)?;
        let add_nested = nested && !super::is_nested(&req.headers);
        let add_trips = !req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(super::TRIPS_HEADER));
        let count = req.headers.len().checked_add(usize::from(add_nested) + usize::from(add_trips))
            .ok_or_else(|| allocation("forwarding header capacity overflow"))?;
        let mut headers = slots(&mut memory, count)?;
        for (name, value) in &req.headers {
            headers.push((text(&mut memory, name)?, text(&mut memory, value)?));
        }
        for name in [add_nested.then_some(super::NESTED_HEADER), add_trips.then_some(super::TRIPS_HEADER)].into_iter().flatten() {
            headers.push((text(&mut memory, name)?, text(&mut memory, "1")?));
        }
        let body = bytes(&mut memory, &req.body)?;
        let dfs = req.dfs.clone();
        let app_plan = req.app_plan.clone();
        Ok(Self { value: RestReq::from_reserved(method,uri,headers,body,hops,dfs,app_plan,memory)?, memory:() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_engine::memory::Kind;
    // Tiny private budgets exercise admitted bytes, not unrelated process RSS.
    fn fixture_breakers(bytes: u64) -> &'static Breakers {
        let budget = Breakers::leak_for_tests(bytes);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        budget
    }
    fn fixture() -> RestReq {
        RestReq::try_new("POST","/i/_search",&[("content-type".into(),"application/json".into())],&vec![b'x';4096],0,
            Dfs::Use(super::super::SharedDfsStats::try_new(&DfsStats { terms: vec![DfsTerm { field: "f".into(), term: vec![42;2048], doc_freq:1,total_term_freq:2,field_doc_count:3 }], fields:vec![] }).unwrap()),
            Some(AppSearchPlan::try_new(&["i".into()],&vec![b'y';1024]).unwrap())).unwrap()
    }
    #[test]
    fn forwarding_copy_admits_nested_payload_and_releases_after_owner() {
        let source = fixture();
        let breakers = fixture_breakers(1 << 20);
        let copy = Request::forwarding(&source, true, breakers).unwrap();
        assert_eq!(copy.value.body, source.body);
        assert_eq!(copy.value.dfs, source.dfs);
        assert_eq!(copy.value.hops, 1);
        assert!(super::super::is_nested(&copy.value.headers));
        assert!(copy.value.headers.iter().any(|(k, _)| k == super::super::TRIPS_HEADER));
        assert_eq!(copy.value.app_plan.as_ref().unwrap().query_json.as_ptr(),source.app_plan.as_ref().unwrap().query_json.as_ptr(),"forwarding shares admitted application plan storage");
        drop(source);
        assert!(breakers.get(Kind::Request).used() >= 4096);
        drop(copy);
        assert_eq!(breakers.get(Kind::Request).used(), 0);
    }
    #[test]
    fn forwarding_refusal_releases_partial_copy_and_does_not_change_source() {
        let source = fixture();
        let breakers = fixture_breakers(1024);
        let error = Request::forwarding(&source, true, breakers).err().unwrap();
        assert_eq!(error.status, 429);
        assert_eq!(source.hops, 0);
        assert!(!super::super::is_nested(&source.headers));
        assert_eq!(breakers.get(Kind::Request).used(), 0);
    }
    #[test]
    fn cluster_callback_preserves_json_and_ndjson_under_its_owner() {
        let budget = fixture_breakers(1 << 20);
        let credentials = vec![("authorization".into(), "Bearer token".into())];
        for body in [serde_json::json!({"query":{"match_all":{}}}), serde_json::Value::String("{}\n{}\n".into())] {
            let request = Request::cluster_call("POST", "/i/_search", &credentials, Some(&body), false, budget).unwrap();
            let expected = match &body { serde_json::Value::String(s) => s.as_bytes().to_vec(), v => serde_json::to_vec(v).unwrap() };
            assert_eq!(request.value.body, expected);
            assert_eq!(request.value.headers[0], credentials[0]);
            assert_eq!(request.value.hops, 0);
            assert!(budget.get(Kind::Request).used() >= request.value.body.len() as u64);
            drop(request);
            assert_eq!(budget.get(Kind::Request).used(), 0);
        }
    }
    #[test]
    fn cluster_callback_refusal_releases_partial_metadata_before_dispatch() {
        let budget = fixture_breakers(512);
        let body = serde_json::json!({"payload":"x".repeat(4096)});
        let error = Request::cluster_call("POST", "/i/_search", &[], Some(&body), false, budget).err().unwrap();
        assert_eq!(error.status, 429);
        assert_eq!(budget.get(Kind::Request).used(), 0);
    }

    #[test]
    fn local_http_body_owner_survives_scope_and_shared_extraction() {
        use elasticsearch_engine::memory::{RequestScope, scoped};
        let budget = fixture_breakers(1 << 20);
        let scope = RequestScope::new("HTTP body", budget);
        let body = scoped(Some(scope.clone()), || http_body(&[7;4096], budget)).unwrap();
        let retained = body.slice(100..200);
        drop(body);
        assert_eq!(scope.held(), 0);
        drop(scope);
        assert_eq!(retained.as_ref(), &[7;100]);
        assert!(budget.get(Kind::Request).used() >= 4096);
        drop(retained);
        assert_eq!(budget.get(Kind::Request).used(), 0);
    }

    #[test]
    fn authorized_copy_preserves_duplicate_headers_without_copying_invalid_text() {
        let budget = fixture_breakers(1 << 20);
        let mut headers = axum::http::HeaderMap::new();
        headers.append("x-test", axum::http::HeaderValue::from_static("one"));
        headers.append("x-test", axum::http::HeaderValue::from_static("two"));
        headers.insert("x-binary", axum::http::HeaderValue::from_bytes(&[255]).unwrap());
        let request = Request::authorized("GET", "/i/_search", &headers, b"{}", budget).unwrap();
        assert_eq!(request.value.headers, vec![("x-test".into(), "one".into()), ("x-test".into(), "two".into())]);
        assert_eq!(request.value.body.as_ref(), b"{}");
        drop(request);
        assert_eq!(budget.get(Kind::Request).used(), 0);
    }

    #[test]
    fn encoded_body_is_bounded_and_owner_follows_slices() {
        let budget=fixture_breakers(1<<20);
        let body=encoded_body(|out| { out.write_all(b"header\n")?; out.write_all(b"source\n") },budget).unwrap();
        assert_eq!(body.as_ref(),b"header\nsource\n");
        let retained=body.slice(7..); drop(body);
        assert!(budget.get(Kind::Request).used()>0);
        assert_eq!(retained.as_ref(),b"source\n"); drop(retained);
        assert_eq!(budget.get(Kind::Request).used(),0);
        let calls=std::cell::Cell::new(0);
        let failure=encoded_body(|out| { let n=calls.get(); calls.set(n+1); out.write_all(if n==0 {b"x"} else {b"longer"}) },budget).unwrap_err();
        assert_eq!(failure.status,429);
        assert_eq!(budget.get(Kind::Request).used(),0);
    }

    #[test]
    fn encoded_body_refuses_before_second_encoding() {
        let budget=fixture_breakers(128);
        let calls=std::cell::Cell::new(0);
        let failure=encoded_body(|out| { calls.set(calls.get()+1); out.write_all(&[0;4096]) },budget).unwrap_err();
        assert_eq!(failure.status,429);
        assert_eq!(calls.get(),1);
        assert_eq!(budget.get(Kind::Request).used(),0);
    }

}
