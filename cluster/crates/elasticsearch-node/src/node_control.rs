//! Admitted control-plane envelopes. Wire fields retain their original order.
use crate::replication::decode::{DecodeMemory, OptionalText, Text};
use elasticsearch_engine::{
    error::{EsError, Result},
    memory::{breakers, Reservation},
};
use serde::{Deserialize, Serialize};
fn queue_bytes(index: &str, aid: &str, size: usize) -> Result<u64> {
    index
        .len()
        .checked_add(aid.len())
        .and_then(|n| n.checked_add(32))
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(size))
        .and_then(|n| n.checked_add(512))
        .map(|n| n as u64)
        .ok_or_else(|| {
            EsError::new(
                429,
                "circuit_breaking_exception",
                "shard report metadata overflow",
            )
        })
}
fn copy(value: &str) -> Result<String> {
    crate::mapping_update::copy_wire_text(value)
}
#[derive(Serialize)]
pub struct ShardStartedReq {
    pub index: String,
    pub shard: u32,
    pub allocation_id: String,
    #[serde(skip)]
    pub(crate) memory: DecodeMemory,
}
impl<'de> Deserialize<'de> for ShardStartedReq {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::{Error, MapAccess, SeqAccess, Visitor};
        #[derive(Deserialize)]
        #[serde(field_identifier)]
        #[allow(non_camel_case_types)]
        enum Field {
            index,
            shard,
            allocation_id,
            #[serde(other)]
            Unknown,
        }
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ShardStartedReq;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("ShardStartedReq")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut memory = breakers()
                    .reserve_request(
                        (std::mem::size_of::<Self::Value>() + 512) as u64,
                        "control envelope",
                    )
                    .map_err(|e| {
                        A::Error::custom(elasticsearch_transport::decode_admission_error(e))
                    })?;
                let index = seq
                    .next_element_seed(Text(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                let shard = seq
                    .next_element::<u32>()?
                    .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                let allocation_id = seq
                    .next_element_seed(Text(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(2, &self))?;
                Ok(ShardStartedReq {
                    index,
                    shard,
                    allocation_id,
                    memory: DecodeMemory::new(memory),
                })
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut memory = breakers()
                    .reserve_request(
                        (std::mem::size_of::<Self::Value>() + 512) as u64,
                        "control envelope",
                    )
                    .map_err(|e| {
                        A::Error::custom(elasticsearch_transport::decode_admission_error(e))
                    })?;
                let mut index = None;
                let mut shard = None;
                let mut allocation_id = None;
                while let Some(field) = map.next_key::<Field>()? {
                    match field {
                        Field::index => {
                            if index.is_some() {
                                return Err(A::Error::duplicate_field("index"));
                            }
                            index = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::shard => {
                            if shard.is_some() {
                                return Err(A::Error::duplicate_field("shard"));
                            }
                            shard = Some(map.next_value::<u32>()?);
                        }
                        Field::allocation_id => {
                            if allocation_id.is_some() {
                                return Err(A::Error::duplicate_field("allocation_id"));
                            }
                            allocation_id = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::Unknown => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                let index = index.ok_or_else(|| A::Error::missing_field("index"))?;
                let shard = shard.ok_or_else(|| A::Error::missing_field("shard"))?;
                let allocation_id =
                    allocation_id.ok_or_else(|| A::Error::missing_field("allocation_id"))?;
                Ok(ShardStartedReq {
                    index,
                    shard,
                    allocation_id,
                    memory: DecodeMemory::new(memory),
                })
            }
        }
        // The master never refuses a shard-state report for memory (ES's
        // `ShardStateAction` handlers do not trip the request breaker): a
        // refused one is only retried for a while by its sender, and a copy
        // whose started report is lost stays initializing. Accounted (the
        // envelope keeps its reservation), never refused.
        crate::control_memory::unrefused("shard started report", || {
            d.deserialize_struct("ShardStartedReq", &["index", "shard", "allocation_id"], V)
        })
    }
}
impl ShardStartedReq {
    pub fn try_new(index: &str, shard: u32, allocation_id: &str) -> Result<Self> {
        let mut memory = DecodeMemory::reserve_texts(breakers(), &[index, allocation_id])?;
        memory.reserve_extra(
            queue_bytes(index, allocation_id, std::mem::size_of::<Self>())?,
            "shard report queue and keys",
        )?;
        Ok(Self {
            index: copy(index)?,
            shard: shard,
            allocation_id: copy(allocation_id)?,
            memory,
        })
    }
}
#[derive(Serialize)]
pub struct ShardFailedReq {
    pub index: String,
    pub shard: u32,
    pub allocation_id: String,
    pub reporter_term: Option<u64>,
    pub message: String,
    pub mark_as_stale: bool,
    #[serde(skip)]
    pub(crate) memory: DecodeMemory,
}
impl<'de> Deserialize<'de> for ShardFailedReq {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::{Error, MapAccess, SeqAccess, Visitor};
        #[derive(Deserialize)]
        #[serde(field_identifier)]
        #[allow(non_camel_case_types)]
        enum Field {
            index,
            shard,
            allocation_id,
            reporter_term,
            message,
            mark_as_stale,
            #[serde(other)]
            Unknown,
        }
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ShardFailedReq;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("ShardFailedReq")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut memory = breakers()
                    .reserve_request(
                        (std::mem::size_of::<Self::Value>() + 512) as u64,
                        "control envelope",
                    )
                    .map_err(|e| {
                        A::Error::custom(elasticsearch_transport::decode_admission_error(e))
                    })?;
                let index = seq
                    .next_element_seed(Text(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                let shard = seq
                    .next_element::<u32>()?
                    .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                let allocation_id = seq
                    .next_element_seed(Text(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(2, &self))?;
                let reporter_term = seq
                    .next_element::<Option<u64>>()?
                    .ok_or_else(|| A::Error::invalid_length(3, &self))?;
                let message = seq
                    .next_element_seed(Text(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(4, &self))?;
                let mark_as_stale = seq
                    .next_element::<bool>()?
                    .ok_or_else(|| A::Error::invalid_length(5, &self))?;
                Ok(ShardFailedReq {
                    index,
                    shard,
                    allocation_id,
                    reporter_term,
                    message,
                    mark_as_stale,
                    memory: DecodeMemory::new(memory),
                })
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut memory = breakers()
                    .reserve_request(
                        (std::mem::size_of::<Self::Value>() + 512) as u64,
                        "control envelope",
                    )
                    .map_err(|e| {
                        A::Error::custom(elasticsearch_transport::decode_admission_error(e))
                    })?;
                let mut index = None;
                let mut shard = None;
                let mut allocation_id = None;
                let mut reporter_term = None;
                let mut message = None;
                let mut mark_as_stale = None;
                while let Some(field) = map.next_key::<Field>()? {
                    match field {
                        Field::index => {
                            if index.is_some() {
                                return Err(A::Error::duplicate_field("index"));
                            }
                            index = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::shard => {
                            if shard.is_some() {
                                return Err(A::Error::duplicate_field("shard"));
                            }
                            shard = Some(map.next_value::<u32>()?);
                        }
                        Field::allocation_id => {
                            if allocation_id.is_some() {
                                return Err(A::Error::duplicate_field("allocation_id"));
                            }
                            allocation_id = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::reporter_term => {
                            if reporter_term.is_some() {
                                return Err(A::Error::duplicate_field("reporter_term"));
                            }
                            reporter_term = Some(map.next_value::<Option<u64>>()?);
                        }
                        Field::message => {
                            if message.is_some() {
                                return Err(A::Error::duplicate_field("message"));
                            }
                            message = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::mark_as_stale => {
                            if mark_as_stale.is_some() {
                                return Err(A::Error::duplicate_field("mark_as_stale"));
                            }
                            mark_as_stale = Some(map.next_value::<bool>()?);
                        }
                        Field::Unknown => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                let index = index.ok_or_else(|| A::Error::missing_field("index"))?;
                let shard = shard.ok_or_else(|| A::Error::missing_field("shard"))?;
                let allocation_id =
                    allocation_id.ok_or_else(|| A::Error::missing_field("allocation_id"))?;
                let reporter_term = reporter_term.unwrap_or(None);
                let message = message.ok_or_else(|| A::Error::missing_field("message"))?;
                let mark_as_stale =
                    mark_as_stale.ok_or_else(|| A::Error::missing_field("mark_as_stale"))?;
                Ok(ShardFailedReq {
                    index,
                    shard,
                    allocation_id,
                    reporter_term,
                    message,
                    mark_as_stale,
                    memory: DecodeMemory::new(memory),
                })
            }
        }
        // As `ShardStartedReq`'s: a lost failure report leaves a failed or
        // diverged copy assigned (and, from a resync, serving).
        crate::control_memory::unrefused("shard failed report", || {
            d.deserialize_struct(
                "ShardFailedReq",
                &[
                    "index",
                    "shard",
                    "allocation_id",
                    "reporter_term",
                    "message",
                    "mark_as_stale",
                ],
                V,
            )
        })
    }
}
impl ShardFailedReq {
    pub fn try_new(
        index: &str,
        shard: u32,
        allocation_id: &str,
        reporter_term: Option<u64>,
        message: &str,
        mark_as_stale: bool,
    ) -> Result<Self> {
        let mut memory = DecodeMemory::reserve_texts(breakers(), &[index, allocation_id, message])?;
        memory.reserve_extra(
            queue_bytes(index, allocation_id, std::mem::size_of::<Self>())?,
            "shard report queue and keys",
        )?;
        Ok(Self {
            index: copy(index)?,
            shard: shard,
            allocation_id: copy(allocation_id)?,
            reporter_term: reporter_term,
            message: copy(message)?,
            mark_as_stale: mark_as_stale,
            memory,
        })
    }
}
#[derive(Serialize)]
pub struct FollowerMetadataReq {
    pub follower: String,
    pub mapping_json: Option<String>,
    pub settings_json: String,
    pub reopen: bool,
    pub aliases_json: Option<String>,
    #[serde(skip)]
    pub(crate) memory: DecodeMemory,
}
impl<'de> Deserialize<'de> for FollowerMetadataReq {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::{Error, MapAccess, SeqAccess, Visitor};
        #[derive(Deserialize)]
        #[serde(field_identifier)]
        #[allow(non_camel_case_types)]
        enum Field {
            follower,
            mapping_json,
            settings_json,
            reopen,
            aliases_json,
            #[serde(other)]
            Unknown,
        }
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = FollowerMetadataReq;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("FollowerMetadataReq")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut memory = breakers()
                    .reserve_request(
                        (std::mem::size_of::<Self::Value>() + 512) as u64,
                        "control envelope",
                    )
                    .map_err(|e| {
                        A::Error::custom(elasticsearch_transport::decode_admission_error(e))
                    })?;
                let follower = seq
                    .next_element_seed(Text(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                let mapping_json = seq
                    .next_element_seed(OptionalText(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                let settings_json = seq
                    .next_element_seed(Text(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(2, &self))?;
                let reopen = seq
                    .next_element::<bool>()?
                    .ok_or_else(|| A::Error::invalid_length(3, &self))?;
                let aliases_json = seq
                    .next_element_seed(OptionalText(&mut memory))?
                    .ok_or_else(|| A::Error::invalid_length(4, &self))?;
                Ok(FollowerMetadataReq {
                    follower,
                    mapping_json,
                    settings_json,
                    reopen,
                    aliases_json,
                    memory: DecodeMemory::new(memory),
                })
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut memory = breakers()
                    .reserve_request(
                        (std::mem::size_of::<Self::Value>() + 512) as u64,
                        "control envelope",
                    )
                    .map_err(|e| {
                        A::Error::custom(elasticsearch_transport::decode_admission_error(e))
                    })?;
                let mut follower = None;
                let mut mapping_json = None;
                let mut settings_json = None;
                let mut reopen = None;
                let mut aliases_json = None;
                while let Some(field) = map.next_key::<Field>()? {
                    match field {
                        Field::follower => {
                            if follower.is_some() {
                                return Err(A::Error::duplicate_field("follower"));
                            }
                            follower = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::mapping_json => {
                            if mapping_json.is_some() {
                                return Err(A::Error::duplicate_field("mapping_json"));
                            }
                            mapping_json = Some(map.next_value_seed(OptionalText(&mut memory))?);
                        }
                        Field::settings_json => {
                            if settings_json.is_some() {
                                return Err(A::Error::duplicate_field("settings_json"));
                            }
                            settings_json = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::reopen => {
                            if reopen.is_some() {
                                return Err(A::Error::duplicate_field("reopen"));
                            }
                            reopen = Some(map.next_value::<bool>()?);
                        }
                        Field::aliases_json => {
                            if aliases_json.is_some() {
                                return Err(A::Error::duplicate_field("aliases_json"));
                            }
                            aliases_json = Some(map.next_value_seed(OptionalText(&mut memory))?);
                        }
                        Field::Unknown => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                let follower = follower.ok_or_else(|| A::Error::missing_field("follower"))?;
                let mapping_json = mapping_json.unwrap_or(None);
                let settings_json =
                    settings_json.ok_or_else(|| A::Error::missing_field("settings_json"))?;
                let reopen = reopen.ok_or_else(|| A::Error::missing_field("reopen"))?;
                let aliases_json = aliases_json.unwrap_or(None);
                Ok(FollowerMetadataReq {
                    follower,
                    mapping_json,
                    settings_json,
                    reopen,
                    aliases_json,
                    memory: DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct(
            "FollowerMetadataReq",
            &[
                "follower",
                "mapping_json",
                "settings_json",
                "reopen",
                "aliases_json",
            ],
            V,
        )
    }
}

fn encoded(memory: &mut Reservation, value: &impl Serialize) -> Result<String> {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(b.len())
                .ok_or_else(|| std::io::Error::other("control JSON overflow"))?;
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    serde_json::to_writer(&mut count, value).map_err(|e| EsError::internal(e.to_string()))?;
    memory.grow(count.0 as u64, "control JSON output")?;
    let mut out = Vec::new();
    out.try_reserve_exact(count.0)
        .map_err(|e| EsError::new(429, "circuit_breaking_exception", e.to_string()))?;
    out.resize(count.0, 0);
    let mut writer = std::io::Cursor::new(out.as_mut_slice());
    serde_json::to_writer(&mut writer, value).map_err(|e| EsError::internal(e.to_string()))?;
    if writer.position() != count.0 as u64 {
        return Err(EsError::internal("control JSON size changed"));
    }
    String::from_utf8(out).map_err(|e| EsError::internal(e.to_string()))
}
impl FollowerMetadataReq {
    pub(super) fn of(
        follower: &str,
        c: &elasticsearch_engine::ccr::FollowerMetadataChange,
    ) -> Result<Self> {
        let mut memory =
            breakers().reserve_request(follower.len() as u64, "follower metadata envelope")?;
        Ok(Self {
            follower: copy(follower)?,
            mapping_json: c
                .mapping
                .as_ref()
                .map(|v| encoded(&mut memory, v))
                .transpose()?,
            settings_json: encoded(&mut memory, &c.settings)?,
            reopen: c.reopen,
            aliases_json: c
                .aliases
                .as_ref()
                .map(|v| encoded(&mut memory, v))
                .transpose()?,
            memory: DecodeMemory::new(memory),
        })
    }
}

/// Follower keys can outlive a submitted metadata job in retry state. Their
/// immutable backing and both bounded B-tree entry allowances share one owner.
#[derive(Clone)]
pub(super) struct FollowerKey(std::sync::Arc<FollowerKeyOwned>);
struct FollowerKeyOwned {
    text: String,
    _memory: Reservation,
}
impl FollowerKey {
    pub(super) fn try_new(text: &str) -> Result<Self> {
        type Failure = (EsError, std::time::Instant, u32, bool);
        let nodes = 4
            * (11 * std::mem::size_of::<(FollowerKey, Failure)>()
                + 20 * std::mem::size_of::<usize>());
        let bytes = text
            .len()
            .checked_add(nodes)
            .and_then(|n| {
                n.checked_add(
                    std::mem::size_of::<FollowerKeyOwned>() + 2 * std::mem::size_of::<usize>(),
                )
            })
            .ok_or_else(|| {
                EsError::new(
                    429,
                    "circuit_breaking_exception",
                    "follower retry metadata overflow",
                )
            })?;
        let memory =
            breakers().reserve_native(bytes as u64, "follower pending and retry metadata")?;
        Ok(Self(std::sync::Arc::new(FollowerKeyOwned {
            text: copy(text)?,
            _memory: memory.into_result_owner(),
        })))
    }
}
/// Keep the last key's structural allowance alive until any empty BTree root
/// has actually been dropped. Nonempty trees remain bounded by live-key nodes.
pub(super) fn remove_pending(set:&mut std::collections::BTreeSet<FollowerKey>,key:&FollowerKey){
    let held=set.get(key).cloned();set.remove(key);
    if set.is_empty(){*set=std::collections::BTreeSet::new();}drop(held);
}
pub(super) fn remove_failure(map:&mut std::collections::BTreeMap<FollowerKey,(EsError,std::time::Instant,u32,bool)>,key:&FollowerKey){
    let held=map.get_key_value(key).map(|(key,_)|key.clone());map.remove(key);
    if map.is_empty(){*map=std::collections::BTreeMap::new();}drop(held);
}
impl std::ops::Deref for FollowerKey {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0.text
    }
}
impl std::borrow::Borrow<str> for FollowerKey {
    fn borrow(&self) -> &str {
        self
    }
}
impl std::fmt::Display for FollowerKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self)
    }
}
impl PartialEq for FollowerKey {
    fn eq(&self, o: &Self) -> bool {
        **self == **o
    }
}
impl Eq for FollowerKey {}
impl PartialOrd for FollowerKey {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for FollowerKey {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.text.cmp(&o.0.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_engine::memory::{scoped, Breakers, Kind, RequestScope};
    fn budget(n: u64) -> &'static Breakers {
        let b = Breakers::leak_for_tests(n);
        b.apply_settings(&|k| {
            (k == "indices.breaker.total.use_real_memory").then(|| "false".into())
        });
        b
    }
    #[test]
    fn report_wire_is_legacy_and_owner_survives_input() {
        let b = budget(1 << 20);
        let scope = RequestScope::new("report wire", b);
        let raw = (
            "index".to_string(),
            3u32,
            "aid".to_string(),
            Some(9u64),
            "x".repeat(4096),
            true,
        );
        let bytes = bincode::serialize(&raw).unwrap();
        let req: ShardFailedReq =
            scoped(Some(scope.clone()), || bincode::deserialize(&bytes)).unwrap();
        assert_eq!(bincode::serialize(&req).unwrap(), bytes);
        drop(raw);
        drop(bytes);
        drop(scope);
        assert_eq!(req.message.len(), 4096);
        assert!(b.get(Kind::Request).used() >= 4096);
        drop(req);
        assert_eq!(b.get(Kind::Request).used(), 0);
    }
    /// A shard-state report over the request breaker's limit is accounted,
    /// never refused (ES's `ShardStateAction` handlers do not trip the
    /// breaker; a refused report would leave a failed copy assigned), and
    /// leaves no charge behind once dropped.
    #[test]
    fn an_envelope_over_the_limit_is_accounted_not_refused_and_fully_released() {
        let b = budget(2048);
        let scope = RequestScope::new("report over the limit", b);
        let bytes = bincode::serialize(&("index", 3u32, "aid", Some(9u64), "x".repeat(4096), true))
            .unwrap();
        let req: ShardFailedReq =
            scoped(Some(scope.clone()), || bincode::deserialize(&bytes)).unwrap();
        assert_eq!(req.message.len(), 4096);
        assert!(b.get(Kind::Request).used() >= 4096);
        drop(scope);
        drop(req);
        assert_eq!(b.get(Kind::Request).used(), 0);
    }
    #[test]
    fn optional_json_fields_preserve_legacy_defaults() {
        let req: ShardFailedReq = serde_json::from_str(
            r#"{"index":"i","shard":0,"allocation_id":"a","message":"m","mark_as_stale":false}"#,
        )
        .unwrap();
        assert_eq!(req.reporter_term, None);
        let req: FollowerMetadataReq =
            serde_json::from_str(r#"{"follower":"f","settings_json":"{}","reopen":false}"#)
                .unwrap();
        assert!(req.mapping_json.is_none() && req.aliases_json.is_none());
    }
    #[test]
    fn retained_retry_key_shares_backing_and_charge() {
        let b = budget(1 << 20);
        let scope = RequestScope::new("retry key", b);
        let key = scoped(Some(scope.clone()), || FollowerKey::try_new("follower")).unwrap();
        let clone = key.clone();
        assert!(std::sync::Arc::ptr_eq(&key.0, &clone.0));
        drop(key);
        drop(scope);
        assert!(b.get(Kind::Accounting).used() > 0);
        drop(clone);
        assert_eq!(b.get(Kind::Accounting).used(), 0);
    }
}
