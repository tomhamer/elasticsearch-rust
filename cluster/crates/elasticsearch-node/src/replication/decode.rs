//! Seeded string decoding for owned replication envelopes. The caller retains
//! the reservation alongside the completed envelope (including on handoff).
use elasticsearch_engine::memory::Reservation;
use serde::de::{DeserializeSeed, Deserializer, Error, Visitor};

pub(crate) struct Text<'a>(pub &'a mut Reservation);
impl<'de> DeserializeSeed<'de> for Text<'_> {
    type Value = String;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<String, D::Error> {
        struct V<'a>(&'a mut Reservation);
        impl<'de> Visitor<'de> for V<'_> {
            type Value = String;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("replication text") }
            fn visit_str<E: Error>(self, text: &str) -> Result<String, E> {
                // deserialize_str lets bincode borrow the input rather than
                // allocate a String before this admission point.
                self.0.grow(text.len() as u64, "replication decoded text").map_err(|e| E::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut value = String::new();
                value.try_reserve_exact(text.len()).map_err(|e| E::custom(elasticsearch_transport::decode_admission_error(e)))?;
                value.push_str(text);
                Ok(value)
            }
            fn visit_borrowed_str<E: Error>(self, text: &'de str) -> Result<String, E> { self.visit_str(text) }
        }
        d.deserialize_str(V(self.0))
    }
}

pub(crate) struct OptionalText<'a>(pub &'a mut Reservation);
impl<'de> DeserializeSeed<'de> for OptionalText<'_> {
    type Value = Option<String>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        struct V<'a>(&'a mut Reservation);
        impl<'de> Visitor<'de> for V<'_> {
            type Value = Option<String>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("optional replication text") }
            fn visit_none<E: Error>(self) -> Result<Self::Value, E> { Ok(None) }
            fn visit_unit<E: Error>(self) -> Result<Self::Value, E> { Ok(None) }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
                Text(self.0).deserialize(d).map(Some)
            }
        }
        d.deserialize_option(V(self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::Options;
    use elasticsearch_engine::memory::Breakers;

    #[test]
    fn decoded_batch_owns_payload_after_input_drop_and_releases_partial_failure() {
        use super::super::ops::ReplicaOp;
        use elasticsearch_engine::memory::Kind;
        let input = vec![ReplicaOp::NoOp { seq_no: 1, primary_term: 2, reason: "x".repeat(4096) }];
        let bytes = bincode::serialize(&input).unwrap();
        // Private allocation tests exclude unrelated process RSS; child/parent limits remain finite.
        let breakers = Breakers::leak_for_tests(65536);
        breakers.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let options = bincode::DefaultOptions::new().with_fixint_encoding().allow_trailing_bytes();
        let mut decoder = bincode::Deserializer::from_slice(&bytes, options);
        let batch = Operations(breakers).deserialize(&mut decoder).unwrap();
        drop(decoder); drop(bytes); drop(input);
        assert!(breakers.get(Kind::Request).used() >= 4096);
        assert_eq!(batch[0].seq_no(), 1);
        assert_eq!(bincode::serialize(&batch).unwrap(), bincode::serialize(&*batch).unwrap());
        drop(batch);
        assert_eq!(breakers.get(Kind::Request).used(), 0);

        let input = vec![ReplicaOp::NoOp { seq_no: 1, primary_term: 2, reason: "ok".into() },
            ReplicaOp::NoOp { seq_no: 2, primary_term: 2, reason: "x".repeat(4096) }];
        let bytes = bincode::serialize(&input).unwrap();
        let tiny = Breakers::leak_for_tests(2048);
        tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let options = bincode::DefaultOptions::new().with_fixint_encoding().allow_trailing_bytes();
        let mut decoder = bincode::Deserializer::from_slice(&bytes, options);
        assert!(Operations(tiny).deserialize(&mut decoder).is_err());
        assert_eq!(tiny.get(Kind::Request).used(), 0, "partial strings and slots released on decode failure");
    }

    #[test]
    fn admitted_operation_seed_preserves_wire_variants_and_rejects_large_source() {
        use super::super::ops::ReplicaOp;
        let operations = [
            ReplicaOp::Index { id: "i".into(), source_json: "{\"x\":\"λ\"}".into(), routing: Some("r".into()), seq_no: 1, version: 2, primary_term: 3 },
            ReplicaOp::Delete { id: "d".into(), seq_no: 4, version: 5, primary_term: 6 },
            ReplicaOp::NoOp { seq_no: 7, primary_term: 8, reason: "why\nλ".into() },
        ];
        for op in operations {
            let breakers = Breakers::leak_for_tests(65536);
            breakers.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
            let mut owner = breakers.reserve_request(0, "operation test").unwrap();
            let encoded = bincode::serialize(&op).unwrap();
            let options = bincode::DefaultOptions::new().with_fixint_encoding().allow_trailing_bytes();
            let mut decoder = bincode::Deserializer::from_slice(&encoded, options);
            assert_eq!(Operation(&mut owner).deserialize(&mut decoder).unwrap(), op);
            let text = serde_json::to_string(&op).unwrap();
            let mut decoder = serde_json::Deserializer::from_str(&text);
            assert_eq!(Operation(&mut owner).deserialize(&mut decoder).unwrap(), op);
        }
        let op = ReplicaOp::Index { id: "i".into(), source_json: "x".repeat(4096), routing: None, seq_no: 1, version: 2, primary_term: 3 };
        let encoded = bincode::serialize(&op).unwrap();
        let breakers = Breakers::leak_for_tests(1024);
        breakers.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let mut owner = breakers.reserve_request(0, "operation test").unwrap();
        let options = bincode::DefaultOptions::new().with_fixint_encoding().allow_trailing_bytes();
        let mut decoder = bincode::Deserializer::from_slice(&encoded, options);
        assert!(Operation(&mut owner).deserialize(&mut decoder).is_err());
        assert_eq!(owner.bytes(), 1, "only the admitted id was copied, not the oversized source");
    }

    #[test]
    fn text_seed_rejects_before_copy_and_retains_exact_text_bytes() {
        let input = "x".repeat(4096);
        let encoded = bincode::serialize(&input).unwrap();
        let tiny = Breakers::leak_for_tests(1024);
        tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let mut denied = tiny.reserve_request(0, "decode test").unwrap();
        let options = bincode::DefaultOptions::new().with_fixint_encoding().allow_trailing_bytes();
        let mut decoder = bincode::Deserializer::from_slice(&encoded, options);
        assert!(Text(&mut denied).deserialize(&mut decoder).is_err());
        assert_eq!(denied.bytes(), 0, "failed admission precedes text copy");
        let roomy = Breakers::leak_for_tests(65536);
        roomy.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let mut owner = roomy.reserve_request(0, "decode test").unwrap();
        let options = bincode::DefaultOptions::new().with_fixint_encoding().allow_trailing_bytes();
        let mut decoder = bincode::Deserializer::from_slice(&encoded, options);
        let result = Text(&mut owner).deserialize(&mut decoder).unwrap();
        assert_eq!(result, input);
        assert_eq!(owner.bytes(), result.len() as u64);
        drop(result);
        drop(owner);
    }

    #[test]
    fn text_seed_preserves_escaped_json_and_optional_wire_text() {
        let breakers = Breakers::leak_for_tests(65536);
        breakers.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let mut owner = breakers.reserve_request(0, "decode test").unwrap();
        let mut decoder = serde_json::Deserializer::from_str(r#""a\n\u03bb""#);
        let text = Text(&mut owner).deserialize(&mut decoder).unwrap();
        assert_eq!(text, "a\nλ");
        for value in [None, Some(text)] {
            let encoded = bincode::serialize(&value).unwrap();
            let options = bincode::DefaultOptions::new().with_fixint_encoding().allow_trailing_bytes();
            let mut decoder = bincode::Deserializer::from_slice(&encoded, options);
            assert_eq!(OptionalText(&mut owner).deserialize(&mut decoder).unwrap(), value);
        }
    }
}

/// One operation decoded under the eventual batch owner's reservation.
pub(crate) struct Operation<'a>(pub &'a mut Reservation);
impl<'de> DeserializeSeed<'de> for Operation<'_> {
    type Value = super::ops::ReplicaOp;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        use serde::de::{EnumAccess, VariantAccess};
        #[derive(serde::Deserialize)]
        #[serde(field_identifier)]
        enum Variant { Index, Delete, NoOp }
        struct V<'a>(&'a mut Reservation);
        impl<'de> Visitor<'de> for V<'_> {
            type Value = super::ops::ReplicaOp;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("replica operation") }
            fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<Self::Value, A::Error> {
                let (variant, fields) = data.variant::<Variant>()?;
                match variant {
                    Variant::Index => fields.struct_variant(&["id", "source_json", "routing", "seq_no", "version", "primary_term"], IndexFields(self.0)),
                    Variant::Delete => fields.struct_variant(&["id", "seq_no", "version", "primary_term"], DeleteFields(self.0)),
                    Variant::NoOp => fields.struct_variant(&["seq_no", "primary_term", "reason"], NoOpFields(self.0)),
                }
            }
        }
        d.deserialize_enum("ReplicaOp", &["Index", "Delete", "NoOp"], V(self.0))
    }
}
struct IndexFields<'a>(&'a mut Reservation);
impl<'de> Visitor<'de> for IndexFields<'_> {
    type Value = super::ops::ReplicaOp;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("Index fields") }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let id = seq.next_element_seed(Text(self.0))?.ok_or_else(|| A::Error::invalid_length(0, &"Index fields"))?;
        let source_json = seq.next_element_seed(Text(self.0))?.ok_or_else(|| A::Error::invalid_length(1, &"Index fields"))?;
        let routing = seq.next_element_seed(OptionalText(self.0))?.ok_or_else(|| A::Error::invalid_length(2, &"Index fields"))?;
        let seq_no = seq.next_element::<i64>()?.ok_or_else(|| A::Error::invalid_length(3, &"Index fields"))?;
        let version = seq.next_element::<u64>()?.ok_or_else(|| A::Error::invalid_length(4, &"Index fields"))?;
        let primary_term = seq.next_element::<u64>()?.ok_or_else(|| A::Error::invalid_length(5, &"Index fields"))?;
        Ok(super::ops::ReplicaOp::Index { id, source_json, routing, seq_no, version, primary_term })
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field { Id, SourceJson, Routing, SeqNo, Version, PrimaryTerm }
        let mut id = None;
        let mut source_json = None;
        let mut routing = None;
        let mut seq_no = None;
        let mut version = None;
        let mut primary_term = None;
        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::Id => {
                    if id.is_some() { return Err(A::Error::duplicate_field("id")); }
                    id = Some(map.next_value_seed(Text(self.0))?);
                }
                Field::SourceJson => {
                    if source_json.is_some() { return Err(A::Error::duplicate_field("source_json")); }
                    source_json = Some(map.next_value_seed(Text(self.0))?);
                }
                Field::Routing => {
                    if routing.is_some() { return Err(A::Error::duplicate_field("routing")); }
                    routing = Some(map.next_value_seed(OptionalText(self.0))?);
                }
                Field::SeqNo => {
                    if seq_no.is_some() { return Err(A::Error::duplicate_field("seq_no")); }
                    seq_no = Some(map.next_value::<i64>()?);
                }
                Field::Version => {
                    if version.is_some() { return Err(A::Error::duplicate_field("version")); }
                    version = Some(map.next_value::<u64>()?);
                }
                Field::PrimaryTerm => {
                    if primary_term.is_some() { return Err(A::Error::duplicate_field("primary_term")); }
                    primary_term = Some(map.next_value::<u64>()?);
                }
            }
        }
        Ok(super::ops::ReplicaOp::Index {
            id: id.ok_or_else(|| A::Error::missing_field("id"))?,
            source_json: source_json.ok_or_else(|| A::Error::missing_field("source_json"))?,
            routing: routing.unwrap_or(None),
            seq_no: seq_no.ok_or_else(|| A::Error::missing_field("seq_no"))?,
            version: version.ok_or_else(|| A::Error::missing_field("version"))?,
            primary_term: primary_term.ok_or_else(|| A::Error::missing_field("primary_term"))?,
        })
    }
}
struct DeleteFields<'a>(&'a mut Reservation);
impl<'de> Visitor<'de> for DeleteFields<'_> {
    type Value = super::ops::ReplicaOp;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("Delete fields") }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let id = seq.next_element_seed(Text(self.0))?.ok_or_else(|| A::Error::invalid_length(0, &"Delete fields"))?;
        let seq_no = seq.next_element::<i64>()?.ok_or_else(|| A::Error::invalid_length(1, &"Delete fields"))?;
        let version = seq.next_element::<u64>()?.ok_or_else(|| A::Error::invalid_length(2, &"Delete fields"))?;
        let primary_term = seq.next_element::<u64>()?.ok_or_else(|| A::Error::invalid_length(3, &"Delete fields"))?;
        Ok(super::ops::ReplicaOp::Delete { id, seq_no, version, primary_term })
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field { Id, SeqNo, Version, PrimaryTerm }
        let mut id = None;
        let mut seq_no = None;
        let mut version = None;
        let mut primary_term = None;
        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::Id => {
                    if id.is_some() { return Err(A::Error::duplicate_field("id")); }
                    id = Some(map.next_value_seed(Text(self.0))?);
                }
                Field::SeqNo => {
                    if seq_no.is_some() { return Err(A::Error::duplicate_field("seq_no")); }
                    seq_no = Some(map.next_value::<i64>()?);
                }
                Field::Version => {
                    if version.is_some() { return Err(A::Error::duplicate_field("version")); }
                    version = Some(map.next_value::<u64>()?);
                }
                Field::PrimaryTerm => {
                    if primary_term.is_some() { return Err(A::Error::duplicate_field("primary_term")); }
                    primary_term = Some(map.next_value::<u64>()?);
                }
            }
        }
        Ok(super::ops::ReplicaOp::Delete {
            id: id.ok_or_else(|| A::Error::missing_field("id"))?,
            seq_no: seq_no.ok_or_else(|| A::Error::missing_field("seq_no"))?,
            version: version.ok_or_else(|| A::Error::missing_field("version"))?,
            primary_term: primary_term.ok_or_else(|| A::Error::missing_field("primary_term"))?,
        })
    }
}
struct NoOpFields<'a>(&'a mut Reservation);
impl<'de> Visitor<'de> for NoOpFields<'_> {
    type Value = super::ops::ReplicaOp;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("NoOp fields") }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let seq_no = seq.next_element::<i64>()?.ok_or_else(|| A::Error::invalid_length(0, &"NoOp fields"))?;
        let primary_term = seq.next_element::<u64>()?.ok_or_else(|| A::Error::invalid_length(1, &"NoOp fields"))?;
        let reason = seq.next_element_seed(Text(self.0))?.ok_or_else(|| A::Error::invalid_length(2, &"NoOp fields"))?;
        Ok(super::ops::ReplicaOp::NoOp { seq_no, primary_term, reason })
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field { SeqNo, PrimaryTerm, Reason }
        let mut seq_no = None;
        let mut primary_term = None;
        let mut reason = None;
        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::SeqNo => {
                    if seq_no.is_some() { return Err(A::Error::duplicate_field("seq_no")); }
                    seq_no = Some(map.next_value::<i64>()?);
                }
                Field::PrimaryTerm => {
                    if primary_term.is_some() { return Err(A::Error::duplicate_field("primary_term")); }
                    primary_term = Some(map.next_value::<u64>()?);
                }
                Field::Reason => {
                    if reason.is_some() { return Err(A::Error::duplicate_field("reason")); }
                    reason = Some(map.next_value_seed(Text(self.0))?);
                }
            }
        }
        Ok(super::ops::ReplicaOp::NoOp {
            seq_no: seq_no.ok_or_else(|| A::Error::missing_field("seq_no"))?,
            primary_term: primary_term.ok_or_else(|| A::Error::missing_field("primary_term"))?,
            reason: reason.ok_or_else(|| A::Error::missing_field("reason"))?,
        })
    }
}

/// Scalar allocation owner for a wire envelope. This field is never serialized
/// and has no default: producers must supply a reservation made before copying.
pub struct DecodeMemory(Reservation);
impl DecodeMemory {
    pub fn new(memory: Reservation) -> Self { Self(memory) }
    /// Reserve before the producer copies any envelope scalar text.
    pub fn reserve_texts(breakers: &'static elasticsearch_engine::memory::Breakers, values: &[&str]) -> Result<Self, elasticsearch_engine::error::EsError> {
        let mut memory = breakers.reserve_request(0, "replication envelope scalars")?;
        for value in values { memory.grow(value.len() as u64, "replication envelope text")?; }
        Ok(Self(memory))
    }
    pub fn reserve_extra(&mut self,bytes:u64,label:&str)->Result<(),elasticsearch_engine::error::EsError>{self.0.grow(bytes,label)}
    pub fn bytes(&self) -> u64 { self.0.bytes() }
}
impl std::fmt::Debug for DecodeMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_tuple("DecodeMemory").field(&self.bytes()).finish() }
}
impl PartialEq for DecodeMemory { fn eq(&self, _: &Self) -> bool { true } }
impl Eq for DecodeMemory {}

pub(crate) struct Operations(pub &'static elasticsearch_engine::memory::Breakers);
impl<'de> DeserializeSeed<'de> for Operations {
    type Value = elasticsearch_engine::translog::AdmittedBatch<super::ops::ReplicaOp>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        struct V(&'static elasticsearch_engine::memory::Breakers);
        impl<'de> Visitor<'de> for V {
            type Value = elasticsearch_engine::translog::AdmittedBatch<super::ops::ReplicaOp>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("admitted replica operations") }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut memory = self.0.reserve_request(0, "replication decoded operations").map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut values = Vec::<super::ops::ReplicaOp>::new();
                loop {
                    if seq.size_hint() == Some(0) { break; }
                    if values.len() == values.capacity() {
                        let old = values.capacity();
                        let capacity = old.checked_mul(2).unwrap_or(usize::MAX).max(1);
                        let bytes = capacity.checked_mul(std::mem::size_of::<super::ops::ReplicaOp>())
                            .ok_or_else(|| A::Error::custom(elasticsearch_transport::decode_admission_error("replication operation capacity overflow")))?;
                        memory.grow(bytes as u64, "replication decoded operation overlap").map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                        values.try_reserve_exact(capacity - values.len()).map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                        memory.shrink((old * std::mem::size_of::<super::ops::ReplicaOp>()) as u64);
                    }
                    match seq.next_element_seed(Operation(&mut memory))? {
                        Some(value) => values.push(value),
                        None => break,
                    }
                }
                Ok(elasticsearch_engine::translog::AdmittedBatch::from_reserved(values, memory))
            }
        }
        d.deserialize_seq(V(self.0))
    }
}
