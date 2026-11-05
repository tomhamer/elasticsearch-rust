//! Explicit native control/cache ownership. Never apply this to user query or
//! document payloads: those retain Request/parent-pressure admission.
use crate::front::SharedText;
use elasticsearch_engine::{error::Result,memory::native_control};
pub(crate) fn text(value:&str)->Result<SharedText>{native_control(||SharedText::try_new(value))}
pub(crate) fn formatted(value:std::fmt::Arguments<'_>)->Result<SharedText>{native_control(||SharedText::try_format(value))}
pub(crate) fn decode<'de,D:serde::Deserializer<'de>,T:serde::Deserialize<'de>>(d:D)->std::result::Result<T,D::Error>{native_control(||T::deserialize(d))}

/// Run `f`, control-plane work ES never refuses for memory (its shard-state,
/// recovery and primary-replica resync actions are not subject to the
/// request breaker): every reservation it makes is still accounted on its
/// breaker, but forced (ES `addWithoutBreaking`), never refused, as a
/// replay's are. Refusing it would drop the work, and nothing else redoes
/// it: a promoted primary that skipped its resync leaves diverged replicas
/// serving, a started copy with no replication target misses writes. Only
/// for work its caller keeps bounded (a few identifiers per shard copy, or
/// what a promotion already holds), and synchronous: nothing it runs may
/// await (the policy is thread-local).
pub(crate) fn unrefused<T>(what:&str,f:impl FnOnce()->T)->T{elasticsearch_engine::memory::replaying(what,f)}

/// An engine error for the wire. A `circuit_breaking_exception` keeps its
/// `bytes_wanted`, `bytes_limit` and `durability` (ES carries them in the
/// exception), so the node rendering it shows ES's shape.
pub(crate) fn remote_error(e:&elasticsearch_engine::error::EsError)->elasticsearch_transport::RemoteError{
    let out=elasticsearch_transport::RemoteError::new(&e.kind,e.reason.clone(),e.status);
    if e.kind!="circuit_breaking_exception"{return out}
    let number=|field:&str|e.breaking_value(field).as_u64().unwrap_or(0);
    out.with_breaking(number("bytes_wanted"),number("bytes_limit"),e.breaking_value("durability").as_str().unwrap_or("PERMANENT"))
}
/// A wire error imported as an engine error (admitted, see
/// `EsError::try_owned`), keeping a circuit breaker trip's fields.
pub(crate) fn imported_error(e:&elasticsearch_transport::RemoteError)->elasticsearch_engine::error::EsError{
    match elasticsearch_engine::error::EsError::try_owned(e.status,&e.error_type,&e.reason) {
        Ok(mut out)=>{
            if e.error_type=="circuit_breaking_exception"{
                for field in elasticsearch_transport::BREAKING_FIELDS{out=out.with(field,e.breaking_value(field));}
            }
            out
        }
        Err(refusal)=>refusal,
    }
}
