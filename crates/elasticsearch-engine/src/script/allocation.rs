//! Allocation accounting enabled only for scripted-metric execution. Owners
//! retain reservations, so aliases do not double-charge and dropping a
//! temporary collection gives its bytes back before the script finishes.
use super::error::{SResult, ScriptErr};
use crate::memory::{Breakers, Reservation};
use std::cell::RefCell;
use std::rc::{Rc, Weak};
use super::value::{Key, Val};
use indexmap::IndexMap;

enum Storage {
    List(Weak<RefCell<Vec<Val>>>),
    Map(Weak<RefCell<IndexMap<Key, Val>>>),
}

enum ScalarOwner {
    Text(Weak<str>),
    Units(Weak<[u16]>),
    Object(Weak<super::jmisc::Obj>),
    Doc(Weak<super::value::DocValues>),
    Big(Weak<super::jbig::Big>),
    Regex(std::sync::Weak<super::regex::CompiledRegex>),
    Pattern(std::sync::Weak<super::regex::PatternData>),
}
impl ScalarOwner {
    fn alive(&self)->bool {match self {Self::Text(v)=>v.strong_count()!=0,Self::Units(v)=>v.strong_count()!=0,Self::Object(v)=>v.strong_count()!=0,Self::Doc(v)=>v.strong_count()!=0,Self::Big(v)=>v.strong_count()!=0,Self::Regex(v)=>v.strong_count()!=0,Self::Pattern(v)=>v.strong_count()!=0}}
    fn id(&self)->usize {match self {Self::Text(v)=>v.as_ptr() as *const () as usize,Self::Units(v)=>v.as_ptr() as *const () as usize,Self::Object(v)=>v.as_ptr() as usize,Self::Doc(v)=>v.as_ptr() as usize,Self::Big(v)=>v.as_ptr() as usize,Self::Regex(v)=>v.as_ptr() as usize,Self::Pattern(v)=>v.as_ptr() as usize}}
}

/// Java fails a String it cannot allocate with an OutOfMemoryError, which
/// Painless reports as a script_exception; Rust would abort the node on a
/// failed allocation. A large string a script asks for is therefore checked
/// first: Java's array-length limit, then the request breaker's free bytes.
pub(crate) fn check_string_bytes(bytes: u64) -> SResult<()> {
    if bytes <= 1 << 20 { return Ok(()); }
    if bytes > i32::MAX as u64 { return Err(out_of_memory("Required length exceeds implementation limit")); }
    // HotSpot's largest array is a few elements short of Integer.MAX_VALUE.
    if bytes > i32::MAX as u64 - 8 { return Err(out_of_memory("Requested array size exceeds VM limit")); }
    crate::memory::breakers().reserve_request(bytes, "<script string>").map(drop).map_err(|_| out_of_memory("Java heap space"))
}
pub(crate) fn out_of_memory(msg: &str) -> Box<ScriptErr> {
    ScriptErr::new("out_of_memory_error", msg)
}

/// Scalar registry size below which amortized sweeps never run.
const SWEEP_MIN: usize = 64;

pub(crate) struct Registry {
    breakers: &'static Breakers,
    storage: RefCell<Vec<Storage>>,
    scalars: RefCell<std::collections::HashMap<usize,(ScalarOwner,Allocation)>>,
    scalar_metadata: RefCell<Option<Reservation>>,
    reservation: RefCell<Option<Reservation>>,
    /// Scalar count at which the next amortized sweep runs.
    sweep_at: std::cell::Cell<usize>,
}
thread_local! {
    static ACTIVE: RefCell<Option<Rc<Registry>>> = const { RefCell::new(None) };
}

/// Only the root metric scope tears down collection cycles, after all phase
/// results have become JSON. Nested reduce and big-stack scopes share it.
pub(crate) struct Scope {
    previous: Option<Rc<Registry>>,
    active: Option<Rc<Registry>>,
    root: bool,
}
impl Scope {
    pub(crate) fn enter() -> Self {
        let previous = Self::current();
        let root = previous.is_none();
        let active = previous.clone().or_else(|| Some(Rc::new(Registry { breakers: crate::memory::breakers(), storage: RefCell::new(Vec::new()), scalars: RefCell::new(std::collections::HashMap::new()), scalar_metadata: RefCell::new(None), reservation: RefCell::new(None), sweep_at: std::cell::Cell::new(SWEEP_MIN) })));
        let mut scope = Self::install(active);
        scope.root = root;
        scope
    }
    pub(crate) fn install(active: Option<Rc<Registry>>) -> Self {
        let previous = ACTIVE.with(|slot| slot.replace(active.clone()));
        Self { previous, active, root: false }
    }
    pub(crate) fn current() -> Option<Rc<Registry>> { ACTIVE.with(|slot| slot.borrow().clone()) }
    /// Whether allocation accounting is installed on this thread (without
    /// the `Rc` round trip of [`Self::current`]). Interpreter fast paths
    /// that skip the accounting hooks run only when this is false; the hooks
    /// are all no-ops then.
    #[inline]
    pub(crate) fn active() -> bool { ACTIVE.with(|slot| slot.borrow().is_some()) }
    fn register(storage: Storage) -> SResult<()> {
        let Some(active) = Self::current() else { return Ok(()) };
        let mut entries = active.storage.borrow_mut();
        if entries.len() == entries.capacity() {
            entries.retain(|entry| match entry {
                Storage::List(v) => v.strong_count() != 0,
                Storage::Map(v) => v.strong_count() != 0,
            });
            if entries.len() == entries.capacity() {
                let capacity = entries.capacity().saturating_mul(2).max(4);
                let bytes = capacity.saturating_mul(std::mem::size_of::<Storage>()) as u64;
                let mut reservation = active.reservation.borrow_mut();
                match reservation.as_mut() {
                    Some(r) => r.grow(bytes.saturating_sub(r.bytes()), "<scripted_metric cycle registry>").map_err(ScriptErr::breaker)?,
                    None => *reservation = Some(active.breakers.reserve_request(bytes, "<scripted_metric cycle registry>").map_err(ScriptErr::breaker)?),
                }
                let extra = capacity - entries.len();
                entries.try_reserve_exact(extra).map_err(|_| ScriptErr::breaker(crate::error::EsError::new(429, "circuit_breaking_exception", "script registry allocation failed")))?;
            }
        }
        entries.push(storage);
        Ok(())
    }
    /// Release decoded scalar buffers once their last runtime alias dies.
    /// Called at every decode boundary (each doc-value access), so it sweeps
    /// only once the registry has doubled since the last sweep: amortized
    /// O(1) per call instead of O(live scalars), which made a metric keeping
    /// N strings O(N^2). Dead entries retained in between are bounded by the
    /// live ones (plus a small floor), and a breaker trip sweeps everything
    /// and retries before it fails (see [`Allocation::reserve`]).
    pub(super) fn collect_dead() {
        if let Some(scope)=Self::current() {
            if let Ok(mut scalars)=scope.scalars.try_borrow_mut() {
                if scalars.len()>=scope.sweep_at.get() {Self::sweep(&scope,&mut scalars);}
            }
        }
    }
    /// Sweep every dead scalar now; true when anything was released.
    pub(super) fn collect_all_dead()->bool {
        let Some(scope)=Self::current() else {return false};
        let Ok(mut scalars)=scope.scalars.try_borrow_mut() else {return false};
        let before=scalars.len();
        Self::sweep(&scope,&mut scalars);
        scalars.len()<before
    }
    fn sweep(scope:&Registry,scalars:&mut std::collections::HashMap<usize,(ScalarOwner,Allocation)>) {
        scalars.retain(|_,(owner,_)|owner.alive());
        scope.sweep_at.set(scalars.len().saturating_mul(2).max(SWEEP_MIN));
    }
    fn scalar(owner:ScalarOwner,bytes:u64)->SResult<()> {
        let Some(scope)=Self::current() else {return Ok(())};
        let id=owner.id();
        // A dead entry may share the address of a new buffer: replace it.
        {
            let mut scalars=scope.scalars.borrow_mut();
            match scalars.get(&id) {
                Some((existing,_)) if existing.alive() => return Ok(()),
                Some(_) => {scalars.remove(&id);}
                None => {}
            }
        }
        // Reserved before the registry is borrowed, so a trip can sweep it.
        let allocation=Allocation::new();
        allocation.reserve(bytes)?;
        let mut scalars=scope.scalars.borrow_mut();
        if scalars.len()==scalars.capacity() {
            let capacity=scalars.capacity().saturating_mul(2).max(4);
            let bytes=(capacity as u64).saturating_mul((std::mem::size_of::<(usize,ScalarOwner,Allocation)>()+32) as u64).saturating_mul(2);
            let mut reservation=scope.scalar_metadata.borrow_mut();
            match reservation.as_mut() {
                Some(r)=>r.grow(bytes.saturating_sub(r.bytes()),"<scripted_metric scalar registry>").map_err(ScriptErr::breaker)?,
                None=>*reservation=Some(scope.breakers.reserve_request(bytes,"<scripted_metric scalar registry>").map_err(ScriptErr::breaker)?),
            }
            let additional=capacity-scalars.len();
            scalars.try_reserve(additional).map_err(|_|ScriptErr::breaker(crate::error::EsError::new(429,"circuit_breaking_exception","script scalar registry allocation failed")))?;
        }
        scalars.insert(id,(owner,allocation));
        Ok(())
    }
    /// Transfer decoded values to their actual owners before releasing the
    /// transient before-decode reservation. Weak scalar entries follow aliases
    /// retained in metric state, without retaining dead document buffers.
    pub(super) fn adopt(value:&Val)->SResult<()> {
        if Self::current().is_none() {return Ok(())}
        fn walk(v:&Val,depth:usize)->SResult<()> {
            if depth>=128 {return Err(ScriptErr::breaker(crate::error::EsError::new(429,"circuit_breaking_exception","decoded script value nesting exceeds limit")));}
            match v {
                Val::Str(v)=>Scope::scalar(ScalarOwner::Text(Rc::downgrade(v)),(v.len() as u64).saturating_add(64))?,
                Val::Utf16(v)=>Scope::scalar(ScalarOwner::Units(Rc::downgrade(v)),(v.len() as u64).saturating_mul(2).saturating_add(64))?,
                Val::List(v)|Val::Set(v)|Val::Array{items:v,..}=>{v.reserve_growth(0)?;for v in v.borrow().iter(){walk(v,depth+1)?;}},
                Val::Map(v)=>{v.reserve_entry(&Key::Null)?;for(k,v) in v.borrow().iter(){walk(&k.to_val(),depth+1)?;walk(v,depth+1)?;}},
                Val::DocValues(v)=>{
                    Scope::scalar(ScalarOwner::Doc(Rc::downgrade(v)),(v.values.capacity() as u64).saturating_mul(std::mem::size_of::<Val>() as u64).saturating_add(v.field.capacity() as u64).saturating_add(128))?;
                    for v in &v.values {walk(v,depth+1)?;}
                },
                Val::Pattern(v)=>Scope::scalar(ScalarOwner::Pattern(std::sync::Arc::downgrade(v)),v.memory_bytes())?,
                Val::Big(v)=>{
                    let bits=match &**v {super::jbig::Big::Int(v)=>v.bits(),super::jbig::Big::Dec(v)=>v.u.bits(),_=>0};
                    Scope::scalar(ScalarOwner::Big(Rc::downgrade(v)),bits.saturating_add(7)/8+256)?;
                },
                Val::Obj(v)=>{
                    let bytes=match &**v {
                        super::jmisc::Obj::DenseVector(v)=>(v.capacity() as u64).saturating_mul(4).saturating_add(128),
                        super::jmisc::Obj::Version(v)=>(v.capacity() as u64).saturating_add(128),
                        _=>serialization_bound(&Val::Obj(v.clone()))?,
                    };
                    Scope::scalar(ScalarOwner::Object(Rc::downgrade(v)),bytes)?;
                },
                _=>{},
            }
            Ok(())
        }
        walk(value,0)
    }

    pub(super) fn adopt_regex(value:&std::sync::Arc<super::regex::CompiledRegex>)->SResult<()> {
        Self::scalar(ScalarOwner::Regex(std::sync::Arc::downgrade(value)),(value.memory_usage() as u64).saturating_add(256))
    }

    pub(super) fn register_list(data: &Rc<RefCell<Vec<Val>>>) -> SResult<()> {
        Self::register(Storage::List(Rc::downgrade(data)))
    }
    pub(super) fn register_map(data: &Rc<RefCell<IndexMap<Key, Val>>>) -> SResult<()> {
        Self::register(Storage::Map(Rc::downgrade(data)))
    }

}
impl Drop for Scope {
    fn drop(&mut self) {
        if self.root {
            if let Some(active) = &self.active {
                for storage in active.storage.borrow_mut().drain(..) {
                    match storage {
                        Storage::List(storage) => if let Some(data) = storage.upgrade() { data.borrow_mut().clear(); },
                        Storage::Map(storage) => if let Some(data) = storage.upgrade() { data.borrow_mut().clear(); },
                    }
                }
            }
        }
        ACTIVE.with(|slot| slot.replace(self.previous.take()));
    }
}

pub(super) struct Allocation {
    breakers: Option<&'static Breakers>,
    reservation: RefCell<Option<Reservation>>,
}
impl std::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricAllocation").field("enabled", &self.breakers.is_some()).finish()
    }
}
impl Allocation {
    pub(super) fn new() -> Self {
        Self { breakers: Scope::current().map(|scope| scope.breakers), reservation: RefCell::new(None) }
    }
    pub(super) fn values(count: usize) -> SResult<Self> {
        let a = Self::new();
        a.reserve((count as u64).saturating_mul(std::mem::size_of::<super::Val>() as u64).saturating_add(256))?;
        Ok(a)
    }
    pub(super) fn iterable(value: &super::Val) -> SResult<Self> {
        use super::Val;
        let count = match value {
            Val::List(v) | Val::Set(v) | Val::Array { items: v, .. } => v.len(),
            Val::Map(v) => v.borrow().len(),
            Val::DocValues(v) => v.values.len(),
            Val::Stream(v) => v.len(),
            Val::Str(v) => v.encode_utf16().count(),
            Val::Utf16(v) => v.len(),
            Val::Builder(v) => v.borrow().len(),
            Val::Obj(v) => match &**v { super::jmisc::Obj::Field { values, .. } => values.len(), _ => 0 },
            _ => 0, // Non-collection objects require their own allocation hook.
        };
        Self::values(count)
    }
    pub(super) fn entries(count: usize) -> SResult<Self> {
        let a = Self::new();
        a.reserve((count.max(4) as u64).saturating_mul(2).saturating_mul((std::mem::size_of::<(super::value::Key, super::Val)>() + 32) as u64))?;
        Ok(a)
    }
    pub(super) fn push(&self, values: &mut Vec<Val>, value: Val) -> SResult<()> {
        if values.len() == values.capacity() {
            let capacity = values.capacity().saturating_mul(2).max(4);
            self.reserve((capacity as u64).saturating_mul(std::mem::size_of::<Val>() as u64).saturating_add(256))?;
            values.try_reserve_exact(capacity - values.len()).map_err(|_| ScriptErr::breaker(crate::error::EsError::new(429, "circuit_breaking_exception", "script stream allocation failed")))?;
        }
        values.push(value);
        Ok(())
    }
    /// Reserve the transient rendering buffers before construction. The caller
    /// adopts the resulting scalar owner before dropping this guard.
    pub(super) fn scalar_text(values:&[&Val])->SResult<Self> {
        Scope::collect_dead();
        let allocation=Self::new();
        if allocation.enabled() {
            let mut bytes=0u64;
            for value in values {bytes=bytes.saturating_add(serialization_bound(value)?);}
            allocation.reserve(bytes)?;
        }
        Ok(allocation)
    }
    pub(super) fn text_copy(value:&Val)->SResult<Self> {
        Scope::collect_dead();
        let allocation=Self::new();
        if allocation.enabled() {allocation.reserve(serialization_bound(value)?)?;}
        Ok(allocation)
    }
    pub(super) fn enabled(&self) -> bool { self.breakers.is_some() }
    pub(super) fn reserve(&self, total: u64) -> SResult<()> {
        match self.reserve_once(total) {
            // Dead scalars may still hold bytes between amortized sweeps;
            // release them before reporting a trip.
            Err(error) if error.breaker.is_some() && Scope::collect_all_dead() => self.reserve_once(total),
            result => result,
        }
    }
    fn reserve_once(&self, total: u64) -> SResult<()> {
        let Some(breakers) = self.breakers else { return Ok(()) };
        let mut reservation = self.reservation.borrow_mut();
        match reservation.as_mut() {
            Some(r) => if total > r.bytes() {
                r.grow(total - r.bytes(), "<scripted_metric collection>").map_err(ScriptErr::breaker)?;
            },
            None => *reservation = Some(breakers.reserve_request(total, "<scripted_metric collection>").map_err(ScriptErr::breaker)?),
        }
        Ok(())
    }
}


/// Borrowed conservative rendering bound shared by metric serialization and
/// Java-text producing runtime calls. Metadata storage is charged too.
pub(crate) fn serialization_bound(value:&Val)->SResult<u64> {
    let scratch=Allocation::new();
    output_bytes(value,&mut Vec::new(),&scratch,&Allocation::new(),&mut 0,0)
}

fn output_bytes(value: &Val, path: &mut Vec<usize>, scratch: &Allocation, output:&Allocation, spent:&mut u64, depth:usize) -> SResult<u64> {
    if depth>=128 {return Err(ScriptErr::breaker(crate::error::EsError::new(429,"circuit_breaking_exception","script output nesting exceeds bounded serializer depth")));}
    let start=*spent;
    *spent=spent.saturating_add(512);
    output.reserve(*spent)?;
    use std::rc::Rc;
    let address = match value {
        Val::Map(v) => Some(Rc::as_ptr(v) as usize),
        Val::List(v) | Val::Set(v) | Val::Array { items: v, .. } => Some(Rc::as_ptr(v) as usize),
        Val::Stream(v) => Some(Rc::as_ptr(v) as usize),
        _ => None,
    };
    if let Some(address) = address {
        if path.contains(&address) { return Err(super::error::illegal_arg("Iterable object is self-referencing itself")); }
        if path.len() == path.capacity() {
            let capacity = path.capacity().saturating_mul(2).max(8);
            scratch.reserve((capacity * std::mem::size_of::<usize>()) as u64)?;
            path.try_reserve_exact(capacity - path.len()).map_err(|_| ScriptErr::breaker(crate::error::EsError::new(429,"circuit_breaking_exception","script serialization stack allocation failed")))?;
        }
        path.push(address);
    }
    let mut bytes = 512u64;
    match value {
        Val::Str(v) => bytes = bytes.saturating_add((v.len() as u64).saturating_mul(48)),
        Val::Utf16(v) => bytes = bytes.saturating_add((v.len() as u64).saturating_mul(48)),
        Val::Builder(v) => bytes = bytes.saturating_add((v.borrow().len() as u64).saturating_mul(48)),
        Val::Exception(v) => bytes = bytes.saturating_add(((v.0.len() as u64).saturating_add(v.1.len() as u64)).saturating_mul(48)),
        Val::Pattern(v) => bytes = bytes.saturating_add((v.source.len() as u64).saturating_mul(48)),
        Val::Matcher(v) => bytes = bytes.saturating_add((v.borrow().pat.source.len() as u64).saturating_mul(48)),
        Val::Time(v) => bytes=bytes.saturating_add(v.text_bound().saturating_mul(48)),
        Val::Big(v) => {
            use crate::script::jbig::Big;
            // A binary bit per decimal digit is an upper bound; scientific
            // BigDecimal text adds only a bounded exponent/decimal suffix.
            let digits=match &**v { Big::Int(v)=>v.bits(),Big::Dec(v)=>v.u.bits(),_=>0 };
            bytes=bytes.saturating_add(digits.saturating_mul(48));
        },
        Val::Obj(v) => {
            use crate::script::jmisc::Obj;
            let units=match &**v {
                Obj::Version(v)=>v.len() as u64,
                Obj::Locale{language,country,variant}=>(language.len() as u64).saturating_add(country.len() as u64).saturating_add(variant.len() as u64).saturating_add(2),
                Obj::Joiner{sep,prefix,suffix,parts,empty,..}=>{
                    let parts=parts.borrow();
                    if parts.is_empty() { empty.borrow().as_ref().map(|v|v.len() as u64).unwrap_or((prefix.len() as u64).saturating_add(suffix.len() as u64)) }
                    else { parts.iter().fold((prefix.len() as u64).saturating_add(suffix.len() as u64).saturating_add((sep.len() as u64).saturating_mul(parts.len().saturating_sub(1) as u64)),|n,v|n.saturating_add(v.len() as u64)) }
                },
                _=>0,
            };
            bytes=bytes.saturating_add(units.saturating_mul(48));
        },
        Val::Map(v) => for (key,value) in v.borrow().iter() {
            bytes = bytes.saturating_add(output_bytes(&key.to_val(),path,scratch,output,spent,depth+1)?).saturating_add(output_bytes(value,path,scratch,output,spent,depth+1)?);
        },
        Val::List(v) | Val::Set(v) | Val::Array { items:v, .. } => for value in v.borrow().iter() { bytes=bytes.saturating_add(output_bytes(value,path,scratch,output,spent,depth+1)?); },
        Val::Stream(v) => for value in v.iter() { bytes=bytes.saturating_add(output_bytes(value,path,scratch,output,spent,depth+1)?); },
        Val::DocValues(v) => for value in &v.values { bytes=bytes.saturating_add(output_bytes(value,path,scratch,output,spent,depth+1)?); },
        Val::Optional(Some(v)) => bytes=bytes.saturating_add(output_bytes(v,path,scratch,output,spent,depth+1)?),
        Val::Entry(v) => bytes=bytes.saturating_add(output_bytes(&v.0,path,scratch,output,spent,depth+1)?).saturating_add(output_bytes(&v.1,path,scratch,output,spent,depth+1)?),
        _ => {},
    }
    *spent=start.saturating_add(bytes);
    output.reserve(*spent)?;
    if address.is_some() { path.pop(); }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Val;
    #[test]
    fn json_library_buffers_obey_scoped_accounting_in_allowed_context() {
        let breakers=Breakers::leak_for_tests(1<<30);
        for retained in [false,true] {
            for limit in ["64kb","16mb"] {
                breakers.apply_settings(&|key|match key {
                    "indices.breaker.request.limit"=>Some(limit.into()),
                    "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
                });
                // Json is a Watcher API, not allowed in metric scripts. Exercise
                // the shared runtime with accounting installed independently.
                let source=if retained {
                    "def copies=[]; for(int i=0;i<100;i++){copies.add(Json.load('[1,2,3,4,5,6,7,8,9,10]'));} return copies.size();"
                } else {
                    "for(int i=0;i<1000;i++){def v=Json.load('{\"value\":\"abcdefghij\"}');} return 1;"
                };
                let compiled=super::super::compile(source,&serde_json::json!({}),super::super::ScriptContext::Watcher).unwrap();
                let request=crate::memory::RequestScope::new("JSON buffers",breakers);
                let result=crate::memory::scoped(Some(request),|| {
                    let _scope=Scope::enter();
                    compiled.executor(None,None).run()
                });
                if retained && limit=="64kb" { assert_eq!(result.unwrap_err().breaker.unwrap().status,429); }
                else { assert_eq!(result.unwrap().to_i64().unwrap(),if retained {100}else{1}); }
                assert_eq!(breakers.request.reserved(),0);
            }
        }
    }

    #[test]
    fn string_repeat_runtime_checks_before_growth() {
        let breakers=Breakers::leak_for_tests(1<<30);
        for limit in ["64kb","64mb"] {
            breakers.apply_settings(&|key|match key {
                "indices.breaker.request.limit"=>Some(limit.into()),
                "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
            });
            let request=crate::memory::RequestScope::new("repeat buffer",breakers);
            let result=crate::memory::scoped(Some(request),|| {
                let _scope=Scope::enter();
                super::super::source::string_method(&Val::str("x"),&[120],"repeat",&[Val::Int(100000)])
            });
            if limit=="64kb" { assert_eq!(result.unwrap_err().breaker.unwrap().status,429); }
            else { assert_eq!(result.unwrap().string_units().len(),100000); }
            assert_eq!(breakers.request.reserved(),0);
        }
    }

    #[test]
    fn amortized_scalar_sweeps_stay_bounded_and_never_cause_a_trip() {
        let breakers = Breakers::leak_for_tests(1 << 30);
        breakers.apply_settings(&|key| match key {
            "indices.breaker.request.limit" => Some("64kb".into()),
            "indices.breaker.total.limit" => Some("-1".into()),
            _ => None,
        });
        let request = crate::memory::RequestScope::new("amortized scalar sweeps", breakers);
        crate::memory::scoped(Some(request), || {
            let scope = Scope::enter();
            let live: Vec<Val> = (0..100).map(|i| { let v = Val::str(&format!("live-{i}")); Scope::adopt(&v).unwrap(); v }).collect();
            for _ in 0..5000 {
                // 4kb each: 16 dead ones exceed the 64kb limit unless the
                // trip path sweeps them first.
                let temporary = Val::str(&"y".repeat(4096));
                Scope::adopt(&temporary).unwrap();
                drop(temporary);
                Scope::collect_dead();
                let registered = Scope::current().unwrap().scalars.borrow().len();
                assert!(registered <= 2 * live.len() + SWEEP_MIN, "{registered} scalars retained");
            }
            drop(live);
            assert!(Scope::collect_all_dead());
            drop(scope);
            assert_eq!(breakers.request.reserved(), 0);
        });
    }

    #[test]
    fn decoded_scalar_reservations_follow_live_aliases() {
        let breakers = Breakers::leak_for_tests(1 << 30);
        breakers.apply_settings(&|key| match key {
            "indices.breaker.request.limit" => Some("64kb".into()),
            "indices.breaker.total.limit" => Some("-1".into()),
            _ => None,
        });
        let request = crate::memory::RequestScope::new("decoded scalar aliases", breakers);
        crate::memory::scoped(Some(request), || {
            let scope = Scope::enter();
            let retained = Val::str(&"x".repeat(4096));
            Scope::adopt(&retained).unwrap();
            let retained_bytes = breakers.request.reserved();
            let alias = retained.clone();
            drop(retained);
            Scope::collect_all_dead();
            assert_eq!(breakers.request.reserved(), retained_bytes);
            for _ in 0..1000 {
                let temporary = Val::str(&"y".repeat(4096));
                Scope::adopt(&temporary).unwrap();
                drop(temporary);
                Scope::collect_all_dead();
                assert_eq!(breakers.request.reserved(), retained_bytes);
            }
            drop(alias);
            Scope::collect_all_dead();
            assert!(breakers.request.reserved() < retained_bytes);
            drop(scope);
            assert_eq!(breakers.request.reserved(), 0);
        });
    }

    #[test]
    fn abandoned_metric_scope_reclaims_cyclic_collection_storage() {
        let _budget = rustic_core::admission::private_shared(256 << 20);
        let breakers = Breakers::private_for_tests(1 << 30);
        let request = crate::memory::RequestScope::new("metric cancelled", breakers);
        crate::memory::scoped(Some(request), || {
            {
                let _scope = Scope::enter();
                let value = Val::new_list(Vec::new());
                let Val::List(list) = &value else { unreachable!() };
                list.reserve_growth(1).unwrap();
                list.borrow_mut().push(value.clone());
                assert!(breakers.request.reserved() > 0);
                // Dropping the execution scope models cancellation before result
                // serialization, when no interpreter root remains reachable.
            }
            assert_eq!(breakers.request.reserved(), 0);
        });
    }

    #[test]
    fn collection_reservation_follows_aliases_and_releases_on_last_drop() {
        let _budget = rustic_core::admission::private_shared(256 << 20);
        let breakers = Breakers::private_for_tests(1 << 30);
        let request = crate::memory::RequestScope::new("metric lifetime", breakers);
        crate::memory::scoped(Some(request), || {
        let _scope = Scope::enter();
        let value = Val::new_list(Vec::new());
        let Val::List(list) = &value else { unreachable!() };
        list.reserve_growth(100).unwrap();
        let reserved = breakers.request.reserved();
        assert!(reserved >= 100 * std::mem::size_of::<Val>() as u64);
        let alias = value.clone();
        drop(value);
        assert_eq!(breakers.request.reserved(), reserved);
        drop(alias);
        // Registry capacity remains owned until the metric root exits.
        assert!(breakers.request.reserved() < reserved);
        drop(_scope);
        assert_eq!(breakers.request.reserved(), 0);
        });
    }
}
