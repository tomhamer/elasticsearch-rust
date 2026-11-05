//! Aggregate admission for owned core caches and transient workspaces.
//! Reserve before growth; move the guard with the allocation, including
//! shared readers that outlive cache eviction. This is not whole-RSS admission.
use crate::error::{Result,RusticError};
use std::sync::{Arc,Weak,Mutex,OnceLock,atomic::{AtomicBool,AtomicU64,Ordering}};

/// Actual admitted-resource ownership, distinct from RSS or estimated heap.
/// A scope attributes allocations made on this thread; guards retain their
/// owner across thread transfers. Callers must report partial coverage until
/// every relevant worker and allocation category is explicitly propagated.
#[derive(Debug, Default)]
pub struct Owner { current:AtomicU64, peak:AtomicU64, categories:[CategoryCounter;5], unplanned:AtomicBool }
#[derive(Debug,Default)]
struct CategoryCounter {current:AtomicU64,peak:AtomicU64}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
#[repr(usize)]
pub enum Category { Core=0, EngineRequest=1, Fielddata=2, Eql=3, Inference=4 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnedBytes { pub current:u64, pub peak:u64 }
thread_local! { static OWNER:std::cell::RefCell<Option<Arc<Owner>>>=const{std::cell::RefCell::new(None)}; }
pub struct OwnerScope { previous:Option<Arc<Owner>>, _thread:std::marker::PhantomData<std::rc::Rc<()>> }
impl Owner {
    pub fn new()->Arc<Self>{Arc::new(Self::default())}
    pub fn enter(self:&Arc<Self>)->OwnerScope {
        let previous=OWNER.with(|slot|slot.replace(Some(self.clone())));
        OwnerScope{previous,_thread:std::marker::PhantomData}
    }
    pub fn bytes(&self)->OwnedBytes {OwnedBytes{current:self.current.load(Ordering::Acquire),peak:self.peak.load(Ordering::Acquire)}}
    pub fn has_unplanned_allocations(&self)->bool {self.unplanned.load(Ordering::Acquire)}
    pub fn category_bytes(&self,category:Category)->OwnedBytes {let c=&self.categories[category as usize];OwnedBytes{current:c.current.load(Ordering::Acquire),peak:c.peak.load(Ordering::Acquire)}}
    fn add(&self,bytes:u64,category:Category)->Result<()> {
        let previous=self.current.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|v.checked_add(bytes))
            .map_err(|_|RusticError::Memory("admitted owner byte counter overflow".into()))?;
        raise_peak(&self.peak,previous+bytes);
        let counter=&self.categories[category as usize];
        let before=counter.current.fetch_add(bytes,Ordering::AcqRel);
        raise_peak(&counter.peak,before+bytes);Ok(())
    }
    fn sub(&self,bytes:u64,category:Category){self.categories[category as usize].current.fetch_sub(bytes,Ordering::AcqRel);self.current.fetch_sub(bytes,Ordering::AcqRel);}
}
impl Drop for OwnerScope {fn drop(&mut self){OWNER.with(|slot|*slot.borrow_mut()=self.previous.take());}}
/// `peak.fetch_max(value)`, skipping the read-modify-write when `value` is
/// not a new peak (the common case on every admission once an owner is
/// warm). A peak only ever rises, so a stale smaller load at worst falls
/// through to the same `fetch_max`; a load at or above `value` means
/// `fetch_max` would not change it. Same values as an unconditional
/// `fetch_max`, without a cross-core exclusive cache-line acquisition.
#[inline]
fn raise_peak(peak:&AtomicU64,value:u64){if peak.load(Ordering::Acquire)<value{peak.fetch_max(value,Ordering::AcqRel);}}

#[derive(Debug)]
pub struct Budget {limit:AtomicU64,used:AtomicU64,
    /// Bytes held by background reservations ([`Budget::reserve_background`]),
    /// also counted in `used`: the background share caps these alone.
    background:AtomicU64}
impl Budget {
    pub fn new(limit:u64)->Arc<Self> {Arc::new(Self{limit:AtomicU64::new(limit),used:AtomicU64::new(0),background:AtomicU64::new(0)})}
    /// Bytes background reservations hold now (part of [`Budget::used`]).
    pub fn background_used(&self)->u64 {self.background.load(Ordering::Acquire)}
    pub fn used(&self)->u64 {self.used.load(Ordering::Acquire)}
    pub fn limit(&self)->u64 {self.limit.load(Ordering::Acquire)}
    /// Lowering the cap never invalidates existing readers. Further growth
    /// is rejected until their retained reservations fall below the new cap.
    pub fn set_limit(&self,limit:u64) {self.limit.store(limit,Ordering::Release);}
    fn add(&self,bytes:u64,label:&str)->Result<()> {
        // A lowered limit blocks new storage, not allocation-free owners.
        if bytes == 0 { return Ok(()); }
        if is_forced() {
            return self.used.fetch_update(Ordering::AcqRel,Ordering::Acquire,|used|used.checked_add(bytes))
                .map(|_|()).map_err(|used|RusticError::Memory(format!("[core] [{label}] needs {bytes} bytes, retained {used}: counter overflow")));
        }
        self.used.fetch_update(Ordering::AcqRel,Ordering::Acquire,|used|used.checked_add(bytes).filter(|next|*next<=self.limit()))
            .map(|_|()).map_err(|used|RusticError::Memory(format!("[core] [{label}] needs {bytes} bytes, retained {used}, limit {}",self.limit())))
    }
    pub fn reserve(self:&Arc<Self>,bytes:u64,label:&str)->Result<Reservation> {self.reserve_categorized(bytes,label,Category::Core)}
    pub fn reserve_categorized(self:&Arc<Self>,bytes:u64,label:&str,category:Category)->Result<Reservation> {
        let lease=LEASE.with(|slot|slot.borrow().clone());
        if let Some(lease)=lease.filter(|lease|Arc::ptr_eq(&lease.budget,self)) {
            if !lease.completed.load(Ordering::Acquire) {
                return lease.borrow(bytes, label, OWNER.with(|slot|slot.borrow().clone()),category);
            }
        }
        self.add(bytes,label)?;
        let owner=OWNER.with(|slot|slot.borrow().clone());
        if let Some(owner)=&owner {if let Err(e)=owner.add(bytes,category){self.used.fetch_sub(bytes,Ordering::AcqRel);return Err(e);}}
        Ok(Reservation{budget:self.clone(),bytes,owner,category,lease:None,background:false})
    }
    /// Background maintenance (segment merges): its own working memory is
    /// admitted against the background share (`limit - limit /
    /// BACKGROUND_HEADROOM_DIVISOR`) of what background reservations hold,
    /// not of everything the budget holds. Resident index data (vector rows,
    /// graphs, caches) never blocks a merge, as ES gates no merge on memory
    /// (its breakers guard request paths only): with the resident data past
    /// the share, every background merge was refused, retried and refused
    /// again, and merges stopped for good. The bytes are still charged to
    /// `used` (foreground admissions see them). Short of share, it waits for
    /// other background owners to release rather than failing, up to `wait`;
    /// a request larger than the share can never fit and fails at once
    /// (callers split their work instead). Under a recovery lease it borrows
    /// plan credit like [`Self::reserve`]. Under [`forced`] replay it is a
    /// forced [`Self::reserve`]: charged past the limit, never waiting and
    /// never refused.
    pub fn reserve_background(self:&Arc<Self>,bytes:u64,label:&str,wait:std::time::Duration)->Result<Reservation> {
        if is_forced() {return self.reserve(bytes,label);}
        if LEASE.with(|slot|slot.borrow().as_ref().is_some_and(|lease|Arc::ptr_eq(&lease.budget,self))) {return self.reserve(bytes,label);}
        let deadline=std::time::Instant::now()+wait;
        let mut pause=std::time::Duration::from_millis(1);
        loop {
            match self.add_background(bytes,label) {
                Ok(()) => {
                    let owner=OWNER.with(|slot|slot.borrow().clone());
                    if let Some(owner)=&owner {if let Err(e)=owner.add(bytes,Category::Core){self.sub_background(bytes);return Err(e);}}
                    return Ok(Reservation{budget:self.clone(),bytes,owner,category:Category::Core,lease:None,background:true});
                }
                Err(error) if bytes>self.background_limit() => return Err(error),
                Err(error) => {
                    let now=std::time::Instant::now();
                    if now>=deadline {return Err(error);}
                    std::thread::sleep(pause.min(deadline-now));
                    pause=(pause*2).min(std::time::Duration::from_millis(50));
                }
            }
        }
    }
    /// Charge `bytes` of background work: refused only past the background
    /// share of background-held bytes; `used` takes them unconditionally.
    fn add_background(&self,bytes:u64,label:&str)->Result<()> {
        if bytes==0 {return Ok(());}
        let cap=self.background_limit();
        self.background.fetch_update(Ordering::AcqRel,Ordering::Acquire,|held|held.checked_add(bytes).filter(|next|*next<=cap))
            .map_err(|held|RusticError::Memory(format!("[core] [{label}] needs {bytes} bytes, background work holds {held}, background share {cap} of limit {} (retained {})",self.limit(),self.used())))?;
        self.used.fetch_add(bytes,Ordering::AcqRel);
        Ok(())
    }
    fn sub_background(&self,bytes:u64) {
        self.background.fetch_sub(bytes,Ordering::AcqRel);
        self.used.fetch_sub(bytes,Ordering::AcqRel);
    }
    /// The part of the limit background reservations may fill.
    pub fn background_limit(&self)->u64 {let limit=self.limit();limit-limit/BACKGROUND_HEADROOM_DIVISOR}
}
/// See [`Budget::reserve_background`]: a quarter of the budget is foreground-only.
pub const BACKGROUND_HEADROOM_DIVISOR:u64=4;
#[derive(Debug)]
pub struct Reservation {budget:Arc<Budget>,bytes:u64,owner:Option<Arc<Owner>>,category:Category,lease:Option<LeaseBinding>,
    /// Admitted by [`Budget::reserve_background`] (counted in its share).
    background:bool}
impl Reservation {
    pub fn bytes(&self)->u64 {self.bytes}
    /// Transfer already admitted bytes, preserving category and lease.
    pub fn split(&mut self,bytes:u64)->Option<Self>{
        if bytes>self.bytes{return None;}
        self.bytes-=bytes;
        Some(Self{budget:self.budget.clone(),bytes,owner:self.owner.clone(),category:self.category,lease:self.lease.as_ref().map(|binding|LeaseBinding{lease:binding.lease.clone(),completed:binding.completed.clone()}),background:self.background})
    }
    /// Independently owned allocation charged to the same budget.
    pub fn reserve_related(&self, bytes:u64, label:&str)->Result<Reservation> {
        if let Some(binding)=&self.lease {
            // Successful recovery hands resident owners back to independent
            // ordinary admission. Cancellation does not grant that transition.
            if !binding.completed.load(Ordering::Acquire) {
                return binding.lease.upgrade().ok_or_else(||RusticError::Memory("recovery admission lease expired".into()))?
                    .borrow(bytes,label,self.owner.clone(),self.category);
            }
        }
        self.budget.add(bytes,label)?;
        if let Some(owner)=&self.owner {if let Err(e)=owner.add(bytes,self.category){self.budget.used.fetch_sub(bytes,Ordering::AcqRel);return Err(e);}}
        Ok(Reservation{budget:self.budget.clone(),bytes,owner:self.owner.clone(),category:self.category,lease:None,background:false})
    }
    pub fn grow(&mut self,more:u64,label:&str)->Result<()> {
        let next=self.bytes.checked_add(more).ok_or_else(||RusticError::Memory("reservation size overflow".into()))?;
        if self.lease.as_ref().is_some_and(|binding|!binding.completed.load(Ordering::Acquire)) {
            // Recovery plan credit: borrow through the lease, then transfer
            // the separately admitted bytes, including their owner charge.
            let mut additional=self.reserve_related(more,label)?;
            additional.bytes=0;
            self.bytes=next;return Ok(());
        }
        if self.background {
            // Background work grows within the background share.
            self.budget.add_background(more,label)?;
            if let Some(owner)=&self.owner {if let Err(e)=owner.add(more,self.category){self.budget.sub_background(more);return Err(e);}}
            self.bytes=next;return Ok(());
        }
        // Ordinary admission: exactly `reserve_related` + transfer, without
        // the temporary guard (its Arc clones, and its zero-byte release of
        // the owner and budget counters, were ~9 atomic RMWs per growth).
        self.budget.add(more,label)?;
        if let Some(owner)=&self.owner {if let Err(e)=owner.add(more,self.category){self.budget.used.fetch_sub(more,Ordering::AcqRel);return Err(e);}}
        self.bytes=next;Ok(())
    }
    fn release(&mut self,bytes:u64) {
        let bytes=bytes.min(self.bytes);
        // Nothing to return (empty guards, zero shrinks): every counter
        // update below would add 0. Skip their atomics and the lease lock.
        if bytes==0 {return;}
        self.bytes-=bytes;
        if let Some(owner)=&self.owner{owner.sub(bytes,self.category);}
        if let Some(lease)=self.lease.as_ref().and_then(|binding|binding.lease.upgrade()) {
            let mut state=lease.state.lock().unwrap_or_else(|e|e.into_inner());
            if state.active {state.available+=bytes;return;}
        }
        if self.background {self.budget.background.fetch_sub(bytes,Ordering::AcqRel);}
        self.budget.used.fetch_sub(bytes,Ordering::AcqRel);
    }
    pub fn shrink(&mut self,bytes:u64) {self.release(bytes);}
}
impl Drop for Reservation {fn drop(&mut self){self.release(self.bytes);}}

/// A pre-admitted allocation plan. Credits become actual allocation guards
/// without a second aggregate charge. Expiration releases only unused credit;
/// allocations already handed out remain charged until their owners drop.
/// This primitive does not certify a plan's coverage or its cluster authority.
#[derive(Debug)]
pub struct Lease {budget:Arc<Budget>,state:Mutex<LeaseState>,completed:Arc<AtomicBool>,unplanned:AtomicBool}
#[derive(Debug)]
struct LeaseBinding {lease:Weak<Lease>,completed:Arc<AtomicBool>}
#[derive(Debug)]
struct LeaseState {active:bool,available:u64,last_allocation:std::time::Instant}
thread_local! {static LEASE:std::cell::RefCell<Option<Arc<Lease>>>=const{std::cell::RefCell::new(None)};}
pub struct LeaseScope {previous:Option<Arc<Lease>>,_thread:std::marker::PhantomData<std::rc::Rc<()>>}
impl Lease {
    pub fn reserve(budget:Arc<Budget>,bytes:u64,label:&str)->Result<Arc<Self>> {
        // Plan credit is deliberately not attributed to the current shard.
        // Actual child allocations acquire their owner when they borrow it.
        budget.add(bytes,label)?;
        Ok(Arc::new(Self{budget,state:Mutex::new(LeaseState{active:true,available:bytes,last_allocation:std::time::Instant::now()}),completed:Arc::new(AtomicBool::new(false)),unplanned:AtomicBool::new(false)}))
    }
    pub fn enter(self:&Arc<Self>)->LeaseScope {
        let previous=LEASE.with(|slot|slot.replace(Some(self.clone())));
        LeaseScope{previous,_thread:std::marker::PhantomData}
    }
    pub fn expire(&self) {
        let mut state=self.state.lock().unwrap_or_else(|e|e.into_inner());
        state.active=false;
        self.budget.used.fetch_sub(std::mem::take(&mut state.available),Ordering::AcqRel);
    }
    /// Call only after the recovery's authority/installation fence succeeds.
    /// Existing allocations retain their charges; subsequent resident growth
    /// uses ordinary shared admission rather than an expired recovery plan.
    /// A canceled/expired lease cannot later be completed.
    pub fn complete(&self)->Result<()> {
        let mut state=self.state.lock().unwrap_or_else(|e|e.into_inner());
        if !state.active || self.unplanned.load(Ordering::Acquire) {return Err(RusticError::Memory("cannot complete expired or incompletely admitted recovery lease".into()));}
        state.active=false;
        self.budget.used.fetch_sub(std::mem::take(&mut state.available),Ordering::AcqRel);
        self.completed.store(true,Ordering::Release);
        Ok(())
    }
    pub fn is_active(&self)->bool {self.state.lock().unwrap_or_else(|e|e.into_inner()).active}
    pub fn available(&self)->u64 {self.state.lock().unwrap_or_else(|e|e.into_inner()).available}
    /// Only successful nonempty allocation progress renews an executing
    /// envelope. Polling, retries and empty reservations cannot keep it alive.
    pub fn last_allocation(&self)->std::time::Instant {self.state.lock().unwrap_or_else(|e|e.into_inner()).last_allocation}
    fn borrow(self:&Arc<Self>,bytes:u64,label:&str,owner:Option<Arc<Owner>>,category:Category)->Result<Reservation> {
        let mut state=self.state.lock().unwrap_or_else(|e|e.into_inner());
        if !state.active || state.available<bytes {
            if is_forced() {
                // Forced work past its plan is charged to the budget itself
                // (still counted, never refused), outside the lease.
                drop(state);
                self.budget.add(bytes,label)?;
                if let Some(owner)=&owner {if let Err(e)=owner.add(bytes,category){self.budget.used.fetch_sub(bytes,Ordering::AcqRel);return Err(e);}}
                return Ok(Reservation{budget:self.budget.clone(),bytes,owner,category,lease:None,background:false});
            }
            return Err(RusticError::Memory(format!("[core] [{label}] recovery lease has insufficient or expired credits")));
        }
        if let Some(owner)=&owner {owner.add(bytes,category)?;}
        state.available-=bytes;
        if bytes!=0 {state.last_allocation=std::time::Instant::now();}
        Ok(Reservation{budget:self.budget.clone(),bytes,owner,category,lease:Some(LeaseBinding{lease:Arc::downgrade(self),completed:self.completed.clone()}),background:false})
    }
}
impl Drop for Lease {fn drop(&mut self){self.expire();}}
impl Drop for LeaseScope {fn drop(&mut self){LEASE.with(|slot|*slot.borrow_mut()=self.previous.take());}}
/// Explicitly carry admission identity across an executor boundary. This is
/// data, not a TLS guard: it is Send/Sync and installs non-Send scopes only on
/// the worker running the closure. Empty contexts clear unrelated worker TLS.
#[derive(Clone,Default)]
pub struct Context {owner:Option<Arc<Owner>>,lease:Option<Arc<Lease>>,forced:bool}
pub struct ContextScope {_owner:OwnerScope,_lease:LeaseScope,_forced:ForcedScope}
impl Context {
    /// Infallible/accounting-only allocations cannot certify a recovery plan.
    pub fn record_unplanned(&self) {if let Some(owner)=&self.owner{owner.unplanned.store(true,Ordering::Release);}if let Some(lease)=&self.lease{lease.unplanned.store(true,Ordering::Release);}}
    pub fn is_attributed(&self)->bool {self.owner.is_some()||self.lease.is_some()}
    /// Whether work run in this context is [`forced`].
    pub fn is_forced(&self)->bool {self.forced}
    pub fn capture()->Self {Self{owner:OWNER.with(|v|v.borrow().clone()),lease:LEASE.with(|v|v.borrow().clone()),forced:is_forced()}}
    pub fn enter(&self)->ContextScope {
        let previous=OWNER.with(|v|v.replace(self.owner.clone()));
        let _owner=OwnerScope{previous,_thread:std::marker::PhantomData};
        let previous=LEASE.with(|v|v.replace(self.lease.clone()));
        let _lease=LeaseScope{previous,_thread:std::marker::PhantomData};
        let _forced=ForcedScope{previous:FORCED.with(|v|v.replace(self.forced)),_thread:std::marker::PhantomData};
        ContextScope{_owner,_lease,_forced}
    }
    pub fn run<T>(&self, f:impl FnOnce()->T)->T {let _scope=self.enter();f()}
    /// Scope each individual poll, never hold thread-local guards across await.
    pub async fn scope_future<F:std::future::Future>(&self,future:F)->F::Output {
        let mut future=std::pin::pin!(future);
        std::future::poll_fn(|cx| {let _scope=self.enter();future.as_mut().poll(cx)}).await
    }
    pub fn with_lease(lease:Arc<Lease>)->Self {Self{owner:None,lease:Some(lease),forced:false}}
}

thread_local! {static FORCED:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}
struct ForcedScope {previous:bool,_thread:std::marker::PhantomData<std::rc::Rc<()>>}
impl Drop for ForcedScope {fn drop(&mut self){FORCED.with(|v|v.set(self.previous));}}
/// Run `f` with every reservation it makes on this thread charged even past
/// the limit instead of refused: the bytes stay accounted, nothing waits or
/// fails. Only for replaying already-acknowledged operations (ES applies
/// translog recovery with no memory check: refusing them cannot free
/// anything, and nothing else can recover them). [`Context`] carries it to
/// the workers such work runs on.
pub fn forced<T>(f:impl FnOnce()->T)->T {
    let _scope=ForcedScope{previous:FORCED.with(|v|v.replace(true)),_thread:std::marker::PhantomData};
    f()
}
/// Whether this thread runs [`forced`] work.
pub fn is_forced()->bool {FORCED.with(|v|v.get())}

/// Keep an admitted allocation and its reservation under the same Arc.
/// Cloning the Arc never releases admission before the last reader drops.
#[derive(Debug)]
pub struct Allocation<T> {value:T,_memory:Reservation}
impl<T> Allocation<T> {pub fn new(value:T,memory:Reservation)->Self{Self{value,_memory:memory}} pub fn into_parts(self)->(T,Reservation){(self.value,self._memory)}}
impl<T> std::ops::Deref for Allocation<T> {type Target=T;fn deref(&self)->&T{&self.value}}
pub fn shared()->Arc<Budget> {
    #[cfg(any(test, feature = "test-seams"))]
    if let Some(budget)=TEST_SHARED.with(|slot|slot.borrow().clone()) {return budget;}
    static SHARED:OnceLock<Arc<Budget>>=OnceLock::new();
    SHARED.get_or_init(||Budget::new(256<<20)).clone()
}

// Test isolation only (never in a production build: compiled for this
// crate's tests and, through the `test-seams` feature that dependent crates
// enable only as a dev-dependency, for theirs): unit tests share one
// process and so one `shared()` budget; a test whose reservations reach it
// (row planes, code logs, engine breaker provenance) is refused when its
// neighbours happen to hold the rest. [`private_shared`] gives the calling
// test thread its own.
#[cfg(any(test, feature = "test-seams"))]
thread_local! {static TEST_SHARED:std::cell::RefCell<Option<Arc<Budget>>>=const{std::cell::RefCell::new(None)};}
/// Until dropped, [`shared`] on this thread is a fresh private budget of
/// `limit` bytes (test builds only).
#[cfg(any(test, feature = "test-seams"))]
#[doc(hidden)]
pub struct PrivateShared {previous:Option<Arc<Budget>>,_thread:std::marker::PhantomData<std::rc::Rc<()>>}
#[cfg(any(test, feature = "test-seams"))]
#[doc(hidden)]
pub fn private_shared(limit:u64)->PrivateShared {
    let previous=TEST_SHARED.with(|slot|slot.replace(Some(Budget::new(limit))));
    PrivateShared{previous,_thread:std::marker::PhantomData}
}
#[cfg(any(test, feature = "test-seams"))]
impl Drop for PrivateShared {fn drop(&mut self){TEST_SHARED.with(|slot|*slot.borrow_mut()=self.previous.take());}}
pub fn configure_shared(limit:u64) {shared().set_limit(limit);}

#[cfg(test)]
mod tests {
    /// Replay of acknowledged operations is charged past the limit instead
    /// of refused, also on the workers its [`Context`] is carried to, and
    /// the bytes are released as usual.
    #[test]
    fn forced_reservations_are_charged_past_the_limit_and_carried_by_context() {
        let budget = Budget::new(8);
        let held = budget.reserve(8, "held").unwrap();
        assert!(budget.reserve(4, "refused").is_err());
        let (mut forced_memory, context) = forced(|| (budget.reserve(4, "replay").unwrap(), Context::capture()));
        assert!(!is_forced(), "the scope ends with the closure");
        assert_eq!(budget.used(), 12);
        forced(|| forced_memory.grow(4, "replay growth")).unwrap();
        assert_eq!(budget.used(), 16);
        let worker_budget = budget.clone();
        let on_worker = std::thread::spawn(move || context.run(|| (is_forced(), worker_budget.reserve(4, "worker").map(|r| r.bytes()).is_ok()))).join().unwrap();
        assert_eq!(on_worker, (true, true));
        assert!(budget.reserve(1, "outside the scope").is_err());
        drop(forced_memory);
        drop(held);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn zero_byte_owner_remains_valid_after_limit_reduction() {
        let budget = Budget::new(8);
        let retained = budget.reserve(8, "retained reader").unwrap();
        budget.set_limit(0);
        let empty = budget.reserve(0, "empty initial reader view").unwrap();
        assert_eq!(budget.used(), 8);
        assert!(budget.reserve(1, "real growth").is_err());
        drop(empty);
        assert_eq!(budget.used(), 8);
        drop(retained);
        assert_eq!(budget.used(), 0);
    }

    use super::*;
    #[test]
    fn category_survives_split_and_related_growth() {
        let budget=Budget::new(100);let owner=Owner::new();
        let mut first={let _owner=owner.enter();budget.reserve_categorized(40,"request",Category::EngineRequest).unwrap()};
        let mut second=first.split(15).unwrap();
        assert_eq!(owner.category_bytes(Category::EngineRequest).current,40);
        second.grow(10,"related").unwrap();drop(first);
        assert_eq!(owner.category_bytes(Category::EngineRequest).current,25);
        assert_eq!(owner.category_bytes(Category::Core).current,0);
        drop(second);assert_eq!(owner.bytes().current,0);assert_eq!(budget.used(),0);
    }
    #[test]
    fn explicit_worker_context_retains_plan_and_restores_unrelated_scope() {
        let budget=Budget::new(100);let owner=Owner::new();let other=Owner::new();
        let lease=Lease::reserve(budget.clone(),80,"plan").unwrap();
        let context={let _owner=owner.enter();let _lease=lease.enter();Context::capture()};
        let retained=std::thread::scope(|s|s.spawn(||{let _other=other.enter();
            let allocation=context.run(||budget.reserve(50,"worker allocation").unwrap());
            assert_eq!(owner.bytes().current,50);assert_eq!(other.bytes().current,0);
            let outside=budget.reserve(10,"restored worker scope").unwrap();assert_eq!(other.bytes().current,10);
            drop(outside);allocation
        }).join().unwrap());
        assert_eq!(budget.used(),80);lease.expire();assert_eq!(budget.used(),50);
        drop(retained);assert_eq!(budget.used(),0);
    }
    #[test]
    fn completed_recovery_resident_growth_uses_normal_budget_after_lease_drop() {
        let budget=Budget::new(100);let owner=Owner::new();
        let lease=Lease::reserve(budget.clone(),80,"plan").unwrap();
        let mut resident={let _lease=lease.enter();let _owner=owner.enter();budget.reserve(50,"resident").unwrap()};
        lease.complete().unwrap();assert_eq!(budget.used(),50);
        drop(lease);
        resident.grow(40,"ordinary post-recovery delete snapshot").unwrap();
        assert_eq!(budget.used(),90);assert_eq!(owner.bytes().current,90);
        assert!(resident.grow(11,"ordinary cap still enforced").is_err());
        let child=resident.reserve_related(10,"ordinary related generation").unwrap();
        drop(resident);assert_eq!(budget.used(),10);assert_eq!(owner.bytes().current,10);
        drop(child);assert_eq!(budget.used(),0);assert_eq!(owner.bytes().current,0);
        let canceled=Lease::reserve(budget.clone(),10,"canceled").unwrap();
        canceled.expire();assert!(canceled.complete().is_err());
    }
    #[test]
    fn recovery_lease_reuses_credit_and_expiration_preserves_live_allocations() {
        let budget=Budget::new(100);let owner=Owner::new();
        let lease=Lease::reserve(budget.clone(),80,"plan").unwrap();
        assert_eq!(budget.used(),80);assert_eq!(owner.bytes().current,0);
        let mut allocation={let _lease=lease.enter();let _owner=owner.enter();
            let allocation=budget.reserve(50,"decode").unwrap();
            assert!(budget.reserve(31,"overflow").is_err());allocation};
        assert_eq!(owner.bytes().current,50);
        allocation.shrink(20);assert_eq!(lease.available(),50);
        allocation.grow(40,"reuse credits across scope").unwrap();
        assert_eq!(budget.used(),80);assert_eq!(owner.bytes().current,70);
        lease.expire();assert_eq!(budget.used(),70);
        assert!(allocation.grow(1,"expired").is_err());
        {let _lease=lease.enter();assert!(budget.reserve(1,"no global fallback").is_err());}
        drop(lease);assert_eq!(budget.used(),70);
        drop(allocation);assert_eq!(budget.used(),0);assert_eq!(owner.bytes().current,0);
    }
    #[test]
    fn recovery_lease_drop_and_related_growth_keep_independent_owners() {
        let budget=Budget::new(100);let a=Owner::new();let b=Owner::new();
        let lease=Lease::reserve(budget.clone(),80,"plan").unwrap();
        let allocation={let _lease=lease.enter();let _a=a.enter();budget.reserve(30,"first").unwrap()};
        let child={let _b=b.enter();allocation.reserve_related(20,"same origin").unwrap()};
        assert_eq!(a.bytes().current,50);assert_eq!(b.bytes().current,0);
        drop(allocation);assert_eq!(lease.available(),60);
        drop(lease);assert_eq!(budget.used(),20);
        assert!(child.reserve_related(1,"gone plan").is_err());
        drop(child);assert_eq!(budget.used(),0);
    }
    #[test]
    fn shard_ownership_follows_guards_across_scopes_threads_and_related_growth() {
        let budget=Budget::new(1000);let a=Owner::new();let b=Owner::new();
        let mut retained={let _a=a.enter();let first=budget.reserve(100,"a").unwrap();
            {let _b=b.enter();let second=budget.reserve(200,"b").unwrap();assert_eq!(b.bytes().current,200);drop(second);}
            assert_eq!(a.bytes().current,100);first};
        std::thread::scope(|s|{s.spawn(||{retained.grow(50,"a growth").unwrap();}).join().unwrap();});
        let child={let _b=b.enter();retained.reserve_related(25,"a related under b scope").unwrap()};
        assert_eq!(a.bytes(),OwnedBytes{current:175,peak:175});assert_eq!(b.bytes().current,0);
        assert!(retained.grow(1000,"deny").is_err());assert_eq!(a.bytes().current,175);
        retained.shrink(25);drop(retained);assert_eq!(a.bytes().current,25);
        drop(child);assert_eq!(a.bytes().current,0);assert_eq!(budget.used(),0);
    }

    #[test]
    fn background_reservations_keep_foreground_headroom_and_wait_for_room() {
        let budget=Budget::new(100);let owner=Owner::new();
        assert_eq!(budget.background_limit(),75);
        // Larger than the background share: refused without waiting.
        let started=std::time::Instant::now();
        assert!(budget.reserve_background(76,"too large",std::time::Duration::from_secs(10)).is_err());
        assert!(started.elapsed()<std::time::Duration::from_secs(1));
        let held={let _owner=owner.enter();budget.reserve_background(70,"merge",std::time::Duration::ZERO).unwrap()};
        assert_eq!(owner.bytes().current,70);
        // The foreground quarter stays available to ordinary admissions.
        let foreground=budget.reserve(25,"fetch decode").unwrap();
        assert!(budget.reserve_background(10,"no room",std::time::Duration::ZERO).is_err());
        drop(foreground);
        // A waiting reservation succeeds once another owner releases.
        std::thread::scope(|s|{
            let waiter=s.spawn(||budget.reserve_background(10,"waits",std::time::Duration::from_secs(10)).map(|r|r.bytes()));
            std::thread::sleep(std::time::Duration::from_millis(20));
            drop(held);
            assert_eq!(waiter.join().unwrap().unwrap(),10);
        });
        assert_eq!(budget.used(),0);assert_eq!(owner.bytes().current,0);
    }

    /// Resident index data past the background share (c22 cohere_vector:
    /// retained 106 GB against a 100 GB share refused every merge's 4 KB
    /// term dictionary, and merges stopped) never blocks background work: a
    /// merge's reservations, and their growth, are admitted against what
    /// background work holds. They are still charged to `used`, so foreground
    /// admissions see them, and the share still bounds background work
    /// itself.
    #[test]
    fn resident_data_past_the_background_share_never_blocks_a_merge() {
        let budget=Budget::new(100);
        let resident=budget.reserve(98,"vector rows").unwrap();
        let mut merge=budget.reserve_background(4,"merge term dictionary",std::time::Duration::ZERO).unwrap();
        merge.grow(30,"merge growth").unwrap();
        assert_eq!((budget.used(),budget.background_used()),(132,34));
        let second=budget.reserve_background(41,"second merge",std::time::Duration::ZERO).unwrap();
        assert!(budget.reserve_background(1,"past the share",std::time::Duration::ZERO).is_err(), "the share bounds background work");
        assert!(merge.grow(1,"past the share").is_err());
        assert!(budget.reserve(1,"foreground").is_err(), "foreground still sees every byte");
        let mut moved=merge.split(10).unwrap();
        moved.shrink(4);
        drop(merge);drop(second);
        assert_eq!((budget.used(),budget.background_used()),(104,6));
        drop(moved);drop(resident);
        assert_eq!((budget.used(),budget.background_used()),(0,0));
    }

    /// Replay work reaching a background admission (a seal's rewrite, a
    /// merge it triggers) is charged past the limit at once: no headroom cap,
    /// no wait, no refusal.
    #[test]
    fn forced_background_reservations_neither_wait_nor_fail() {
        let budget=Budget::new(100);
        let held=budget.reserve(100,"held").unwrap();
        let started=std::time::Instant::now();
        let replay=forced(||budget.reserve_background(80,"replay seal",std::time::Duration::from_secs(30))).unwrap();
        assert!(started.elapsed()<std::time::Duration::from_secs(1));
        assert_eq!(budget.used(),180);
        // Outside replay a background reservation is admitted against the
        // background share alone: retained data past the limit does not
        // refuse it.
        let merge=budget.reserve_background(1,"outside replay",std::time::Duration::ZERO).unwrap();
        assert_eq!((budget.used(),budget.background_used()),(181,1));
        drop(merge);drop(replay);drop(held);assert_eq!((budget.used(),budget.background_used()),(0,0));
    }

    /// Ordinary growth charges the budget, owner and category exactly as
    /// the old transfer-from-a-temporary-guard did; empty releases and a
    /// growth below the peak change nothing else.
    #[test]
    fn ordinary_growth_and_empty_release_keep_counters_exact() {
        let budget=Budget::new(100);let owner=Owner::new();
        let mut r={let _o=owner.enter();budget.reserve_categorized(10,"base",Category::EngineRequest).unwrap()};
        r.grow(30,"grow").unwrap();
        assert_eq!(budget.used(),40);assert_eq!(r.bytes(),40);
        assert_eq!(owner.bytes(),OwnedBytes{current:40,peak:40});
        assert_eq!(owner.category_bytes(Category::EngineRequest),OwnedBytes{current:40,peak:40});
        assert_eq!(owner.category_bytes(Category::Core),OwnedBytes{current:0,peak:0});
        assert!(r.grow(61,"refused").is_err());
        assert_eq!(budget.used(),40);assert_eq!(r.bytes(),40);assert_eq!(owner.bytes().current,40);
        r.shrink(0);assert_eq!(budget.used(),40);
        r.shrink(25);assert_eq!(budget.used(),15);assert_eq!(owner.bytes(),OwnedBytes{current:15,peak:40});
        r.grow(5,"below the peak").unwrap();
        assert_eq!(owner.bytes(),OwnedBytes{current:20,peak:40});
        assert_eq!(owner.category_bytes(Category::EngineRequest),OwnedBytes{current:20,peak:40});
        r.grow(0,"empty").unwrap();assert_eq!(budget.used(),20);
        drop(budget.reserve(0,"empty guard").unwrap());assert_eq!(budget.used(),20);
        drop(r);assert_eq!(budget.used(),0);assert_eq!(owner.bytes().current,0);
        assert_eq!(owner.category_bytes(Category::EngineRequest).current,0);
    }

    #[test]
    fn retained_owner_and_failed_growth_keep_aggregate_exact() {
        let budget=Budget::new(100);
        let mut owner=budget.reserve(60,"cache").unwrap();
        assert!(owner.grow(41,"workspace").is_err());assert_eq!(owner.bytes(),60);
        owner.grow(40,"workspace").unwrap();assert_eq!(budget.used(),100);
        owner.shrink(40);assert_eq!(budget.used(),60);
        budget.set_limit(40);assert!(budget.reserve(1,"new").is_err());
        drop(owner);assert_eq!(budget.used(),0);
        assert!(budget.reserve(u64::MAX,"overflow").is_err());
    }
    #[test]
    fn concurrent_cache_and_workspace_admission_is_bounded() {
        let budget=Budget::new(64);let barrier=std::sync::Barrier::new(16);
        std::thread::scope(|scope|for _ in 0..16 {let budget=budget.clone();let barrier=&barrier;scope.spawn(move||{
            let guard=budget.reserve(16,"parallel");barrier.wait();assert!(budget.used()<=64);barrier.wait();drop(guard);
        });});assert_eq!(budget.used(),0);
    }
}

impl<T> std::ops::DerefMut for Allocation<T> { fn deref_mut(&mut self)->&mut T { &mut self.value } }
