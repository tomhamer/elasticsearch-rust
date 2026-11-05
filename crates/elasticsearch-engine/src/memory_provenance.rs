//! Actual engine reservation owners across shard/worker/request handoff.
//! Logical inflight and accounting estimates are excluded: their underlying
//! buffers already have Request/Core owners and must not be counted twice.
use super::{Kind, EsError};
use rustic_core::admission::{self, Category, Context, Reservation};
struct Node { bytes:u64, next:Option<Box<Node>>, memory:Reservation }
impl Drop for Node {
    fn drop(&mut self) {
        // Request reduction can retain many independent shard components.
        // Tear the list down iteratively, not recursively on the native stack.
        let mut next=self.next.take();
        while let Some(mut node)=next {next=node.next.take();}
    }
}
pub(super) struct Ledger { origin:Context, category:Option<Category>, head:Option<Box<Node>>, actual_native:bool }
impl Ledger {
    fn category(kind:Kind)->Option<Category>{match kind {
        Kind::Request=>Some(Category::EngineRequest),Kind::Fielddata=>Some(Category::Fielddata),
        Kind::EqlSequence=>Some(Category::Eql),Kind::ModelInference=>Some(Category::Inference),
        Kind::InflightRequests|Kind::Accounting=>None,
    }}
    pub(super) fn new(kind:Kind,bytes:u64,label:&str,checked:bool)->Result<Self,EsError>{
        let mut result=Self{origin:Context::capture(),category:Self::category(kind),head:None,actual_native:false};
        if checked {result.grow(bytes,label)?;} else {result.unchecked_growth(bytes);}
        Ok(result)
    }
    pub(super) fn native(bytes:u64,label:&str)->Result<Self,EsError>{
        // Actual protocol backing, unlike logical Accounting gauges, must also
        // enter the shared aggregate budget even without a shard attribution.
        let mut result=Self{origin:Context::capture(),category:Some(Category::EngineRequest),head:None,actual_native:true};
        result.grow(bytes,label)?;Ok(result)
    }
    pub(super) fn unchecked(kind:Kind,bytes:u64)->Self {
        let mut result=Self{origin:Context::capture(),category:Self::category(kind),head:None,actual_native:false};
        result.unchecked_growth(bytes);result
    }
    pub(super) fn grow(&mut self,bytes:u64,label:&str)->Result<(),EsError>{
        let Some(category)=self.category.filter(|_|self.actual_native||self.origin.is_attributed()) else{return Ok(())};
        if bytes==0{return Ok(())}
        if let Some(head)=self.head.as_mut(){
            // A public reservation grows only before request-retention. Its
            // existing owner/lease remains the origin even on another thread.
            head.memory.grow(bytes,label)?;
            head.bytes=head.bytes.checked_add(bytes).expect("admitted component size");
        }else{
            let total=bytes.checked_add(std::mem::size_of::<Node>() as u64).ok_or_else(||EsError::new(429,"circuit_breaking_exception","reservation owner size overflow"))?;
            let memory=self.origin.run(||admission::shared().reserve_categorized(total,label,category))?;
            self.head=Some(Box::new(Node{bytes,next:None,memory}));
        }
        Ok(())
    }
    pub(super) fn unchecked_growth(&self,bytes:u64){
        if bytes!=0 && self.category.is_some(){self.origin.record_unplanned();}
    }
    pub(super) fn shrink(&mut self,mut bytes:u64){
        let mut current=self.head.as_deref_mut();
        while let Some(node)=current {
            let released=bytes.min(node.bytes);node.memory.shrink(released);node.bytes-=released;bytes-=released;
            if bytes==0{break}current=node.next.as_deref_mut();
        }
    }
    pub(super) fn append(&mut self,other:&mut Self){
        // Nodes were preallocated/admitted with each original reservation.
        // No allocation or owner reassignment at the result handoff.
        let mut incoming=other.head.take();
        let mut tail=&mut incoming;
        while let Some(node)=tail {tail=&mut node.next;}
        *tail=self.head.take();
        self.head=incoming;
    }
}
