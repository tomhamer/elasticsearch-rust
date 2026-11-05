//! Opt-in parser allocation ownership. This is not an emitter allocator API.
//! Parser methods enter an intrusive allocation session; failed scans may leave
//! local scratch behind, which the session reclaims after parser/event cleanup.
use core::{alloc::Layout, marker::PhantomData, mem, ptr};
use std::{alloc, cell::Cell, rc::Rc};
use rustic_core::{admission::{self, Reservation, Context}, error::{Result, RusticError}};
use crate::{libc, success::{Success, FAIL}, yaml_parser_t, yaml_event_t, YAML_MEMORY_ERROR};

#[repr(C,align(16))]
struct Header { size:usize, owner:*mut State, previous:*mut Header, next:*mut Header, memory:Option<Reservation> }
struct State { head:*mut Header, failed:bool }
std::thread_local! {static CURRENT:Cell<*mut State>=const{Cell::new(ptr::null_mut())};}

/// Owns parser allocations; neither it nor raw parser/event pointers may cross
/// threads. Delete every parser/event before dropping this session. Allocation
/// refusal is reported as YAML_MEMORY_ERROR, never unwinds or dereferences NULL.
pub struct ParserAllocator {state:*mut State, context:Context, _memory:Reservation, _thread:PhantomData<Rc<()>>}
impl ParserAllocator {
    pub fn new()->Result<Self>{
        let memory=admission::shared().reserve(mem::size_of::<State>() as u64,"YAML parser allocation session")?;
        let state=unsafe{alloc::alloc(Layout::new::<State>()).cast::<State>()};
        if state.is_null(){return Err(RusticError::Memory("YAML session allocation refused".into()));}
        unsafe{state.write(State{head:ptr::null_mut(),failed:false});}
        Ok(Self{state,context:Context::capture(),_memory:memory,_thread:PhantomData})
    }
    #[cfg(test)]
    pub(crate) unsafe fn test_run(&mut self,parser:*mut yaml_parser_t,f:impl FnOnce()->Success)->Success{self.run(parser,f)}
    pub fn failed(&self)->bool{unsafe{(*self.state).failed}}
    unsafe fn run(&mut self,parser:*mut yaml_parser_t, f:impl FnOnce()->Success)->Success{
        let _context=self.context.enter();
        let old=CURRENT.with(|slot|slot.replace(self.state));
        struct Restore(*mut State);impl Drop for Restore{fn drop(&mut self){CURRENT.with(|slot|slot.set(self.0));}}
        let _restore=Restore(old);
        if self.failed(){(*parser).error=YAML_MEMORY_ERROR;return FAIL;}
        let result=f();
        if self.failed(){(*parser).error=YAML_MEMORY_ERROR;FAIL}else{result}
    }
    pub unsafe fn initialize(&mut self,parser:*mut yaml_parser_t)->Success{
        // Even a previously refused session leaves a fresh parser deletable.
        // The caller must have deleted any previous contents before reuse.
        ptr::write_bytes(parser,0,1);
        self.run(parser,||crate::yaml_parser_initialize(parser))
    }
    pub unsafe fn parse(&mut self,parser:*mut yaml_parser_t,event:*mut yaml_event_t)->Success{self.run(parser,||crate::yaml_parser_parse(parser,event))}
    /// Deletion performs no allocation and is valid after failed initialization
    /// because initialization clears the complete parser before its first alloc.
    pub unsafe fn delete(&mut self,parser:*mut yaml_parser_t){crate::yaml_parser_delete(parser);}
}
impl Drop for ParserAllocator {fn drop(&mut self){unsafe{
    while !(*self.state).head.is_null(){free((*self.state).head.add(1).cast());}
    ptr::drop_in_place(self.state);alloc::dealloc(self.state.cast(),Layout::new::<State>());
}}}
fn failed(owner:*mut State){if owner.is_null(){alloc::handle_alloc_error(Layout::new::<u8>());}unsafe{(*owner).failed=true;}}
pub(crate) unsafe fn malloc(size:u64)->*mut libc::c_void{
    let owner=CURRENT.with(Cell::get);
    let Some(size)=usize::try_from(size).ok().and_then(|n|n.checked_add(mem::size_of::<Header>())) else {failed(owner);return ptr::null_mut();};
    let Ok(layout)=Layout::from_size_align(size,mem::align_of::<Header>()) else {failed(owner);return ptr::null_mut();};
    let memory=if owner.is_null(){None}else{match admission::shared().reserve(size as u64,"YAML parser allocation"){Ok(memory)=>Some(memory),Err(_)=>{failed(owner);return ptr::null_mut();}}};
    let header=alloc::alloc(layout).cast::<Header>();
    if header.is_null(){if owner.is_null(){alloc::handle_alloc_error(layout);}failed(owner);return ptr::null_mut();}
    let next=if owner.is_null(){ptr::null_mut()}else{(*owner).head};
    header.write(Header{size,owner,previous:ptr::null_mut(),next,memory});
    if !next.is_null(){(*next).previous=header;}
    if !owner.is_null(){(*owner).head=header;}
    header.add(1).cast()
}
pub(crate) unsafe fn free(value:*mut libc::c_void){
    if value.is_null(){return;}
    let header=value.cast::<Header>().sub(1);let size=(*header).size;
    if !(*header).owner.is_null(){
        if (*header).previous.is_null(){(*(*header).owner).head=(*header).next;}else{(*(*header).previous).next=(*header).next;}
        if !(*header).next.is_null(){(*(*header).next).previous=(*header).previous;}
    }
    // Release the allocation before its reservation, including abandoned locals.
    let memory=(*header).memory.take();
    ptr::drop_in_place(header);alloc::dealloc(header.cast(),Layout::from_size_align_unchecked(size,mem::align_of::<Header>()));drop(memory);
}
pub(crate) unsafe fn realloc(value:*mut libc::c_void,size:u64)->*mut libc::c_void{
    if value.is_null(){return malloc(size);}
    let header=value.cast::<Header>().sub(1);
    let old_size=(*header).size-mem::size_of::<Header>();
    if (*header).owner.is_null() && CURRENT.with(Cell::get).is_null() {
        let next=usize::try_from(size).ok().and_then(|n|n.checked_add(mem::size_of::<Header>())).unwrap_or_else(||alloc::handle_alloc_error(Layout::new::<u8>()));
        let layout=Layout::from_size_align_unchecked((*header).size,mem::align_of::<Header>());
        let next_layout=Layout::from_size_align(next,mem::align_of::<Header>()).unwrap_or_else(|_|alloc::handle_alloc_error(Layout::new::<u8>()));
        let replaced=alloc::realloc(header.cast(),layout,next).cast::<Header>();
        if replaced.is_null(){alloc::handle_alloc_error(next_layout);}
        (*replaced).size=next;return replaced.add(1).cast();
    }
    // A replacement is independently admitted while the old buffer remains
    // live. Refusal preserves the original allocation and all parser offsets.
    let replacement=malloc(size);
    if replacement.is_null(){return ptr::null_mut();}
    ptr::copy_nonoverlapping(value.cast::<u8>(),replacement.cast::<u8>(),old_size.min(size as usize));
    free(value);replacement
}

pub(crate) trait FailureValue{fn failure()->Self;}
impl FailureValue for Success{fn failure()->Self{FAIL}}
impl FailureValue for (){fn failure(){}}
impl FailureValue for i32{fn failure()->Self{0}}
impl<T> FailureValue for *mut T{fn failure()->Self{ptr::null_mut()}}
pub(crate) fn failure<T:FailureValue>()->T{T::failure()}

#[cfg(test)]
mod tests {
    use super::*;
    use rustic_core::admission::{Lease, Owner};
    #[test]
    fn initialization_denial_is_deletable_and_releases_all_owners() {
        let owner=Owner::new();
        let _owner=owner.enter();
        let lease=Lease::reserve(admission::shared(),1024*1024,"YAML init fixture").unwrap();
        let _lease=lease.enter();
        let mut allocator=ParserAllocator::new().unwrap();
        lease.expire();
        let mut parser=mem::MaybeUninit::<yaml_parser_t>::uninit();
        unsafe {
            assert!(allocator.initialize(parser.as_mut_ptr()).fail);
            assert!(allocator.failed());
            assert_eq!((*parser.as_ptr()).error,YAML_MEMORY_ERROR);
            allocator.delete(parser.as_mut_ptr());
        }
        drop(allocator);
        drop(lease);
        assert_eq!(owner.bytes().current,0);
    }
    #[test]
    fn refused_replacement_keeps_original_bytes_and_session_reclaims_scratch() {
        let owner = Owner::new();
        let _owner = owner.enter();
        let lease = Lease::reserve(admission::shared(), 1024 * 1024, "YAML replacement fixture").unwrap();
        let _lease = lease.enter();
        let mut allocator = ParserAllocator::new().unwrap();
        let mut parser = mem::MaybeUninit::<yaml_parser_t>::zeroed();
        let mut original = ptr::null_mut();
        unsafe {
            assert!(allocator.test_run(parser.as_mut_ptr(), || {
                original = malloc(32);
                assert!(!original.is_null());
                ptr::write_bytes(original, 0x5a, 32);
                crate::success::OK
            }).ok);
            lease.expire();
            let retained = owner.bytes().current;
            assert!(allocator.test_run(parser.as_mut_ptr(), || {
                assert!(realloc(original, 64).is_null());
                assert_eq!(core::slice::from_raw_parts(original.cast::<u8>(), 32), &[0x5a; 32]);
                crate::success::OK
            }).fail);
            assert_eq!(owner.bytes().current, retained);
            // This allocation models scanner-local scratch abandoned by an
            // early refusal. It is not in any parser/event cleanup structure.
            allocator.delete(parser.as_mut_ptr());
        }
        drop(allocator);
        drop(lease);
        assert_eq!(owner.bytes().current, 0);
    }

}
