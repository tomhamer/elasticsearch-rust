//! Bounded gzip output with before-growth ownership for compressor workspace,
//! output chunks and the chunk queue. No backend-selected hidden workspace.
use axum::body::Body;
use bytes::Bytes;
use elasticsearch_engine::{error::{EsError, Result}, memory::{self, Reservation}};
use miniz_oxide::deflate::core::{compress, create_comp_flags_from_zip_params, CompressorOxide, TDEFLFlush, TDEFLStatus};

const INPUT_CHUNK: usize = 1024 * 1024;
const OUTPUT_BLOCK: usize = 128 * 1024;
fn allocation(error:impl std::fmt::Display)->EsError { memory::allocation_refused(error.to_string(),0) }
struct Chunk { bytes:Vec<u8>, memory:Reservation }
impl AsRef<[u8]> for Chunk {fn as_ref(&self)->&[u8]{&self.bytes}}
impl Chunk {
    fn new()->Result<Self>{
        let memory=memory::breakers().reserve_request((std::mem::size_of::<Self>()+4*std::mem::size_of::<usize>()) as u64,"gzip chunk owner")?;
        Ok(Self{bytes:Vec::new(),memory})
    }
    fn append(&mut self,input:&[u8])->Result<()> {
        let needed=self.bytes.len().checked_add(input.len()).ok_or_else(||allocation("gzip output size overflow"))?;
        if needed>self.bytes.capacity(){
            let old=self.bytes.capacity();
            let next=needed.max(old.checked_mul(2).ok_or_else(||allocation("gzip output capacity overflow"))?);
            self.memory.grow(next as u64,"gzip output replacement")?;
            self.bytes.try_reserve_exact(next-self.bytes.len()).map_err(allocation)?;
            self.memory.shrink(old as u64);
        }
        self.bytes.extend_from_slice(input);Ok(())
    }
    fn finish(self)->Bytes {
        Bytes::from_owner(Self{bytes:self.bytes,memory:self.memory.into_result_owner()})
    }
}
struct Chunks { values:Vec<Bytes>, memory:Reservation }
impl std::ops::Deref for Chunks {type Target=[Bytes];fn deref(&self)->&[Bytes]{&self.values}}
impl Chunks {
    fn new()->Result<Self>{Ok(Self{values:Vec::new(),memory:memory::breakers().reserve_request(0,"gzip chunk queue")?})}
    fn push(&mut self,chunk:Chunk)->Result<()> {
        if self.values.len()==self.values.capacity(){
            let old=self.values.capacity();let next=old.max(2).checked_mul(2).ok_or_else(||allocation("gzip chunk queue overflow"))?;
            let bytes=next.checked_mul(std::mem::size_of::<Bytes>()).ok_or_else(||allocation("gzip chunk queue overflow"))?;
            self.memory.grow(bytes as u64,"gzip chunk queue replacement")?;
            self.values.try_reserve_exact(next-self.values.len()).map_err(allocation)?;
            self.memory.shrink((old*std::mem::size_of::<Bytes>()) as u64);
        }
        self.values.push(chunk.finish());Ok(())
    }
}
fn gzip_chunks(input:&[u8])->Result<Chunks> {
    // The vendored backend exposes all concrete private heap arrays. Reserve
    // before constructing the boxed compressor, and retain through its drop.
    let _workspace=memory::breakers().reserve_request(CompressorOxide::boxed_heap_bytes() as u64,"gzip compressor workspace")?;
    let mut encoder=Box::new(CompressorOxide::new(create_comp_flags_from_zip_params(1,0,0)));
    let mut scratch=[0u8;OUTPUT_BLOCK]; // fixed stack, not hidden heap growth
    let mut chunks=Chunks::new()?;
    let mut chunk=Chunk::new()?;
    chunk.append(&[0x1f,0x8b,8,0,0,0,0,0,4,255])?;
    for part in input.chunks(INPUT_CHUNK) {
        let mut consumed=0;
        while consumed<part.len(){
            let (status,read,written)=compress(&mut encoder,&part[consumed..],&mut scratch,TDEFLFlush::None);
            if status==TDEFLStatus::BadParam || status==TDEFLStatus::PutBufFailed || (read==0 && written==0) {return Err(EsError::internal("gzip compressor made no progress"));}
            consumed+=read;chunk.append(&scratch[..written])?;
        }
        if input.len()>INPUT_CHUNK {
            loop {
                let (status,_,written)=compress(&mut encoder,&[],&mut scratch,TDEFLFlush::Sync);
                if status==TDEFLStatus::BadParam || status==TDEFLStatus::PutBufFailed {return Err(EsError::internal("gzip compressor flush failed"));}
                chunk.append(&scratch[..written])?;
                if written<scratch.len(){break;}
            }
            chunks.push(chunk)?;chunk=Chunk::new()?;
        }
    }
    loop {
        let (status,_,written)=compress(&mut encoder,&[],&mut scratch,TDEFLFlush::Finish);
        chunk.append(&scratch[..written])?;
        if status==TDEFLStatus::Done{break;}
        if status==TDEFLStatus::BadParam || status==TDEFLStatus::PutBufFailed || written==0 {return Err(EsError::internal("gzip compressor finish failed"));}
    }
    chunk.append(&elasticsearch_engine::rustic_core::codec::crc32(input).to_le_bytes())?;
    chunk.append(&(input.len() as u32).to_le_bytes())?;
    chunks.push(chunk)?;
    Ok(chunks)
}

pub fn gzip_body(input:&[u8])->Result<Body> {
    let mut chunks=gzip_chunks(input)?;
    if chunks.len()==1{return Ok(Body::from(chunks.values.pop().unwrap()));}
    struct Queue { values:std::vec::IntoIter<Bytes>, _memory:Reservation }
    let queue=Queue{values:chunks.values.into_iter(),_memory:chunks.memory.into_result_owner()};
    // The queue owner survives until body drain/cancellation; each extracted
    // Bytes independently retains its encoded payload reservation.
    Ok(Body::from_stream(futures::stream::unfold(queue,|mut queue|async move {
        queue.values.next().map(|bytes|(Ok::<_,std::io::Error>(bytes),queue))
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn bounded_chunks_preserve_one_complete_gzip_stream() {
        for size in [0, 64, INPUT_CHUNK, 3 * INPUT_CHUNK + 19] {
            let input: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            let chunks = gzip_chunks(&input).unwrap();
            assert_eq!(chunks.len() == 1, size <= INPUT_CHUNK);
            let encoded: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
            let mut decoded = Vec::new();
            flate2::read::GzDecoder::new(encoded.as_slice()).read_to_end(&mut decoded).unwrap();
            assert_eq!(decoded, input);
        }
    }
}
