//! Kernel floor measurement (ignored): the knn_ops graph exported into dense
//! slot-indexed arrays and searched by a minimal kernel, to measure what a
//! step can cost with one load per vector, fixed-stride level-0 links and a
//! generation-stamped visited array, under several prefetch strategies.
use super::*;
use std::time::{Duration, Instant};

struct Aligned { ptr: *mut u8, layout: std::alloc::Layout }
impl Aligned {
    fn new(bytes: usize, huge: bool) -> Self {
        let align = if huge { 2 << 20 } else { 64 };
        let size = bytes.div_ceil(align) * align;
        let layout = std::alloc::Layout::from_size_align(size.max(align), align).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        #[cfg(target_os = "linux")]
        if huge { unsafe { libc::madvise(ptr.cast(), layout.size(), libc::MADV_HUGEPAGE); } }
        // Touch after madvise so faults take huge pages.
        unsafe { std::ptr::write_bytes(ptr, 0, layout.size()); }
        Self { ptr, layout }
    }
}
impl Drop for Aligned { fn drop(&mut self) { unsafe { std::alloc::dealloc(self.ptr, self.layout) } } }

struct Dense {
    dim: usize,
    vstride: usize,
    vecs: Aligned,
    lstride: usize,
    links0: Aligned,
    upper: Vec<std::collections::HashMap<u32, Vec<u32>>>,
    entry: u32,
    top: usize,
    len: u32,
    orig: Vec<u32>,
    fake: bool,
    cstride: usize,
    codes: Aligned,
    qscale: f32,
    q4scale: f32,
    codes4: Aligned,
    chain: Option<(Vec<u32>, Aligned, Aligned)>,
    vis: Option<(Vec<u32>, Vec<u64>)>,
}
impl Dense {
    #[inline(always)]
    fn row(&self, id: u32) -> &[f32] {
        let id = if self.fake { id & 1023 } else { id };
        unsafe { std::slice::from_raw_parts((self.vecs.ptr as *const f32).add(id as usize * self.vstride), self.dim) }
    }
    #[inline(always)]
    fn links(&self, id: u32) -> &[u32] {
        unsafe {
            let p = (self.links0.ptr as *const u32).add(id as usize * self.lstride);
            std::slice::from_raw_parts(p.add(1), *p as usize)
        }
    }
    fn export(view: &MutableHnswSnapshot, rows: &Rows, huge: bool) -> Self {
        let dim = view.dim as usize;
        let vstride = dim.div_ceil(16) * 16;
        let len = view.len;
        let vecs = Aligned::new(len as usize * vstride * 4, huge);
        let lstride = (1 + 2 * view.m).div_ceil(16) * 16;
        let links0 = Aligned::new(len as usize * lstride * 4, huge);
        let mut upper: Vec<std::collections::HashMap<u32, Vec<u32>>> = Vec::new();
        // `KERNEL_BFS=1`: renumber nodes in breadth-first order of level 0
        // from the entry (neighbours land near each other in memory).
        let mut newid: Vec<u32> = (0..len).collect();
        if std::env::var("KERNEL_BFS").is_ok() {
            let mut seen = vec![false; len as usize];
            let mut order = vec![0u32];
            let mut queue = std::collections::VecDeque::from([view.entry as u32]);
            seen[view.entry as u32 as usize] = true;
            while let Some(id) = queue.pop_front() {
                order.push(id);
                if let Some(node) = view.current_node(id) {
                    for &n in node.links(0).unwrap_or(&[]) { if !seen[n as usize] { seen[n as usize] = true; queue.push_back(n); } }
                }
            }
            for id in 1..len { if !seen[id as usize] { order.push(id); } }
            for (new, &old) in order.iter().enumerate() { newid[old as usize] = new as u32; }
        }
        let mut orig = vec![0u32; len as usize];
        for old in 0..len { orig[newid[old as usize] as usize] = old; }
        for old in 1..len {
            let id = newid[old as usize];
            let Some(node) = view.current_node(old) else { continue };
            let row = rows.get(node.key()).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), (vecs.ptr as *mut f32).add(id as usize * vstride), dim); }
            let l0: Vec<u32> = node.links(0).unwrap_or(&[]).iter().map(|&n| newid[n as usize]).collect();
            unsafe {
                let p = (links0.ptr as *mut u32).add(id as usize * lstride);
                *p = l0.len() as u32;
                std::ptr::copy_nonoverlapping(l0.as_ptr(), p.add(1), l0.len());
            }
            for level in 1..=node.level() {
                while upper.len() < level { upper.push(Default::default()); }
                upper[level - 1].insert(id, node.links(level).unwrap_or(&[]).iter().map(|&n| newid[n as usize]).collect());
            }
        }
        Self { dim, vstride, vecs, lstride, links0, upper, chain: None, vis: None, qscale: 0.0, q4scale: 0.0, codes4: Aligned::new(64, false), cstride: 0, codes: Aligned::new(64, false), orig, entry: newid[view.entry as u32 as usize], top: (view.entry >> 32) as usize, len, fake: std::env::var("KERNEL_FAKE").is_ok() }
    }
}

impl Dense {
    fn quantize(&mut self, cstride: usize, huge: bool) {
        let mut max = 0f32;
        for id in 1..self.len { for &x in self.row(id) { max = max.max(x.abs()); } }
        let scale = 127.0 / max;
        self.cstride = cstride;
        self.codes = Aligned::new(self.len as usize * cstride, huge);
        for id in 1..self.len {
            let row: Vec<i8> = self.row(id).iter().map(|&x| (x * scale).round() as i8).collect();
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), (self.codes.ptr as *mut i8).add(id as usize * cstride), self.dim); }
        }
        self.qscale = scale;
    }
    /// Emulate the production pointer chain to a node's level-0 links:
    /// `levels[id]` -> a levels entry at a random place -> a links array at
    /// a random place (64-byte aligned slots of a shuffled pool).
    fn build_chain(&mut self) {
        let n = self.len as usize;
        let mut perm: Vec<u32> = (0..n as u32).collect();
        let mut rng = XorShift64Star::new(7);
        for i in (1..n).rev() { perm.swap(i, (rng.next_f64() * (i + 1) as f64) as usize % (i + 1)); }
        let mut perm2 = perm.clone();
        for i in (1..n).rev() { perm2.swap(i, (rng.next_f64() * (i + 1) as f64) as usize % (i + 1)); }
        let levels_pool = Aligned::new(n * 16, false); // 16-byte level arrays
        let links_pool = Aligned::new(n * self.lstride * 4, false);
        let mut top = vec![0u32; n];
        for id in 1..n {
            let a = perm[id] as usize; let b = perm2[id] as usize;
            top[id] = a as u32;
            unsafe {
                *(levels_pool.ptr as *mut u32).add(a * 4) = b as u32;
                let src = (self.links0.ptr as *const u32).add(id * self.lstride);
                std::ptr::copy_nonoverlapping(src, (links_pool.ptr as *mut u32).add(b * self.lstride), self.lstride);
            }
        }
        self.chain = Some((top, levels_pool, links_pool));
    }
    fn build_vis(&mut self) {
        let n = self.len as usize;
        self.vis = Some(((0..n as u32).map(|i| i | 0x8000_0000).collect(), vec![1u64; n]));
    }
    #[inline(always)]
    fn links_any(&self, id: u32) -> &[u32] {
        match &self.chain {
            None => self.links(id),
            Some((top, levels, pool)) => unsafe {
                let a = std::ptr::read_volatile(top.as_ptr().add(id as usize)) as usize;
                let b = *(levels.ptr as *const u32).add(a * 4) as usize;
                let p = (pool.ptr as *const u32).add(b * self.lstride);
                std::slice::from_raw_parts(p.add(1), *p as usize)
            },
        }
    }
    #[inline(always)]
    fn code(&self, id: u32) -> *const i8 { unsafe { (self.codes.ptr as *const i8).add(id as usize * self.cstride) } }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn sdot4(q: *const i8, rows: [*const i8; 4], n: usize) -> [i32; 4] {
    use core::arch::aarch64::*;
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_s8(q.add(i));
            for (r, a) in rows.iter().zip(acc.iter_mut()) {
                let y = vld1q_s8(r.add(i));
                core::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) *a, x = in(vreg) x, y = in(vreg) y,
                    options(pure, nomem, nostack, preserves_flags));
            }
            i += 16;
        }
        [vaddvq_s32(acc[0]), vaddvq_s32(acc[1]), vaddvq_s32(acc[2]), vaddvq_s32(acc[3])]
    }
}

impl Dense {
    /// 4-bit codes, 48 bytes a row: byte j holds dims j (low nibble) and
    /// j + dim/2 (high), each `round(x * s) + 8` clipped to 0..15.
    fn quantize4(&mut self, huge: bool) {
        let mut sq = 0f64; let mut n = 0f64;
        for id in 1..self.len { for &x in self.row(id) { sq += f64::from(x * x); n += 1.0; } }
        let std = (sq / n).sqrt() as f32;
        let clip: f32 = std::env::var("KERNEL_CLIP").ok().and_then(|v| v.parse().ok()).unwrap_or(2.5);
        let s = 7.0 / (clip * std);
        self.q4scale = s;
        let half = self.dim / 2;
        self.codes4 = Aligned::new(self.len as usize * 64, huge);
        for id in 1..self.len {
            let row = self.row(id).to_vec();
            let c = |x: f32| ((x * s).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
            for j in 0..half {
                unsafe { *self.codes4.ptr.add(id as usize * 64 + j) = c(row[j]) | (c(row[j + half]) << 4); }
            }
        }
    }
    #[inline(always)]
    fn code4(&self, id: u32) -> *const u8 { unsafe { self.codes4.ptr.add(id as usize * 64) } }
}
/// Four 4-bit rows (48 bytes each, 96 dims) against an i8 query split in
/// halves (`qlo`: dims 0..48, `qhi`: 48..96); codes biased by 8.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot4_i4(qlo: *const i8, qhi: *const i8, rows: [*const u8; 4]) -> [i32; 4] {
    use core::arch::aarch64::*;
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        let mask = vdupq_n_u8(15);
        let bias = vdupq_n_s8(8);
        for i in [0usize, 16, 32] {
            let a = vld1q_s8(qlo.add(i));
            let b = vld1q_s8(qhi.add(i));
            for (r, acc) in rows.iter().zip(acc.iter_mut()) {
                let v = vld1q_u8(r.add(i));
                let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), bias);
                let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), bias);
                core::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b", acc = inout(vreg) *acc, x = in(vreg) a, y = in(vreg) lo, options(pure, nomem, nostack, preserves_flags));
                core::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b", acc = inout(vreg) *acc, x = in(vreg) b, y = in(vreg) hi, options(pure, nomem, nostack, preserves_flags));
            }
        }
        [vaddvq_s32(acc[0]), vaddvq_s32(acc[1]), vaddvq_s32(acc[2]), vaddvq_s32(acc[3])]
    }
}

/// 4-bit traversal, the final `ef` rescored with the int8 codes.
fn kernel_i4(g: &Dense, q: &[f32], ef: usize, v: &mut Stamp, steps: &mut u64) -> Vec<(f32, u32)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    v.next();
    let qc: Vec<i8> = q.iter().map(|&x| (x * g.qscale).round().clamp(-127.0, 127.0) as i8).collect();
    let half = g.dim / 2;
    let one = |id: u32| -> f32 { let r = unsafe { dot4_i4(qc.as_ptr(), qc.as_ptr().add(half), [g.code4(id); 4]) }; -(r[0] as f32) };
    let one8 = |id: u32| -> f32 { let r = unsafe { sdot4(qc.as_ptr(), [g.code(id); 4], g.dim) }; -(r[0] as f32) };
    let mut cur = g.entry;
    let mut cd = one(cur);
    *steps += 1;
    for level in (1..=g.top).rev() {
        loop {
            let mut improved = false;
            if let Some(links) = g.upper[level - 1].get(&cur) {
                for &n in links { let d = one(n); *steps += 1; if d < cd { cd = d; cur = n; improved = true; } }
            }
            if !improved { break; }
        }
    }
    let key = |d: f32, id: u32| (u64::from(ordered(d)) << 32) | u64::from(id);
    let mut cand: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(4 * ef);
    let mut top: BinaryHeap<u64> = BinaryHeap::with_capacity(ef + 1);
    v.test_set(cur);
    cand.push(Reverse(key(cd, cur)));
    top.push(key(cd, cur));
    let mut limit = u64::MAX;
    let mut buf = [0u32; 132];
    while let Some(Reverse(c)) = cand.pop() {
        if c > limit { break; }
        let id = c as u32;
        if let Some(Reverse(next)) = cand.peek() {
            let p = unsafe { (g.links0.ptr as *const u32).add((*next as u32) as usize * g.lstride) };
            prefetch(p.cast(), 65 * 4);
        }
        let links = g.links(id);
        let mut n = 0;
        for &x in links { if v.test_set(x) { buf[n] = x; n += 1; prefetch(g.code4(x), 48); } }
        let mut i = 0;
        while i < n {
            let m = (n - i).min(4);
            let r = unsafe { dot4_i4(qc.as_ptr(), qc.as_ptr().add(half), [g.code4(buf[i]), g.code4(buf[i + 1.min(m - 1)]), g.code4(buf[i + 2.min(m - 1)]), g.code4(buf[i + 3.min(m - 1)])]) };
            *steps += m as u64;
            for j in 0..m {
                let k = key(-(r[j] as f32), buf[i + j]);
                if k < limit {
                    cand.push(Reverse(k));
                    top.push(k);
                    if top.len() > ef { top.pop(); limit = *top.peek().unwrap(); }
                }
            }
            i += m;
        }
    }
    let ids: Vec<u32> = top.into_iter().map(|k| k as u32).collect();
    for &id in &ids { prefetch(g.code(id).cast(), g.dim); }
    let mut out: Vec<(f32, u32)> = ids.iter().map(|&id| (one8(id), id)).collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// int8 traversal (naive symmetric scale), optional f32 rescore of the
/// final `ef` candidates.
fn kernel_i8(g: &Dense, q: &[f32], ef: usize, dist: usize, rescore: bool, v: &mut Stamp, steps: &mut u64) -> Vec<(f32, u32)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    v.next();
    let qc: Vec<i8> = q.iter().map(|&x| (x * g.qscale).round().clamp(-127.0, 127.0) as i8).collect();
    let one = |id: u32| -> f32 { let r = unsafe { sdot4(qc.as_ptr(), [g.code(id); 4], g.dim) }; -(r[0] as f32) };
    let mut cur = g.entry;
    let mut cd = one(cur);
    *steps += 1;
    for level in (1..=g.top).rev() {
        loop {
            let mut improved = false;
            if let Some(links) = g.upper[level - 1].get(&cur) {
                for &n in links { let d = one(n); *steps += 1; if d < cd { cd = d; cur = n; improved = true; } }
            }
            if !improved { break; }
        }
    }
    let key = |d: f32, id: u32| (u64::from(ordered(d)) << 32) | u64::from(id);
    let mut cand: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(4 * ef);
    let mut top: BinaryHeap<u64> = BinaryHeap::with_capacity(ef + 1);
    v.test_set(cur);
    cand.push(Reverse(key(cd, cur)));
    top.push(key(cd, cur));
    let mut limit = u64::MAX;
    let mut buf = [0u32; 132];
    while let Some(Reverse(c)) = cand.pop() {
        if c > limit { break; }
        let id = c as u32;
        if g.chain.is_none() {
            if let Some(Reverse(next)) = cand.peek() {
                let p = unsafe { (g.links0.ptr as *const u32).add((*next as u32) as usize * g.lstride) };
                prefetch(p.cast(), 65 * 4);
            }
        }
        let links = g.links_any(id);
        let mut n = 0;
        if let Some((levels, pubs)) = &g.vis {
            let mut m = 0;
            let mut tmp = [0u32; 132];
            for &x in links { if v.test_set(x) { tmp[m] = x; m += 1;
                prefetch((&levels[x as usize] as *const u32).cast(), 4); prefetch((&pubs[x as usize] as *const u64).cast(), 8);
                prefetch(g.code(x).cast(), g.dim); } }
            for &x in &tmp[..m] {
                if levels[x as usize] != 0 && pubs[x as usize] <= 1 { buf[n] = x; n += 1; }
            }
        } else {
            for &x in links { if v.test_set(x) { buf[n] = x; n += 1; } }
        }
        for &x in buf[..n].iter().take(dist) { prefetch(g.code(x).cast(), g.dim); }
        // Pad to a multiple of 4 with a repeat of the last id (scored, ignored).
        let mut i = 0;
        while i < n {
            let m = (n - i).min(4);
            for j in i + dist..(i + dist + m).min(n) { prefetch(g.code(buf[j]).cast(), g.dim); }
            let r = unsafe { sdot4(qc.as_ptr(), [g.code(buf[i]), g.code(buf[i + 1.min(m - 1)]), g.code(buf[i + 2.min(m - 1)]), g.code(buf[i + 3.min(m - 1)])], g.dim) };
            *steps += m as u64;
            for j in 0..m {
                let k = key(-(r[j] as f32), buf[i + j]);
                if k < limit {
                    cand.push(Reverse(k));
                    top.push(k);
                    if top.len() > ef { top.pop(); limit = *top.peek().unwrap(); }
                }
            }
            i += m;
        }
    }
    let ids: Vec<u32> = top.into_iter().map(|k| k as u32).collect();
    let mut out: Vec<(f32, u32)> = if rescore {
        for &id in &ids { prefetch(g.row(id).as_ptr().cast(), g.dim * 4); }
        ids.iter().map(|&id| (-dot(q, g.row(id)), id)).collect()
    } else { ids.iter().map(|&id| (one(id), id)).collect() };
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

#[inline(always)]
fn ordered(d: f32) -> u32 { let b = d.to_bits(); if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 } }
#[inline(always)]
fn unordered(o: u32) -> f32 { f32::from_bits(if o & 0x8000_0000 != 0 { o & 0x7fff_ffff } else { !o }) }

struct Stamp { marks: Vec<u16>, epoch: u16 }
impl Stamp {
    fn new(n: usize) -> Self { Self { marks: vec![0; n], epoch: 0 } }
    fn next(&mut self) { self.epoch = self.epoch.wrapping_add(1); if self.epoch == 0 { self.marks.fill(0); self.epoch = 1; } }
    #[inline(always)] fn test_set(&mut self, id: u32) -> bool {
        let m = unsafe { self.marks.get_unchecked_mut(id as usize) };
        if *m == self.epoch { false } else { *m = self.epoch; true }
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode { All, Pipe(usize), AllLook, PipeLook(usize) }

fn kernel(g: &Dense, q: &[f32], ef: usize, mode: Mode, v: &mut Stamp, steps: &mut u64) -> Vec<(f32, u32)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    v.next();
    let mut cur = g.entry;
    let mut cd = -dot(q, g.row(cur));
    *steps += 1;
    for level in (1..=g.top).rev() {
        loop {
            let mut improved = false;
            if let Some(links) = g.upper[level - 1].get(&cur) {
                for &n in links {
                    let d = -dot(q, g.row(n));
                    *steps += 1;
                    if d < cd { cd = d; cur = n; improved = true; }
                }
            }
            if !improved { break; }
        }
    }
    let key = |d: f32, id: u32| (u64::from(ordered(d)) << 32) | u64::from(id);
    let mut cand: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(4 * ef);
    let mut top: BinaryHeap<u64> = BinaryHeap::with_capacity(ef + 1);
    v.test_set(cur);
    cand.push(Reverse(key(cd, cur)));
    top.push(key(cd, cur));
    let mut limit = u64::MAX;
    let mut buf = [0u32; 128];
    let look = matches!(mode, Mode::AllLook | Mode::PipeLook(_));
    while let Some(Reverse(c)) = cand.pop() {
        if c > limit { break; }
        let id = c as u32;
        if look {
            if let Some(Reverse(next)) = cand.peek() {
                let p = unsafe { (g.links0.ptr as *const u32).add((*next as u32) as usize * g.lstride) };
                prefetch(p.cast(), (1 + g.links(*next as u32).len().min(64)) * 4);
            }
        }
        let links = g.links(id);
        let mut n = 0;
        for &x in links { if v.test_set(x) { buf[n] = x; n += 1; } }
        let ids = &buf[..n];
        let rowp = |x: u32| { let x = if g.fake { x & 1023 } else { x }; (unsafe { (g.vecs.ptr as *const f32).add(x as usize * g.vstride) }) as *const u8 };
        let dist = match mode { Mode::All | Mode::AllLook => 0, Mode::Pipe(d) | Mode::PipeLook(d) => d };
        if dist == 0 { for &x in ids { prefetch(rowp(x), g.dim * 4); } }
        else { for &x in ids.iter().take(dist) { prefetch(rowp(x), g.dim * 4); } }
        let mut i = 0;
        while i < n {
            let m = (n - i).min(4);
            if dist > 0 { for j in i + dist..(i + dist + m).min(n) { prefetch(rowp(ids[j]), g.dim * 4); } }
            let mut ds = [0f32; 4];
            if m == 4 {
                let p = dot_four(q, [g.row(ids[i]), g.row(ids[i + 1]), g.row(ids[i + 2]), g.row(ids[i + 3])]);
                for j in 0..4 { ds[j] = -p[j]; }
            } else { for j in 0..m { ds[j] = -dot(q, g.row(ids[i + j])); } }
            *steps += m as u64;
            for j in 0..m {
                let k = key(ds[j], ids[i + j]);
                if k < limit {
                    cand.push(Reverse(k));
                    top.push(k);
                    if top.len() > ef { top.pop(); limit = *top.peek().unwrap(); }
                }
            }
            i += m;
        }
    }
    let mut out: Vec<(f32, u32)> = top.into_iter().map(|k| (unordered((k >> 32) as u32), k as u32)).collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// `cargo test --profile fast -p rustic-core --lib knn_dense_kernel_floor -- --ignored --nocapture`
/// `RUSTIC_BENCH_GRAPH=<file>`: reuse (or write) the built graph.
#[test]
#[ignore]
fn knn_dense_kernel_floor() {
    use std::collections::HashSet;
    crate::admission::configure_shared(24 << 30);
    let docs: usize = std::env::var("RUSTIC_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let (vecs, queries) = knn_ops_corpus(docs);
    let truths: Vec<HashSet<u64>> = knn_ops_truths(&vecs, &queries);
    let graph = knn_ops_graph(&vecs);
    let view = graph.snapshot();
    let base = |name: &str| {
        let (mut ops, mut recall, mut time) = (0u64, 0usize, Duration::ZERO);
        for q in queries.iter().take(10) { let _ = view.search(q, 100, 1000, &vecs, |_| true); }
        for (q, want) in queries.iter().zip(&truths) {
            let t = Instant::now();
            let r = view.search(q, 100, 1000, &vecs, |_| true);
            time += t.elapsed();
            ops += r.vector_ops - r.final_score_ops;
            recall += r.hits.iter().filter(|(k, _)| want.contains(k)).count();
        }
        eprintln!("{name:12} {:>7} steps/q {:6.1} ns/step {:7.3} ms/q recall {:.4}", ops / 100, time.as_nanos() as f64 / ops as f64, time.as_secs_f64() * 10.0, recall as f64 / 10_000.0);
    };
    base("prod-vecs");
    for huge in [false, true] {
        let mut dense = Dense::export(&view, &vecs, huge);
        dense.quantize(96, huge);
        dense.quantize4(huge);
        for ef in [1000usize, 1200, 1500] {
            let mut stamp = Stamp::new(dense.len as usize);
            let mut steps = 0u64;
            for q in queries.iter().take(10) { let _ = kernel_i4(&dense, q, ef, &mut stamp, &mut steps); }
            steps = 0;
            let (mut recall, mut time) = (0usize, Duration::ZERO);
            for (q, want) in queries.iter().zip(&truths) {
                let t = Instant::now();
                let r = kernel_i4(&dense, q, ef, &mut stamp, &mut steps);
                time += t.elapsed();
                recall += r.iter().take(100).filter(|(_, id)| want.contains(&u64::from(dense.orig[*id as usize] - 1))).count();
            }
            eprintln!("{:12} {:>7} steps/q {:6.1} ns/step {:7.3} ms/q recall {:.4}", format!("i4/ef{ef}{}", if huge { "+thp" } else { "" }),
                steps / 100, time.as_nanos() as f64 / steps as f64, time.as_secs_f64() * 10.0, recall as f64 / 10_000.0);
        }
        for cstride in [96usize, 128] {
            dense.quantize(cstride, huge);
            if std::env::var("KERNEL_CHAIN").is_ok() { dense.build_chain(); }
            if std::env::var("KERNEL_VIS").is_ok() { dense.build_vis(); }
            let mut stamp = Stamp::new(dense.len as usize);
            for (dist, rescore) in [(4, false), (4, true), (8, true), (16, true)] {
                let mut steps = 0u64;
                for q in queries.iter().take(10) { let _ = kernel_i8(&dense, q, 1000, dist, rescore, &mut stamp, &mut steps); }
                steps = 0;
                let (mut recall, mut time) = (0usize, Duration::ZERO);
                for (q, want) in queries.iter().zip(&truths) {
                    let t = Instant::now();
                    let r = kernel_i8(&dense, q, 1000, dist, rescore, &mut stamp, &mut steps);
                    time += t.elapsed();
                    recall += r.iter().take(100).filter(|(_, id)| want.contains(&u64::from(dense.orig[*id as usize] - 1))).count();
                }
                eprintln!("{:12} {:>7} steps/q {:6.1} ns/step {:7.3} ms/q recall {:.4}", format!("i8/{cstride}/p{dist}{}{}", if rescore { "/rs" } else { "" }, if huge { "+thp" } else { "" }),
                    steps / 100, time.as_nanos() as f64 / steps as f64, time.as_secs_f64() * 10.0, recall as f64 / 10_000.0);
            }
        }
        let mut stamp = Stamp::new(dense.len as usize);
        let modes: Vec<Mode> = std::env::var("RUSTIC_KERNEL_MODES").ok().map(|s| s.split(',').map(|m| match m {
            "all" => Mode::All, "alllook" => Mode::AllLook,
            m if m.starts_with("look") => Mode::PipeLook(m[4..].parse().unwrap()),
            m => Mode::Pipe(m.parse().unwrap()) }).collect())
            .unwrap_or(vec![Mode::All, Mode::AllLook, Mode::Pipe(4), Mode::Pipe(8), Mode::PipeLook(4), Mode::PipeLook(8), Mode::PipeLook(12)]);
        for mode in modes {
            let mut steps = 0u64;
            for q in queries.iter().take(10) { let _ = kernel(&dense, q, 1000, mode, &mut stamp, &mut steps); }
            steps = 0;
            let (mut recall, mut time) = (0usize, Duration::ZERO);
            for (q, want) in queries.iter().zip(&truths) {
                let t = Instant::now();
                let r = kernel(&dense, q, 1000, mode, &mut stamp, &mut steps);
                time += t.elapsed();
                // ids are slot + 1 = key + 1 in this bench.
                recall += r.iter().take(100).filter(|(_, id)| want.contains(&u64::from(dense.orig[*id as usize] - 1))).count();
            }
            eprintln!("{:12} {:>7} steps/q {:6.1} ns/step {:7.3} ms/q recall {:.4}", format!("{mode:?}{}", if huge { "+thp" } else { "" }),
                steps / 100, time.as_nanos() as f64 / steps as f64, time.as_secs_f64() * 10.0, recall as f64 / 10_000.0);
        }
    }
}

pub(super) struct BenchRows { pub(super) rows: Vec<Vec<f32>> }
impl MutableVectorSource for BenchRows {
    fn get(&self, key: u64) -> Option<&[f32]> { self.rows.get(key as usize).map(Vec::as_slice) }
}
type Rows = BenchRows;
/// The knn_ops corpus of `knn_ns_per_step_by_source` (same seed and order).
pub(super) fn knn_ops_corpus(docs: usize) -> (BenchRows, Vec<Vec<f32>>) {
    let dim = 96;
    let mut rng = XorShift64Star::new(0x5EED);
    let mut unit = || {
        let mut v: Vec<f32> = (0..dim).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v { *x /= n; }
        v
    };
    let rows = BenchRows { rows: (0..docs).map(|_| unit()).collect() };
    let queries: Vec<Vec<f32>> = (0..100).map(|_| unit()).collect();
    (rows, queries)
}
pub(super) fn knn_ops_truths(vecs: &BenchRows, queries: &[Vec<f32>]) -> Vec<std::collections::HashSet<u64>> {
    use rayon::prelude::*;
    queries.par_iter().map(|q| {
        let mut all: Vec<(f32, u64)> = (0..vecs.rows.len() as u64).map(|k| (-dot(q, &vecs.rows[k as usize]), k)).collect();
        all.select_nth_unstable_by(99, |a, b| a.0.total_cmp(&b.0));
        all.into_iter().take(100).map(|(_, k)| k).collect()
    }).collect()
}
/// The knn_ops graph (m=32, efc=100, dot product), built as the bench
/// builds it, or reopened from `RUSTIC_BENCH_GRAPH` (written after a build).
pub(super) fn knn_ops_graph(vecs: &BenchRows) -> Arc<MutableHnsw> {
    use rayon::prelude::*;
    let cache = std::env::var("RUSTIC_BENCH_GRAPH").ok().map(|p| format!("{p}.{}", vecs.rows.len()));
    if let Some(path) = &cache {
        if let Ok(bytes) = std::fs::read(path) {
            let graph = Arc::new(MutableHnsw::open(&bytes).unwrap());
            eprintln!("reopened graph {path} ({} nodes)", graph.num_nodes());
            return graph;
        }
    }
    let params = HnswParams { m: 32, ef_construction: 100, ..HnswParams::default() };
    let graph = Arc::new(MutableHnsw::new(96, Similarity::DotProduct, params));
    let started = Instant::now();
    for chunk in (0..vecs.rows.len() as u64).collect::<Vec<_>>().chunks(512) {
        let reservations: Vec<_> = chunk.iter().map(|&k| graph.reserve_insert(k).unwrap()).collect();
        let prepared: Vec<_> = reservations.into_par_iter().map(|r| graph.prepare(r, vecs).unwrap()).collect();
        for p in prepared { graph.complete(p, vecs).unwrap(); }
    }
    eprintln!("built {} nodes in {:?}", graph.num_nodes(), started.elapsed());
    if let Some(path) = &cache { std::fs::write(path, graph.encode()).unwrap(); }
    graph
}
