//! Java semantics the profiling responses depend on: `String.hashCode`,
//! `Objects.hash`, and the iteration order of `java.util.HashMap` and
//! `java.util.concurrent.ConcurrentHashMap` (ES renders several maps in
//! their iteration order, and the flamegraph's `Edges` are `HashMap`
//! values).

/// `String.hashCode` (UTF-16 code units, wrapping `int` arithmetic).
pub fn string_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// One argument of `Objects.hash`.
pub enum J<'a> {
    S(&'a str),
    I(i32),
}

/// `Objects.hash(...)` (= `Arrays.hashCode`).
pub fn objects_hash(args: &[J]) -> i32 {
    args.iter().fold(1i32, |r, a| {
        let h = match a {
            J::S(s) => string_hash(s),
            J::I(i) => *i,
        };
        r.wrapping_mul(31).wrapping_add(h)
    })
}

/// `HashMap.tableSizeFor`.
fn table_size_for(c: i64) -> usize {
    if c <= 1 {
        return 1;
    }
    let mut n = 1usize;
    while (n as i64) < c && n < (1 << 30) {
        n <<= 1;
    }
    n
}

/// `java.util.HashMap<String, V>` iteration order: keys are kept in
/// insertion order; the bucket of a key in the final table determines its
/// position (entries of one bucket keep insertion order across resizes).
#[derive(Clone, Debug)]
pub struct JHashMap<V> {
    entries: Vec<(String, V)>,
    cap: usize,
    threshold: usize,
}

impl<V> Default for JHashMap<V> {
    fn default() -> Self {
        JHashMap { entries: Vec::new(), cap: 16, threshold: 12 }
    }
}

impl<V> JHashMap<V> {
    /// `new HashMap<>(initialCapacity)`.
    pub fn with_capacity(initial: usize) -> Self {
        let cap = table_size_for(initial as i64);
        JHashMap { entries: Vec::new(), cap, threshold: (cap as f64 * 0.75) as usize }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, k: &str) -> Option<&V> {
        self.entries.iter().find(|(x, _)| x == k).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, k: &str) -> Option<&mut V> {
        self.entries.iter_mut().find(|(x, _)| x == k).map(|(_, v)| v)
    }

    pub fn contains_key(&self, k: &str) -> bool {
        self.get(k).is_some()
    }

    /// `put`: replaces the value of an existing key in place.
    pub fn put(&mut self, k: String, v: V) {
        if let Some(slot) = self.get_mut(&k) {
            *slot = v;
            return;
        }
        self.entries.push((k, v));
        if self.entries.len() > self.threshold {
            self.cap *= 2;
            self.threshold = (self.cap as f64 * 0.75) as usize;
        }
    }

    /// Entries in Java iteration order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        let mask = self.cap - 1;
        let mut idx: Vec<usize> = (0..self.entries.len()).collect();
        idx.sort_by_key(|&i| spread_hm(string_hash(&self.entries[i].0)) as usize & mask);
        idx.into_iter().map(move |i| (&self.entries[i].0, &self.entries[i].1))
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }
}

fn spread_hm(h: i32) -> u32 {
    let h = h as u32;
    h ^ (h >> 16)
}

fn spread_chm(h: i32) -> u32 {
    let h = h as u32;
    (h ^ (h >> 16)) & 0x7fff_ffff
}

/// `java.util.concurrent.ConcurrentHashMap<String, _>` key order (as a
/// set of keys; values live elsewhere). Simulates table sizing, bin
/// appends, `transfer` (which partially reverses bins) and the
/// treeify-triggered presize of small tables.
pub struct JConcurrentKeys {
    table: Vec<Vec<(u32, String)>>,
    size_ctl: i64,
    size: usize,
}

impl JConcurrentKeys {
    /// `new ConcurrentHashMap<>(initialCapacity)` (JDK 21+ sizing).
    pub fn with_capacity(initial: usize) -> Self {
        let size = (1.0f64 + initial as f64 / 0.75f32 as f64) as i64;
        let cap = table_size_for(size);
        JConcurrentKeys { table: Vec::new(), size_ctl: cap as i64, size: 0 }
    }

    fn init_table(&mut self) {
        let n = if self.size_ctl > 0 { self.size_ctl as usize } else { 16 };
        self.table = vec![Vec::new(); n];
        self.size_ctl = (n - (n >> 2)) as i64;
    }

    fn transfer(&mut self) {
        let n = self.table.len();
        let mut next: Vec<Vec<(u32, String)>> = vec![Vec::new(); n << 1];
        for (i, bin) in std::mem::take(&mut self.table).into_iter().enumerate() {
            if bin.is_empty() {
                continue;
            }
            let bit = |h: u32| (h as usize) & n;
            let mut run_bit = bit(bin[0].0);
            let mut last_run = 0;
            for (j, (h, _)) in bin.iter().enumerate().skip(1) {
                let b = bit(*h);
                if b != run_bit {
                    run_bit = b;
                    last_run = j;
                }
            }
            let (mut ln, mut hn): (Vec<(u32, String)>, Vec<(u32, String)>) = (Vec::new(), Vec::new());
            let tail: Vec<(u32, String)> = bin[last_run..].to_vec();
            if run_bit == 0 {
                ln = tail;
            } else {
                hn = tail;
            }
            for (h, k) in bin[..last_run].iter() {
                if bit(*h) == 0 {
                    ln.insert(0, (*h, k.clone()));
                } else {
                    hn.insert(0, (*h, k.clone()));
                }
            }
            next[i] = ln;
            next[i + n] = hn;
        }
        self.table = next;
        let nn = self.table.len();
        self.size_ctl = (nn - (nn >> 2)) as i64;
    }

    fn try_presize(&mut self, size: usize) {
        let c = table_size_for((size + (size >> 1) + 1) as i64);
        loop {
            if (c as i64) <= self.size_ctl || self.table.len() >= (1 << 30) {
                break;
            }
            self.transfer();
        }
    }

    pub fn contains(&self, k: &str) -> bool {
        if self.table.is_empty() {
            return false;
        }
        let h = spread_chm(string_hash(k));
        self.table[(h as usize) & (self.table.len() - 1)].iter().any(|(_, x)| x == k)
    }

    /// `putIfAbsent` / `add`; returns whether the key was inserted.
    pub fn insert(&mut self, k: &str) -> bool {
        if self.table.is_empty() {
            self.init_table();
        }
        let h = spread_chm(string_hash(k));
        let n = self.table.len();
        let bin = &mut self.table[(h as usize) & (n - 1)];
        if bin.iter().any(|(_, x)| x == k) {
            return false;
        }
        bin.push((h, k.to_string()));
        let bin_count = bin.len();
        if bin_count >= 8 && n < 64 {
            self.try_presize(n << 1);
        }
        self.size += 1;
        while self.size as i64 >= self.size_ctl && self.table.len() < (1 << 30) {
            self.transfer();
        }
        true
    }

    /// Keys in iteration order.
    pub fn keys(&self) -> Vec<String> {
        self.table.iter().flat_map(|b| b.iter().map(|(_, k)| k.clone())).collect()
    }
}

/// `Double.toString`-style rendering is not needed: ES renders these values
/// through `NumberUtils.doubleToString` (four fractional digits).
pub fn double_to_string(value: f64) -> String {
    if value < 0.0001 {
        return "0".to_string();
    }
    let i = value as i32;
    let f = ((value - i as f64) * 10000.0 + 0.5) as i32;
    format!("{i}.{f:04}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes() {
        assert_eq!(string_hash("abc"), 96354);
        assert_eq!(objects_hash(&[J::S("a"), J::I(1)]), 31 * (31 + 97) + 1);
    }

    #[test]
    fn hashmap_order_small() {
        let mut m = JHashMap::default();
        for k in ["b", "a", "c"] {
            m.put(k.to_string(), 1);
        }
        let keys: Vec<&String> = m.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn double_fmt() {
        assert_eq!(double_to_string(0.0004359), "0.0004");
        assert_eq!(double_to_string(0.00001), "0");
        assert_eq!(double_to_string(12.5), "12.5000");
    }
}
