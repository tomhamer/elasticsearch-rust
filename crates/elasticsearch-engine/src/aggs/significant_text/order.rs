//! ES significant-term reduction visits its Java-string paged hash table,
//! then drains a score-only min-heap in reverse. Lexical tie-breaking changes
//! both the order and membership of equal-score buckets.

type Bucket = (String, u64, u64, f64);
const EMPTY: usize = usize::MAX;

fn hash(key: &str) -> usize {
    key.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32)) as usize
}

fn insert(slots: &mut [usize], values: &[Bucket], id: usize) {
    let mask = slots.len() - 1;
    let mut slot = hash(&values[id].0) & mask;
    while slots[slot] != EMPTY { slot = (slot + 1) & mask; }
    slots[slot] = id;
}

fn down(heap: &mut [usize], values: &[Bucket]) {
    if heap.is_empty() { return; }
    let mut i = 0;
    let value = heap[0];
    loop {
        let mut child = i * 2 + 1;
        if child >= heap.len() { break; }
        if child + 1 < heap.len() && values[heap[child + 1]].3 < values[heap[child]].3 { child += 1; }
        if values[heap[child]].3 >= values[value].3 { break; }
        heap[i] = heap[child]; i = child;
    }
    heap[i] = value;
}

pub(super) fn select(values: &[Bucket], initial: usize, limit: usize, eligible: impl Fn(&Bucket) -> bool) -> Vec<usize> {
    if limit == 0 || values.is_empty() { return Vec::new(); }
    // AbstractPagedHashMap uses float arithmetic for its 0.6 load factor.
    let capacity = (1 + (initial as f32 / 0.6f32) as usize).next_power_of_two();
    let mut slots = vec![EMPTY; capacity];
    for id in 0..values.len() {
        if id >= (slots.len() as f32 * 0.6f32) as usize {
            let old = slots.len();
            slots.resize(old * 2, EMPTY);
            // Match the in-place rehash, including wrapped collision chains.
            for i in 0..old {
                if slots[i] != EMPTY { let v = slots[i]; slots[i] = EMPTY; insert(&mut slots, values, v); }
            }
            for i in old..slots.len() {
                if slots[i] == EMPTY { break; }
                let v = slots[i]; slots[i] = EMPTY; insert(&mut slots, values, v);
            }
        }
        insert(&mut slots, values, id);
    }
    let mut heap: Vec<usize> = Vec::new();
    for id in slots.into_iter().filter(|id| *id != EMPTY) {
        if !eligible(&values[id]) { continue; }
        if heap.len() < limit {
            heap.push(id);
            let mut i = heap.len() - 1;
            while i > 0 {
                let parent = (i - 1) / 2;
                if values[id].3 >= values[heap[parent]].3 { break; }
                heap[i] = heap[parent]; i = parent;
            }
            heap[i] = id;
        } else if values[id].3 >= values[heap[0]].3 {
            heap[0] = id; down(&mut heap, values);
        }
    }
    let mut result = Vec::with_capacity(heap.len());
    while !heap.is_empty() {
        result.push(heap[0]);
        let last = heap.pop().unwrap();
        if !heap.is_empty() { heap[0] = last; down(&mut heap, values); }
    }
    result.reverse(); result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn score_ties_follow_reference_hash_iteration_and_heap_order() {
        let values: Vec<Bucket> = [("apple", 1.0), ("melon", 0.8), ("banana", 0.6), ("pear", 0.6)]
            .into_iter().map(|(k, s)| (k.into(), 1, 1, s)).collect::<Vec<_>>();
        let keys = |ids: Vec<usize>| ids.into_iter().map(|i| values[i].0.as_str()).collect::<Vec<_>>();
        assert_eq!(keys(select(&values, 4, 10, |_| true)), ["apple", "melon", "pear", "banana"]);
        assert_eq!(keys(select(&values, 4, 2, |_| true)), ["apple", "melon"]);
        assert!(select(&values, 4, 0, |_| true).is_empty());
        let ids = select(&values, 0, 10, |_| true);
        assert_eq!(ids.len(), 4);
        assert_eq!(ids.iter().copied().collect::<std::collections::HashSet<_>>().len(), 4);
    }
}
