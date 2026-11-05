// Shared by the root workspace and cluster/: moved verbatim from
// cluster/crates/elasticsearch-cluster/src/routing.rs, minus `from_metadata`
// (which depends on the cluster state type and stays in the cluster crate).
//! ES-compatible document routing: which shard of an index a document (or a
//! routing value) lives on.
//!
//! This must be bit-for-bit identical to Elasticsearch, otherwise documents
//! indexed by ES and read here (snapshots, rolling migrations, client-side
//! routing in clients that pre-compute shards) land on different shards:
//!
//! * `Murmur3HashFunction.hash(String)`: murmur3 x86 32-bit, seed 0, over the
//!   string's UTF-16 code units, each written as two little-endian bytes.
//! * `IndexRouting`: `shard = floorMod(hash(routing), routing_num_shards) /
//!   (routing_num_shards / number_of_shards)`; with
//!   `index.routing_partition_size > 1` the hash is
//!   `hash(routing) + floorMod(hash(_id), partition_size)` (Java `int`
//!   arithmetic, so the addition wraps).
//! * Indices created on or after `IndexVersions.MOD_ROUTING_FUNCTION`
//!   (ES 9) route with `floorMod(hash, number_of_shards)`
//!   (`IndexRouting.shouldUseShardCountModRouting`); the routing shards
//!   above still apply to older indices.
//! * `MetadataCreateIndexService.calculateNumRoutingShards`: the default
//!   `index.number_of_routing_shards` is the largest `num_shards * 2^k <= 1024`
//!   (at least one split).

use std::fmt;

/// `org.elasticsearch.cluster.routing.Murmur3HashFunction#hash(String)`.
pub fn murmur3_hash(routing: &str) -> i32 {
    let mut bytes = Vec::with_capacity(routing.len() * 2);
    for unit in routing.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    murmur3_x86_32(&bytes, 0) as i32
}

/// Lucene's `StringHelper.murmurhash3_x86_32`.
pub fn murmur3_x86_32(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h1 = seed;
    let mut chunks = data.chunks_exact(4);
    for block in &mut chunks {
        let mut k1 = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        k1 = k1.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let tail = chunks.remainder();
    if !tail.is_empty() {
        let mut k1 = 0u32;
        for (i, b) in tail.iter().enumerate() {
            k1 ^= (*b as u32) << (8 * i);
        }
        k1 = k1.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h1 ^= k1;
    }
    h1 ^= data.len() as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85eb_ca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2_ae35);
    h1 ^= h1 >> 16;
    h1
}

/// `MetadataCreateIndexService.calculateNumRoutingShards` for indices
/// created on 7.0+: the largest `num_shards * 2^k` that is `<= 1024`, with
/// `k >= 1` so every index can be split at least once.
pub fn calculate_num_routing_shards(num_shards: u32) -> u32 {
    let num_shards = num_shards.max(1);
    let log2_max = 10i32; // log2(1024)
    let log2_num = 32 - (num_shards - 1).leading_zeros() as i32; // ceil(log2(num_shards))
    let splits = (log2_max - log2_num).max(1);
    num_shards << splits
}

/// Validation of an explicit `index.number_of_routing_shards` (ES
/// `IndexMetadata.Builder` / `getRoutingFactor`).
pub fn validate_routing_shards(num_shards: u32, routing_num_shards: u32) -> Result<(), String> {
    if routing_num_shards < num_shards || num_shards == 0 || routing_num_shards % num_shards != 0 {
        return Err(format!("the number of source shards [{num_shards}] must be a factor of [{routing_num_shards}]"));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingError {
    /// Partitioned indices need an explicit routing value for single-doc ops.
    RoutingRequired,
}

impl fmt::Display for RoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoutingError::RoutingRequired => f.write_str("A routing value is required for gets from a partitioned index"),
        }
    }
}

/// Shard selection for one index (ES `IndexRouting.Unpartitioned` /
/// `IndexRouting.Partitioned`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexRouting {
    pub number_of_shards: u32,
    pub routing_num_shards: u32,
    /// `index.routing_partition_size` (1 = unpartitioned).
    pub partition_size: u32,
    /// `RoutingFunction.moduloNumberOfShards` (indices created on ES 9),
    /// else `legacyRoutingNumberOfShards`.
    pub modulo: bool,
}

/// `IndexVersions.MOD_ROUTING_FUNCTION` (9.0.0-era index version id).
pub const MOD_ROUTING_FUNCTION: u32 = 9_064_0_00;

/// `IndexRouting.shouldUseShardCountModRouting`.
pub fn should_use_shard_count_mod_routing(index_version_created: u32) -> bool {
    index_version_created >= MOD_ROUTING_FUNCTION
}

impl IndexRouting {
    pub fn new(number_of_shards: u32, routing_num_shards: u32, partition_size: u32) -> IndexRouting {
        let number_of_shards = number_of_shards.max(1);
        let routing_num_shards = if routing_num_shards == 0 { number_of_shards } else { routing_num_shards };
        IndexRouting { number_of_shards, routing_num_shards, partition_size: partition_size.max(1), modulo: false }
    }

    /// Use the ES 9 modulo routing function (see [`should_use_shard_count_mod_routing`]).
    pub fn with_modulo(mut self, modulo: bool) -> IndexRouting {
        self.modulo = modulo;
        self
    }

    /// `routing_num_shards / number_of_shards`.
    pub fn routing_factor(&self) -> u32 {
        (self.routing_num_shards / self.number_of_shards).max(1)
    }

    pub fn is_partitioned(&self) -> bool {
        self.partition_size > 1
    }

    /// `calculateScaledShardId`: floorMod over the routing hash space, then
    /// scaled down to the current shard count.
    pub fn hash_to_shard(&self, hash: i32) -> u32 {
        if self.modulo {
            return hash.rem_euclid(self.number_of_shards as i32) as u32;
        }
        (hash.rem_euclid(self.routing_num_shards as i32) as u32) / self.routing_factor()
    }

    /// Shard for a single-document operation (index/get/delete/update).
    pub fn shard_id(&self, id: &str, routing: Option<&str>) -> Result<u32, RoutingError> {
        if self.is_partitioned() {
            let routing = routing.ok_or(RoutingError::RoutingRequired)?;
            let offset = murmur3_hash(id).rem_euclid(self.partition_size as i32);
            Ok(self.hash_to_shard(murmur3_hash(routing).wrapping_add(offset)))
        } else {
            Ok(self.hash_to_shard(murmur3_hash(routing.unwrap_or(id))))
        }
    }

    /// Shards a search with `?routing=value` must visit: one shard, or
    /// `partition_size` consecutive hash offsets for partitioned indices.
    pub fn search_shards(&self, routing: &str) -> Vec<u32> {
        let h = murmur3_hash(routing);
        let mut out: Vec<u32> = (0..self.partition_size as i32).map(|o| self.hash_to_shard(h.wrapping_add(o))).collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[cfg(test)]
mod es955_vectors;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulo_routing_matches_es_9_5_5() {
        for &(n, key, shard) in super::es955_vectors::PLAIN {
            let r = IndexRouting::new(n, calculate_num_routing_shards(n), 1).with_modulo(true);
            assert_eq!(r.shard_id(key, None).unwrap(), shard, "{n} shards, routing {key:?}");
        }
        let r = IndexRouting::new(6, calculate_num_routing_shards(6), 3).with_modulo(true);
        for &(id, routing, shard) in super::es955_vectors::PARTITIONED {
            assert_eq!(r.shard_id(id, Some(routing)).unwrap(), shard, "id {id} routing {routing}");
        }
        for &(routing, shards) in super::es955_vectors::PARTITIONED_SEARCH {
            assert_eq!(r.search_shards(routing), shards.to_vec(), "routing {routing}");
        }
    }

    #[test]
    fn modulo_routing() {
        // ES 9.5.5 reference placements (5 shards, index created on 9.5.5).
        let r = IndexRouting::new(5, calculate_num_routing_shards(5), 1).with_modulo(true);
        assert_eq!(r.shard_id("1", None).unwrap(), 3);
        assert_eq!(r.shard_id("3", None).unwrap(), 4);
        assert!(should_use_shard_count_mod_routing(9_111_0_00));
        assert!(!should_use_shard_count_mod_routing(8_521_0_00));
    }

    #[test]
    fn murmur3_known_values() {
        // Murmur3HashFunctionTests#testKnownValues in Elasticsearch.
        assert_eq!(murmur3_hash("hell"), 0x5a0cb7c3u32 as i32);
        assert_eq!(murmur3_hash("hello"), 0xd7c31989u32 as i32);
        assert_eq!(murmur3_hash("hello w"), 0x22ab2984u32 as i32);
        assert_eq!(murmur3_hash("hello wo"), 0xdf0ca123u32 as i32);
        assert_eq!(murmur3_hash("hello wor"), 0xe7744d61u32 as i32);
        assert_eq!(murmur3_hash("The quick brown fox jumps over the lazy dog"), 0xe07db09cu32 as i32);
        assert_eq!(murmur3_hash("The quick brown fox jumps over the lazy cog"), 0x4e63d2adu32 as i32);
    }

    #[test]
    fn murmur3_raw_reference_vectors() {
        // Reference vectors of murmur3_x86_32 itself (SMHasher / Appleby).
        assert_eq!(murmur3_x86_32(b"", 0), 0);
        assert_eq!(murmur3_x86_32(b"", 1), 0x514e28b7);
        assert_eq!(murmur3_x86_32(b"", 0xffffffff), 0x81f16f39);
        assert_eq!(murmur3_x86_32(&[0, 0, 0, 0], 0), 0x2362f9de);
        assert_eq!(murmur3_x86_32(b"Hello, world!", 1234), 0xfaf6cdb3);
        assert_eq!(murmur3_x86_32(b"The quick brown fox jumps over the lazy dog", 0x9747b28c), 0x2fa826cd);
    }

    #[test]
    fn non_ascii_uses_utf16_code_units() {
        // A supplementary character is two UTF-16 units (4 bytes), not 4 UTF-8 bytes.
        let s = "\u{1F600}";
        let mut bytes = Vec::new();
        for u in s.encode_utf16() {
            bytes.extend_from_slice(&u.to_le_bytes());
        }
        assert_eq!(bytes.len(), 4);
        assert_eq!(murmur3_hash(s), murmur3_x86_32(&bytes, 0) as i32);
        assert_ne!(murmur3_hash("é"), murmur3_x86_32("é".as_bytes(), 0) as i32);
    }

    #[test]
    fn routing_shards_defaults() {
        assert_eq!(calculate_num_routing_shards(1), 1024);
        assert_eq!(calculate_num_routing_shards(2), 1024);
        assert_eq!(calculate_num_routing_shards(3), 768);
        assert_eq!(calculate_num_routing_shards(5), 640);
        assert_eq!(calculate_num_routing_shards(10), 640);
        assert_eq!(calculate_num_routing_shards(512), 1024);
        assert_eq!(calculate_num_routing_shards(1024), 2048);
        assert_eq!(calculate_num_routing_shards(2000), 4000);
        assert!(validate_routing_shards(5, 640).is_ok());
        assert_eq!(validate_routing_shards(3, 640).unwrap_err(), "the number of source shards [3] must be a factor of [640]");
    }

    #[test]
    fn shard_selection_matches_formula() {
        let r = IndexRouting::new(5, 640, 1);
        assert_eq!(r.routing_factor(), 128);
        for id in ["1", "2", "foo", "bar", "a-much-longer-document-id", "\u{1F600}"] {
            let h = murmur3_hash(id);
            let expected = (((h % 640) + 640) % 640) as u32 / 128;
            assert_eq!(r.shard_id(id, None).unwrap(), expected);
            // routing overrides id
            assert_eq!(r.shard_id("whatever", Some(id)).unwrap(), expected);
        }
        // single shard index: always 0
        let one = IndexRouting::new(1, 1024, 1);
        for id in ["1", "x", "y", "zzzz"] {
            assert_eq!(one.shard_id(id, None).unwrap(), 0);
        }
    }

    #[test]
    fn negative_hashes_use_floor_mod() {
        // "hello" hashes negative (0xd7c31989); floorMod keeps it in range.
        let h = murmur3_hash("hello");
        assert!(h < 0);
        let r = IndexRouting::new(3, 3, 1);
        assert_eq!(r.hash_to_shard(h), (h.rem_euclid(3)) as u32);
        assert_eq!(r.hash_to_shard(-1), 2);
        assert_eq!(r.hash_to_shard(i32::MIN), (i32::MIN.rem_euclid(3)) as u32);
    }

    #[test]
    fn split_index_routes_consistently() {
        // An index split from 2 to 4 shards (routing_num_shards fixed at 1024)
        // routes a doc to a shard whose parent was the doc's old shard.
        let before = IndexRouting::new(2, 1024, 1);
        let after = IndexRouting::new(4, 1024, 1);
        for i in 0..200 {
            let id = format!("doc-{i}");
            assert_eq!(after.shard_id(&id, None).unwrap() / 2, before.shard_id(&id, None).unwrap());
        }
    }

    #[test]
    fn partitioned_routing() {
        let r = IndexRouting::new(6, 6, 3);
        assert_eq!(r.shard_id("1", None), Err(RoutingError::RoutingRequired));
        let base = murmur3_hash("user1");
        let s = r.shard_id("doc", Some("user1")).unwrap();
        let offset = murmur3_hash("doc").rem_euclid(3);
        assert_eq!(s, base.wrapping_add(offset).rem_euclid(6) as u32);
        let search = r.search_shards("user1");
        assert!(search.contains(&s));
        assert!(search.len() <= 3);
        for i in 0..50 {
            let id = format!("d{i}");
            assert!(search.contains(&r.shard_id(&id, Some("user1")).unwrap()));
        }
        // the wrapping addition near i32::MAX must not panic
        assert!(r.hash_to_shard(i32::MAX.wrapping_add(2)) < 6);
    }
}
