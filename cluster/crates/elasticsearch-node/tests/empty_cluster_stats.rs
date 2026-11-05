//! `_stats` on a cluster without indices answers exactly as ES 9.5.5 does:
//! `IndicesStatsResponse` builds `_all.primaries` / `_all.total` as
//! `new CommonStats()` (every section null) and adds no shard, and
//! `CommonStats.toXContent` writes only non-null sections, so both are
//! empty objects (no zeroed sections), after a zero `_shards` header and
//! before an empty `indices`.
mod common;
use common::start;
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_cluster_stats_match_es_shape() {
    let cl = start(&[]).await;
    let want = json!({"_shards": {"total": 0, "successful": 0, "failed": 0}, "_all": {"primaries": {}, "total": {}}, "indices": {}});
    for path in ["/_stats", "/_stats/merge", "/_stats/indexing,refresh,flush", "/_all/_stats"] {
        let r = cl.c.get(format!("http://127.0.0.1:{}{path}", cl.p.http[0])).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200, "{path}");
        let text = r.text().await.unwrap();
        let got: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(got, want, "{path}: {text}");
        // ES's member order: the header, `_all`, then `indices`.
        let (s, a, i) = (text.find("\"_shards\"").unwrap(), text.find("\"_all\"").unwrap(), text.find("\"indices\"").unwrap());
        assert!(s < a && a < i, "{path}: member order {text}");
    }
}
