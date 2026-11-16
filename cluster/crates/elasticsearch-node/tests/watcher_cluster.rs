//! Scheduled watches on a three-node cluster of real processes: every
//! `.watches` copy triggers its share of the watches, so each watch fires
//! once per trigger across the cluster (no duplicate scheduled times in the
//! history), and the watches keep firing after the node holding the
//! `.watches` primary is killed (a replica is promoted, the survivors take
//! the watches over).
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

const WATCHES: usize = 6;

/// Per watch: the scheduled times of its history records (with repeats).
async fn history(pl: &Procs, via: usize) -> BTreeMap<String, Vec<String>> {
    let _ = pl.call(via, Method::POST, "/.watcher-history-18*/_refresh?expand_wildcards=all&ignore_unavailable=true", None).await;
    let (st, v) = pl
        .call(via, Method::POST, "/.watcher-history-18*/_search?expand_wildcards=all&ignore_unavailable=true&size=10000", Some(json!({"query": {"match_all": {}}, "_source": ["watch_id", "trigger_event"]})))
        .await;
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    if st != 200 {
        return out;
    }
    for h in v["hits"]["hits"].as_array().cloned().unwrap_or_default() {
        let s = &h["_source"];
        if let (Some(w), Some(t)) = (s["watch_id"].as_str(), s["trigger_event"]["schedule"]["scheduled_time"].as_str()) {
            out.entry(w.to_string()).or_default().push(t.to_string());
        }
    }
    out
}

fn assert_no_duplicates(h: &BTreeMap<String, Vec<String>>) {
    for (w, times) in h {
        let unique: BTreeSet<&String> = times.iter().collect();
        assert_eq!(unique.len(), times.len(), "watch [{w}] fired a trigger more than once: {times:?}");
    }
}

/// Wait until every watch has at least `n` records (scheduled after `since`
/// when given); returns the history.
async fn wait_records(pl: &Procs, via: usize, n: usize, since: Option<&str>, secs: u64) -> BTreeMap<String, Vec<String>> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let h = history(pl, via).await;
        let enough = (0..WATCHES).all(|i| h.get(&format!("w{i}")).is_some_and(|t| t.iter().filter(|t| since.is_none_or(|s| t.as_str() > s)).count() >= n));
        if enough {
            return h;
        }
        assert!(Instant::now() < deadline, "watches did not fire {n} times{}: {h:?}", since.map(|s| format!(" after {s}")).unwrap_or_default());
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scheduled_watches_fire_once_per_trigger_across_node_loss() {
    let mut pl = start_procs(3, &[]).await;
    for i in 0..WATCHES {
        let watch = json!({
            "trigger": {"schedule": {"interval": "2s"}},
            "input": {"simple": {"n": i}},
            "condition": {"always": {}},
            "actions": {"log": {"logging": {"text": "watch {{ctx.watch_id}}"}}}
        });
        let (st, v) = pl.call(0, Method::PUT, &format!("/_watcher/watch/w{i}"), Some(watch)).await;
        assert_eq!(st, 201, "{v}");
    }
    pl.wait_health(0, ".watches", "green", 60).await;
    let before = wait_records(&pl, 0, 3, None, 90).await;
    assert_no_duplicates(&before);

    // Kill the node holding the `.watches` primary (any node: the master
    // too, which forces an election).
    let ids = pl.node_ids(0).await;
    let shards = pl.shards(0, ".watches").await;
    let primary = shards.iter().find(|s| s["prirep"] == "p" && s["state"] == "STARTED").unwrap_or_else(|| panic!("no started .watches primary: {shards:?}"));
    let name = primary["node"].as_str().unwrap().to_string();
    let victim: usize = name.trim_start_matches('n').parse().unwrap_or_else(|_| panic!("node {name} ({ids:?})"));
    let killed_at = chrono_now();
    pl.kill(victim);
    let via = (0..3).find(|i| *i != victim).unwrap();
    pl.wait_nodes(via, 2, 120).await;
    let after = wait_records(&pl, via, 3, Some(&killed_at), 120).await;
    assert_no_duplicates(&after);
    for i in 0..WATCHES {
        let w = format!("w{i}");
        assert!(after[&w].len() > before[&w].len(), "{w} kept firing: {:?} -> {:?}", before[&w], after[&w]);
    }

    // GET watch reads `.watches` (its status is written by whichever node
    // executes the watch), never a stale cached copy.
    for i in 0..WATCHES {
        let (st, doc) = pl.call(via, Method::GET, &format!("/.watches/_doc/w{i}"), None).await;
        assert_eq!(st, 200, "{doc}");
        let (st, got) = pl.call(via, Method::GET, &format!("/_watcher/watch/w{i}"), None).await;
        assert_eq!(st, 200, "{got}");
        assert!(got["_seq_no"].as_i64().unwrap() >= doc["_seq_no"].as_i64().unwrap(), "stale watch: {got} vs {doc}");
        assert!(got["status"]["last_checked"].is_string(), "{got}");
    }

    // History retention is ILM's (ES WatcherIndexTemplateRegistry).
    let (st, tpl) = pl.call(via, Method::GET, "/_index_template/.watch-history-18", None).await;
    assert_eq!(st, 200, "{tpl}");
    let settings = &tpl["index_templates"][0]["index_template"]["template"]["settings"];
    let named = settings["index"]["lifecycle"]["name"] == "watch-history-ilm-policy-16" || settings["index.lifecycle.name"] == "watch-history-ilm-policy-16";
    assert!(named, "{tpl}");
    let (st, policy) = pl.call(via, Method::GET, "/_ilm/policy/watch-history-ilm-policy-16", None).await;
    assert_eq!(st, 200, "{policy}");
    assert_eq!(policy["watch-history-ilm-policy-16"]["policy"]["phases"]["delete"]["min_age"], "4d", "{policy}");
    let (st, explain) = pl.call(via, Method::GET, "/.watcher-history-18/_ilm/explain", None).await;
    assert_eq!(st, 200, "{explain}");
    assert!(explain["indices"].as_object().is_some_and(|m| !m.is_empty() && m.values().all(|i| i["managed"] == true && i["policy"] == "watch-history-ilm-policy-16")), "{explain}");

    // `_watcher/_stop` stops every node's scheduler (cluster metadata), and
    // `_start` resumes them.
    let (st, v) = pl.call(via, Method::POST, "/_watcher/_stop", None).await;
    assert_eq!(st, 200, "{v}");
    let (_, cs) = pl.call(via, Method::GET, "/_cluster/state/metadata", None).await;
    assert_eq!(cs["metadata"]["watcher"]["manually_stopped"], true, "{cs}");
    // Executions already started may still land.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let total = |h: &BTreeMap<String, Vec<String>>| h.values().map(Vec::len).sum::<usize>();
    let stopped = total(&history(&pl, via).await);
    tokio::time::sleep(Duration::from_secs(6)).await;
    let still = history(&pl, via).await;
    assert_eq!(total(&still), stopped, "no node fires while watcher is stopped: {still:?}");
    let resumed_at = chrono_now();
    let (st, v) = pl.call(via, Method::POST, "/_watcher/_start", None).await;
    assert_eq!(st, 200, "{v}");
    let resumed = wait_records(&pl, via, 2, Some(&resumed_at), 60).await;
    assert_no_duplicates(&resumed);
}

/// Now as the history formats `scheduled_time` (UTC, millis), for ordering.
fn chrono_now() -> String {
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
    let secs = ms.div_euclid(1000);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z", rem / 3600, rem % 3600 / 60, rem % 60, ms.rem_euclid(1000))
}
