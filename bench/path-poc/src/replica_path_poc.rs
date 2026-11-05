//! Attribute replica replay time using real Metricbeat mappings and documents.
//! No network, WAL or fencing is modeled: this is a component diagnostic.
use elasticsearch_engine::node::Node;
use rayon::prelude::*;
use serde_json::json;
use std::{hint::black_box, io::{BufRead, BufReader}, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let docs: Vec<String> = BufReader::new(std::fs::File::open(&args[2]).unwrap()).lines()
        .step_by(100).take(10000).map(Result::unwrap).collect();
    assert_eq!(docs.len(), 10000);
    let path = std::path::PathBuf::from(&args[3]).join(format!("replica-component-{}", std::process::id()));
    assert!(!path.exists());
    let node = Node::open(&path).unwrap();
    let mut body = report["mapping"].clone();
    body["settings"]["index.number_of_shards"] = json!(1);
    body["settings"]["index.number_of_replicas"] = json!(0);
    let mut results = Vec::new();
    for pass in 0..3 {
        let svc = node.create_index(&format!("replica-{pass}"), &body).unwrap();
        let shard = svc.shard(0).unwrap();
        // Warm mapping discovery before comparing serial and parallel parse.
        for (i, source) in docs.iter().enumerate() { black_box(svc.parse_for_replica(source, &i.to_string()).unwrap()); }
        // Actual cluster writes serialize/compare the mapping per request.
        // Scale to one snapshot per 500 documents (one half of a bulk).
        let snapshots = docs.len().div_ceil(500);
        let start = Instant::now();
        for _ in 0..snapshots { black_box(svc.meta().mappings.to_json()); }
        results.push(json!({"pass":pass,"stage":"mapping_snapshot","snapshots":snapshots,"ms":start.elapsed().as_secs_f64()*1000.0}));
        assert_eq!(svc.mapping_json(),svc.meta().mappings.to_json());
        let start = Instant::now();
        for _ in 0..snapshots { black_box(svc.mapping_json()); }
        results.push(json!({"pass":pass,"stage":"mapping_snapshot_without_clone","snapshots":snapshots,"ms":start.elapsed().as_secs_f64()*1000.0}));
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let start = Instant::now();
            pool.install(|| docs.par_iter().enumerate().for_each(|(i, source)| {
                black_box(svc.parse_for_replica(source, &i.to_string()).unwrap());
            }));
            results.push(json!({"pass":pass,"stage":"parse_and_finish","threads":threads,"ms":start.elapsed().as_secs_f64()*1000.0}));
        }
        let start = Instant::now();
        for (i, source) in docs.iter().enumerate() {
            shard.replay_source(&i.to_string(), source, None, i as i64, 1).unwrap();
        }
        results.push(json!({"pass":pass,"stage":"full_replay_source","threads":1,"ms":start.elapsed().as_secs_f64()*1000.0}));
        shard.shard.refresh().unwrap();
        assert_eq!(shard.doc_count().0, docs.len() as u64);
        // A separate empty shard receives the same already-parsed documents.
        let apply = node.create_index(&format!("apply-{pass}"), &body).unwrap();
        let parsed: Vec<_> = docs.iter().enumerate().map(|(i, source)| apply.parse_for_replica(source, &i.to_string()).unwrap()).collect();
        let target = apply.shard(0).unwrap();
        target.sync_schema();
        let start = Instant::now();
        for (i, (pd, source)) in parsed.into_iter().zip(&docs).enumerate() {
            target.shard.replay_index(i.to_string(), pd.source.unwrap_or_else(|| source.clone()), pd.doc, pd.ignored, None, i as i64, 1).unwrap();
        }
        results.push(json!({"pass":pass,"stage":"preparsed_replay_index","threads":1,"ms":start.elapsed().as_secs_f64()*1000.0}));
        target.shard.refresh().unwrap();
        assert_eq!(target.doc_count().0, docs.len() as u64);
        let apply = node.create_index(&format!("guarded-{pass}"), &body).unwrap();
        let parsed: Vec<_> = docs.iter().enumerate().map(|(i, source)| apply.parse_for_replica(source, &i.to_string()).unwrap()).collect();
        let target = apply.shard(0).unwrap();
        target.sync_schema();
        let ids: Vec<_> = (0..docs.len()).map(|i|i.to_string()).collect();
        let mut parsed = parsed.into_iter();
        let start = Instant::now();
        for first in (0..docs.len()).step_by(256) {
            let end = (first+256).min(docs.len());
            let ops: Vec<_> = (first..end).map(|i|elasticsearch_engine::shard::ReplayIndex{
                id:&ids[i],source:&docs[i],routing:None,seq_no:i as i64,version:1,
            }).collect();
            let mut prepared: Vec<_> = parsed.by_ref().take(end-first).collect();
            if !target.try_replay_new_batch(&ops,&mut prepared).unwrap() {
                for (op,pd) in ops.iter().zip(prepared) {
                    target.replay_prepared(op.id,op.source,pd,None,op.seq_no,op.version).unwrap();
                }
            }
        }
        results.push(json!({"pass":pass,"stage":"guarded_replica_writer","threads":4,"ms":start.elapsed().as_secs_f64()*1000.0}));
        target.shard.refresh().unwrap();
        assert_eq!(target.doc_count().0,docs.len() as u64);
        assert_eq!(target.shard.local_checkpoint(),docs.len() as i64-1);
        // Upper bound for a future replica batching mechanism. This does NOT
        // implement the WAL, checkpoints, stale checks or partial-failure
        // semantics required by the production replica applier.
        for threads in [1, 4] {
            let apply = node.create_index(&format!("writer-{pass}-{threads}"), &body).unwrap();
            let mut parsed: Vec<_> = docs.iter().enumerate()
                .map(|(i, source)| apply.parse_for_replica(source, &i.to_string()).unwrap()).collect();
            let target = apply.shard(0).unwrap();
            target.sync_schema();
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let start = Instant::now();
            pool.install(|| parsed.par_chunks_mut(256).enumerate().for_each(|(chunk, items)| {
                use elasticsearch_engine::mapping::{ID_FIELD, SOURCE_FIELD, VERSION_FIELD, SEQ_NO_FIELD, PRIMARY_TERM_FIELD, IGNORED_FIELD};
                use rustic_core::FieldValue;
                let (mut writer, schema) = target.shard.core.write().checkout_writer();
                for (offset, pd) in items.iter_mut().enumerate() {
                    let i = chunk * 256 + offset;
                    let mut doc = pd.doc.clone();
                    doc.fields.push((ID_FIELD, FieldValue::Keyword(i.to_string())));
                    doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(pd.source.clone().unwrap_or_else(|| docs[i].clone()))));
                    doc.fields.push((VERSION_FIELD, FieldValue::I64(1)));
                    doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(i as i64)));
                    doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(1)));
                    for field in &pd.ignored { doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(field.clone()))); }
                    writer.add_document(&doc, &schema);
                }
                target.shard.core.write().checkin_writer(writer);
            }));
            results.push(json!({"pass":pass,"stage":"private_writer_upper_bound","threads":threads,"ms":start.elapsed().as_secs_f64()*1000.0}));
            target.shard.refresh().unwrap();
            assert_eq!(target.doc_count().0, docs.len() as u64);
            for i in 0..docs.len() {
                let got = target.shard.get(&i.to_string()).unwrap().unwrap();
                assert_eq!(got.seq_no, i as i64);
                assert_eq!(got.version, 1);
                assert_eq!(serde_json::from_str::<serde_json::Value>(got.source.as_ref().unwrap()).unwrap(),
                    serde_json::from_str::<serde_json::Value>(&docs[i]).unwrap());
            }
        }
        for clients in [1, 4] {
            for auto_id in [false, true] {
                let svc = node.create_index(&format!("primary-{pass}-{clients}-{auto_id}"), &body).unwrap();
                for (i, source) in docs.iter().enumerate() { black_box(svc.parse_for_replica(source, &i.to_string()).unwrap()); }
                let start = Instant::now();
                std::thread::scope(|scope| {
                    for client in 0..clients {
                        let (svc, docs) = (&svc, &docs);
                        scope.spawn(move || {
                            for (batch, sources) in docs.chunks(500).enumerate().skip(client).step_by(clients) {
                                let reqs = sources.iter().enumerate().map(|(offset, source)| elasticsearch_engine::index::WriteReq::Index {
                                    id: (!auto_id).then(|| (batch*500+offset).to_string()), source:source.clone(), routing:None, cond:Default::default(),
                                }).collect();
                                for result in svc.write(reqs,true) { result.unwrap(); }
                            }
                        });
                    }
                });
                results.push(json!({"pass":pass,"stage":"primary_write_request_durability","clients":clients,"auto_id":auto_id,"batch_size":500,"ms":start.elapsed().as_secs_f64()*1000.0}));
                svc.refresh().unwrap();
                assert_eq!(svc.doc_count().0,docs.len() as u64);
            }
        }
    }
    println!("{}", json!({"scope":"replica component only; excludes WAL, fencing, network and coordinator stale checks","documents":docs.len(),"results":results}));
    drop(node);
    std::fs::remove_dir_all(path).unwrap();
}
