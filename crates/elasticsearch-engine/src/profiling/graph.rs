//! `POST /_profiling/flamegraph` and `POST /_profiling/topn/functions`
//! (ES `TransportGetFlamegraphAction`, `TransportGetTopNFunctionsAction`,
//! `FrameGroupID`, `TopNFunction`).

use super::java::{double_to_string, objects_hash, JHashMap, J};
use super::stacktraces::{StackFrame, StackTracesResponse, SubGroup};
use serde_json::{json, Map, Value};

/// `NumberUtils.doubleToString` as a JSON number.
fn num4(v: f64) -> Value {
    let s = double_to_string(v);
    serde_json::from_str(&s).unwrap_or(json!(0))
}

/// `FrameGroupID.getBasenameAndParent`.
fn basename_and_parent(p: &str) -> &str {
    if p.is_empty() {
        return p;
    }
    let Some(last) = p.rfind('/') else { return p };
    if last == 0 {
        return p;
    }
    match p[..last].rfind('/') {
        None => p,
        Some(n) => &p[n + 1..],
    }
}

/// `FrameGroupID.create`.
pub fn frame_group_id(file_id: &str, address_or_line: i32, exe: &str, source_file: &str, function: &str) -> String {
    let h = if function.is_empty() {
        objects_hash(&[J::S(file_id), J::I(address_or_line)])
    } else if source_file.is_empty() {
        objects_hash(&[J::S(file_id), J::S(function)])
    } else {
        objects_hash(&[J::S(exe), J::S(function), J::S(basename_and_parent(source_file))])
    };
    h.to_string()
}

struct Node {
    edges: JHashMap<usize>,
    file_id: String,
    frame_type: i32,
    inline: bool,
    exe: String,
    address_or_line: i32,
    function_name: String,
    function_offset: i32,
    source_file: String,
    source_line: i32,
    count_inclusive: i64,
    count_exclusive: i64,
    co2_inclusive: f64,
    co2_exclusive: f64,
    costs_inclusive: f64,
    costs_exclusive: f64,
}

/// `TransportGetFlamegraphAction.buildFlamegraph`.
pub fn flamegraph(resp: &StackTracesResponse) -> Value {
    let mut nodes: Vec<Node> = Vec::new();
    let (mut self_cpu, mut total_cpu) = (0i64, 0i64);
    let new_node = |file_id: &str, frame_type: i32, inline: bool, exe: &str, aol: i32, fname: &str, foff: i32, sfile: &str, sline: i32, samples: i64, co2: f64, costs: f64| Node {
        edges: JHashMap::default(),
        file_id: file_id.into(),
        frame_type,
        inline,
        exe: exe.into(),
        address_or_line: aol,
        function_name: fname.into(),
        function_offset: foff,
        source_file: sfile.into(),
        source_line: sline,
        count_inclusive: samples,
        count_exclusive: 0,
        co2_inclusive: co2,
        co2_exclusive: 0.0,
        costs_inclusive: costs,
        costs_exclusive: 0.0,
    };
    // ES 9: the root is `FRAMETYPE_ROOT`; each trace event hangs under an
    // executable node and a thread node (`addOrUpdateAggregationNode`).
    const FRAMETYPE_ROOT: i32 = 0x100;
    const FRAMETYPE_THREAD: i32 = 0x102;
    const FRAMETYPE_EXECUTABLE: i32 = 0x103;
    nodes.push(new_node("", FRAMETYPE_ROOT, false, "", 0, "", 0, "", 0, 0, 0.0, 0.0));
    if resp.total_frames != 0 {
        let traces: std::collections::HashMap<&str, &super::stacktraces::StackTrace> = resp.stack_traces.iter().flatten().map(|(k, v)| (k.as_str(), v)).collect();
        let empty = StackFrame::empty();
        for ev in &resp.trace_events {
            let Some(st) = traces.get(ev.stacktrace.as_str()) else { continue };
            let (samples, co2, costs) = (ev.count, ev.annual_co2_tons, ev.annual_costs_usd);
            let mut exe_name = ev.executable.as_str();
            if exe_name.is_empty() && st.type_ids.first() == Some(&super::stacktraces::KERNEL_FRAME_TYPE) {
                exe_name = "kernel";
            }
            nodes[0].count_inclusive += samples;
            total_cpu += samples;
            nodes[0].co2_inclusive += co2;
            nodes[0].costs_inclusive += costs;
            let mut current = 0usize;
            for (name, ft) in [(exe_name, FRAMETYPE_EXECUTABLE), (ev.thread.as_str(), FRAMETYPE_THREAD)] {
                let gid = objects_hash(&[J::S(name), J::I(ft)]).to_string();
                current = match nodes[current].edges.get(&gid).copied() {
                    Some(id) => {
                        nodes[id].count_inclusive += samples;
                        total_cpu += samples;
                        nodes[id].co2_inclusive += co2;
                        nodes[id].costs_inclusive += costs;
                        id
                    }
                    None => {
                        let id = nodes.len();
                        nodes.push(new_node("", ft, false, name, 0, "", 0, "", 0, samples, co2, costs));
                        total_cpu += samples;
                        nodes[current].edges.put(gid, id);
                        id
                    }
                };
            }
            let n = st.frame_ids.len();
            for i in 0..n {
                let file_id = &st.file_ids[i];
                let frame_type = st.type_ids[i];
                let aol = st.address_or_lines[i];
                let sf = resp.frame(&st.frame_ids[i]).unwrap_or(&empty);
                let exe = resp.executables.as_ref().and_then(|e| e.iter().find(|(k, _)| k == file_id)).map(|(_, v)| v.as_str()).unwrap_or("");
                let leaf = i == n - 1;
                for f in sf.frames() {
                    let gid = frame_group_id(file_id, aol, exe, &f.file_name, &f.function_name);
                    let node_id = match nodes[current].edges.get(&gid).copied() {
                        Some(id) => {
                            nodes[id].count_inclusive += samples;
                            total_cpu += samples;
                            nodes[id].co2_inclusive += co2;
                            nodes[id].costs_inclusive += costs;
                            id
                        }
                        None => {
                            let id = nodes.len();
                            nodes.push(new_node(file_id, frame_type, f.inline, exe, aol, &f.function_name, f.function_offset, &f.file_name, f.line_number, samples, co2, costs));
                            total_cpu += samples;
                            nodes[current].edges.put(gid, id);
                            id
                        }
                    };
                    if leaf && f.last {
                        nodes[node_id].count_exclusive += samples;
                        self_cpu += samples;
                        nodes[node_id].co2_exclusive += co2;
                        nodes[node_id].costs_exclusive += costs;
                    }
                    current = node_id;
                }
            }
        }
    }
    let col = |f: &dyn Fn(&Node) -> Value| -> Value { Value::Array(nodes.iter().map(f).collect()) };
    let mut o = Map::new();
    o.insert("Edges".into(), col(&|n| Value::Array(n.edges.values().map(|v| json!(v)).collect())));
    o.insert("FileID".into(), col(&|n| json!(n.file_id)));
    o.insert("FrameType".into(), col(&|n| json!(n.frame_type)));
    o.insert("Inline".into(), col(&|n| json!(n.inline)));
    o.insert("ExeFilename".into(), col(&|n| json!(n.exe)));
    o.insert("AddressOrLine".into(), col(&|n| json!(n.address_or_line)));
    o.insert("FunctionName".into(), col(&|n| json!(n.function_name)));
    o.insert("FunctionOffset".into(), col(&|n| json!(n.function_offset)));
    o.insert("SourceFilename".into(), col(&|n| json!(n.source_file)));
    o.insert("SourceLine".into(), col(&|n| json!(n.source_line)));
    o.insert("CountInclusive".into(), col(&|n| json!(n.count_inclusive)));
    o.insert("CountExclusive".into(), col(&|n| json!(n.count_exclusive)));
    o.insert("AnnualCO2TonsInclusive".into(), col(&|n| num4(n.co2_inclusive)));
    o.insert("AnnualCO2TonsExclusive".into(), col(&|n| num4(n.co2_exclusive)));
    o.insert("AnnualCostsUSDInclusive".into(), col(&|n| num4(n.costs_inclusive)));
    o.insert("AnnualCostsUSDExclusive".into(), col(&|n| num4(n.costs_exclusive)));
    o.insert("Size".into(), json!(nodes.len()));
    o.insert("SamplingRate".into(), json!(resp.sampling_rate));
    o.insert("SelfCPU".into(), json!(self_cpu));
    o.insert("TotalCPU".into(), json!(total_cpu));
    o.insert("TotalSamples".into(), json!(resp.total_samples));
    Value::Object(o)
}

struct TopN {
    id: String,
    rank: usize,
    frame_type: i32,
    inline: bool,
    address_or_line: i32,
    function_name: String,
    source_file: String,
    source_line: i32,
    exe: String,
    self_count: i64,
    total_count: i64,
    self_co2: f64,
    total_co2: f64,
    self_costs: f64,
    total_costs: f64,
    sub_groups: Option<SubGroup>,
}

/// `TransportGetTopNFunctionsAction.buildTopNFunctions`.
pub fn topn_functions(resp: &StackTracesResponse, limit: Option<i64>) -> Value {
    let mut funcs: Vec<TopN> = Vec::new();
    let mut by_id: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    if resp.total_frames != 0 {
        let empty = StackFrame::empty();
        for (_, st) in resp.stack_traces.iter().flatten() {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            let n = st.frame_ids.len();
            for i in 0..n {
                let file_id = &st.file_ids[i];
                let aol = st.address_or_lines[i];
                let sf = resp.frame(&st.frame_ids[i]).unwrap_or(&empty);
                let exe = resp.executables.as_ref().and_then(|e| e.iter().find(|(k, _)| k == file_id)).map(|(_, v)| v.as_str()).unwrap_or("");
                let leaf = i == n - 1;
                for f in sf.frames() {
                    let gid = frame_group_id(file_id, aol, exe, &f.file_name, &f.function_name);
                    let idx = *by_id.entry(gid.clone()).or_insert_with(|| {
                        funcs.push(TopN {
                            id: gid.clone(),
                            rank: 0,
                            frame_type: st.type_ids[i],
                            inline: f.inline,
                            address_or_line: aol,
                            function_name: f.function_name.clone(),
                            source_file: f.file_name.clone(),
                            source_line: f.line_number,
                            exe: exe.to_string(),
                            self_count: 0,
                            total_count: 0,
                            self_co2: 0.0,
                            total_co2: 0.0,
                            self_costs: 0.0,
                            total_costs: 0.0,
                            sub_groups: None,
                        });
                        funcs.len() - 1
                    });
                    let cur = &mut funcs[idx];
                    if let Some(sg) = &st.sub_groups {
                        match &mut cur.sub_groups {
                            None => cur.sub_groups = Some(sg.copy()),
                            Some(mine) => mine.merge(sg),
                        }
                    }
                    if seen.insert(gid) {
                        cur.total_count += st.count;
                        cur.total_co2 += st.annual_co2_tons;
                        cur.total_costs += st.annual_costs_usd;
                    }
                    if leaf && f.last {
                        cur.self_count += st.count;
                        cur.self_co2 += st.annual_co2_tons;
                        cur.self_costs += st.annual_costs_usd;
                    }
                }
            }
        }
    }
    // Descending by (self_count, id).
    funcs.sort_by(|a, b| b.self_count.cmp(&a.self_count).then_with(|| java_compare(&b.id, &a.id)));
    let (mut sum_self, mut sum_total, mut sum_co2, mut sum_costs) = (0i64, 0i64, 0.0f64, 0.0f64);
    for (i, f) in funcs.iter_mut().enumerate() {
        f.rank = i + 1;
        sum_self += f.self_count;
        sum_total += f.total_count;
        sum_co2 += f.self_co2;
        sum_costs += f.self_costs;
    }
    if let Some(l) = limit {
        if l > 0 && (l as usize) < funcs.len() {
            funcs.truncate(l as usize);
        }
    }
    let topn: Vec<Value> = funcs
        .iter()
        .map(|f| {
            let mut o = Map::new();
            o.insert("id".into(), json!(f.id));
            o.insert("rank".into(), json!(f.rank));
            o.insert(
                "frame".into(),
                json!({
                    "frame_type": f.frame_type, "inline": f.inline, "address_or_line": f.address_or_line, "function_name": f.function_name,
                    "file_name": f.source_file, "line_number": f.source_line, "executable_file_name": f.exe,
                }),
            );
            if let Some(sg) = &f.sub_groups {
                let mut m = Map::new();
                sg.render_into(&mut m);
                o.insert("sub_groups".into(), Value::Object(m));
            }
            o.insert("self_count".into(), json!(f.self_count));
            o.insert("total_count".into(), json!(f.total_count));
            o.insert("self_annual_co2_tons".into(), num4(f.self_co2));
            o.insert("total_annual_co2_tons".into(), num4(f.total_co2));
            o.insert("self_annual_costs_usd".into(), num4(f.self_costs));
            o.insert("total_annual_costs_usd".into(), num4(f.total_costs));
            Value::Object(o)
        })
        .collect();
    json!({
        "self_count": sum_self, "total_count": sum_total,
        "self_annual_co2_tons": num4(sum_co2), "self_annual_cost_usd": num4(sum_costs),
        "topn": topn,
    })
}

/// `String.compareTo` (UTF-16 code unit order).
fn java_compare(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}
