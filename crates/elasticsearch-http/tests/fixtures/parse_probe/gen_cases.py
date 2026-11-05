#!/usr/bin/env python3
"""Generates cases.jsonl: malformed requests for the parse-error fidelity probe.

Each case: {"name", "method", "path", "body"} where body is the raw request
text (may be invalid JSON) or null for no body. `probe.py` sends every case
to a reference Elasticsearch and to our server and diffs the error shape.
"""
import json, os

cases = []


def add(name, method, path, body=None):
    cases.append({"name": name, "method": method, "path": path, "body": body})


S = "/probe/_search"

# --- A. JSON syntax errors -------------------------------------------------
syntax = {
    "trailing_comma_obj": '{"query":{"match_all":{}},}',
    "trailing_comma_arr": '{"sort":["n",]}',
    "missing_close": '{"query":{"match_all":{}}',
    "missing_value": '{"size":}',
    "missing_colon": '{"size" 1}',
    "unquoted_key": '{size:1}',
    "single_quotes": "{'size':1}",
    "bad_literal_true": '{"explain":tru}',
    "bad_literal_null": '{"size":nul}',
    "leading_zero": '{"size":01}',
    "plus_number": '{"size":+1}',
    "dot_number": '{"size":.5}',
    "trailing_dot_number": '{"size":1.}',
    "nan": '{"size":NaN}',
    "unterminated_string": '{"query":{"match":{"f":"abc}}}',
    "bad_escape": '{"query":{"match":{"f":"a\\qb"}}}',
    "control_char": '{"query":{"match":{"f":"a\tb"}}}',
    "duplicate_key": '{"size":1,"size":2}',
    "duplicate_key_nested": '{"query":{"match":{"f":{"query":"a","query":"b"}}}}',
    "garbage": 'abc',
    "top_array": '[1,2]',
    "top_string": '"x"',
    "top_number": '5',
    "extra_after": '{"size":1} x',
    "multiline_trailing_comma": '{\n  "query": {\n    "match_all": {}\n  },\n}',
    "multiline_missing_value": '{\n  "size":\n}',
    "close_bracket_mismatch": '{"sort":["n"}',
    "colon_in_array": '{"sort":["n":1]}',
    "empty_key_value": '{"":1}',
    "comment": '{/* c */"size":1}',
}
for k, b in syntax.items():
    add("syntax." + k, "POST", S, b)

# --- B. search top level ----------------------------------------------------
top = {
    "unknown_obj": '{"foo":{}}',
    "unknown_arr": '{"foo":[]}',
    "unknown_str": '{"foo":"x"}',
    "unknown_num": '{"foo":1}',
    "unknown_bool": '{"foo":true}',
    "unknown_null": '{"foo":null}',
    "unknown_after_query": '{"query":{"match_all":{}},"bar":1}',
    "size_str": '{"size":"abc"}',
    "size_float": '{"size":1.5}',
    "size_bool": '{"size":true}',
    "size_arr": '{"size":[]}',
    "size_obj": '{"size":{}}',
    "size_neg": '{"size":-1}',
    "from_str": '{"from":"x"}',
    "from_neg": '{"from":-2}',
    "track_total_hits_str": '{"track_total_hits":"x"}',
    "timeout_num": '{"timeout":5}',
    "timeout_bad": '{"timeout":"5x"}',
    "explain_str": '{"explain":"yes"}',
    "min_score_str": '{"min_score":"x"}',
    "terminate_after_str": '{"terminate_after":"x"}',
    "source_num": '{"_source":5}',
    "source_unknown": '{"_source":{"foo":1}}',
    "stored_fields_num": '{"stored_fields":5}',
    "search_after_obj": '{"search_after":{}}',
    "query_arr": '{"query":[]}',
    "query_str": '{"query":"x"}',
    "query_num": '{"query":1}',
    "post_filter_unknown": '{"post_filter":{"nope":{}}}',
    "indices_boost_bad": '{"indices_boost":5}',
    "script_fields_bad": '{"script_fields":{"a":{"foo":1}}}',
    "docvalue_fields_bad": '{"docvalue_fields":[{"foo":1}]}',
    "stats_bad": '{"stats":5}',
    "version_str": '{"version":"x"}',
    "profile_num": '{"profile":5}',
    "slice_unknown": '{"slice":{"id":0,"max":2,"foo":1}}',
    "runtime_mappings_bad": '{"runtime_mappings":5}',
    "pit_unknown": '{"pit":{"id":"x","foo":1}}',
}
for k, b in top.items():
    add("top." + k, "POST", S, b)

# --- C. queries --------------------------------------------------------------
Q = {
    "match_all": '{"match_all":{ZZ}}',
    "match_none": '{"match_none":{ZZ}}',
    "match": '{"match":{"f":{"query":"x",ZZ}}}',
    "match_phrase": '{"match_phrase":{"f":{"query":"x",ZZ}}}',
    "match_phrase_prefix": '{"match_phrase_prefix":{"f":{"query":"x",ZZ}}}',
    "match_bool_prefix": '{"match_bool_prefix":{"f":{"query":"x",ZZ}}}',
    "multi_match": '{"multi_match":{"query":"x",ZZ}}',
    "term": '{"term":{"f":{"value":"x",ZZ}}}',
    "terms": '{"terms":{"f":["x"],ZZ}}',
    "range": '{"range":{"n":{"gte":1,ZZ}}}',
    "range_top": '{"range":{"n":{"gte":1},ZZ}}',
    "exists": '{"exists":{"field":"f",ZZ}}',
    "prefix": '{"prefix":{"f":{"value":"x",ZZ}}}',
    "wildcard": '{"wildcard":{"f":{"value":"x",ZZ}}}',
    "regexp": '{"regexp":{"f":{"value":"x",ZZ}}}',
    "fuzzy": '{"fuzzy":{"f":{"value":"x",ZZ}}}',
    "ids": '{"ids":{"values":["1"],ZZ}}',
    "bool": '{"bool":{"must":[],ZZ}}',
    "boosting": '{"boosting":{"positive":{"match_all":{}},"negative":{"match_all":{}},"negative_boost":0.5,ZZ}}',
    "constant_score": '{"constant_score":{"filter":{"match_all":{}},ZZ}}',
    "dis_max": '{"dis_max":{"queries":[{"match_all":{}}],ZZ}}',
    "function_score": '{"function_score":{"query":{"match_all":{}},ZZ}}',
    "nested": '{"nested":{"path":"o","query":{"match_all":{}},ZZ}}',
    "query_string": '{"query_string":{"query":"x",ZZ}}',
    "simple_query_string": '{"simple_query_string":{"query":"x",ZZ}}',
    "combined_fields": '{"combined_fields":{"query":"x","fields":["f"],ZZ}}',
    "geo_distance": '{"geo_distance":{"distance":"1km","g":[0,0],ZZ}}',
    "geo_bounding_box": '{"geo_bounding_box":{"g":{"top_left":[0,1],"bottom_right":[1,0]},ZZ}}',
    "script": '{"script":{"script":"true",ZZ}}',
    "knn": '{"knn":{"field":"v","query_vector":[1,1],ZZ}}',
    "span_term": '{"span_term":{"f":{"value":"x",ZZ}}}',
    "span_near": '{"span_near":{"clauses":[{"span_term":{"f":"x"}}],ZZ}}',
    "more_like_this": '{"more_like_this":{"like":"x",ZZ}}',
    "pinned": '{"pinned":{"ids":["1"],"organic":{"match_all":{}},ZZ}}',
    "terms_set": '{"terms_set":{"f":{"terms":["x"],ZZ}}}',
    "rank_feature": '{"rank_feature":{"field":"n",ZZ}}',
    "percolate": '{"percolate":{"field":"p","document":{},ZZ}}',
    "sparse_vector": '{"sparse_vector":{"field":"f","query_vector":{"a":1},ZZ}}',
    "match_two_fields": '{"match":{"f":"x",ZZ}}',
    "term_two_fields": '{"term":{"f":"x",ZZ}}',
}
for k, q in Q.items():
    add("query.unknown_field." + k, "POST", S, '{"query":' + q.replace("ZZ", '"zzz":1') + '}')
# nesting depth
add("query.nested.bool_must_arr", "POST", S, '{"query":{"bool":{"must":[{"match_all":{"zz":1}}]}}}')
add("query.nested.bool_must_obj", "POST", S, '{"query":{"bool":{"must":{"match_all":{"zz":1}}}}}')
add("query.nested.bool_filter_term", "POST", S, '{"query":{"bool":{"filter":[{"term":{"f":{"value":"x","zz":1}}}]}}}')
add("query.nested.bool_should_unknown_q", "POST", S, '{"query":{"bool":{"should":[{"match_all":{}},{"nope":{}}]}}}')
add("query.nested.bool_bool", "POST", S, '{"query":{"bool":{"must":[{"bool":{"filter":[{"ids":{"values":[],"zz":1}}]}}]}}}')
add("query.nested.constant_score_filter", "POST", S, '{"query":{"constant_score":{"filter":{"match_all":{"zz":1}}}}}')
add("query.nested.dis_max", "POST", S, '{"query":{"dis_max":{"queries":[{"nope":{}}]}}}')
add("query.nested.boosting_positive", "POST", S, '{"query":{"boosting":{"positive":{"nope":{}},"negative":{"match_all":{}},"negative_boost":1}}}')
add("query.nested.nested_query", "POST", S, '{"query":{"nested":{"path":"o","query":{"match_all":{"zz":1}}}}}')
add("query.nested.function_score_query", "POST", S, '{"query":{"function_score":{"query":{"nope":{}}}}}')
add("query.nested.function_score_fn", "POST", S, '{"query":{"function_score":{"functions":[{"nope":{}}]}}}')
add("query.nested.post_filter", "POST", S, '{"post_filter":{"bool":{"zz":1}}}')
add("query.nested.multiline", "POST", S, '{\n  "query": {\n    "bool": {\n      "must": [\n        { "match_all": { "zz": 1 } }\n      ]\n    }\n  }\n}')
add("query.unknown.top", "POST", S, '{"query":{"nope":{}}}')
add("query.unknown.suggest_name", "POST", S, '{"query":{"match_al":{}}}')
add("query.unknown.type", "POST", S, '{"query":{"type":{"value":"x"}}}')
add("query.empty", "POST", S, '{"query":{}}')
add("query.two_names", "POST", S, '{"query":{"match_all":{},"match_none":{}}}')
add("query.name_not_obj", "POST", S, '{"query":{"match_all":1}}')
add("query.name_array", "POST", S, '{"query":{"match_all":[]}}')
add("query.bool_must_str", "POST", S, '{"query":{"bool":{"must":"x"}}}')
add("query.bool_boost_str", "POST", S, '{"query":{"bool":{"boost":"abc"}}}')
add("query.match_all_boost_str", "POST", S, '{"query":{"match_all":{"boost":"abc"}}}')
add("query.match_all_boost_arr", "POST", S, '{"query":{"match_all":{"boost":[]}}}')
add("query.term_array", "POST", S, '{"query":{"term":{"f":["a","b"]}}}')
add("query.terms_obj_bad", "POST", S, '{"query":{"terms":{"f":{"foo":1}}}}')
add("query.match_no_query", "POST", S, '{"query":{"match":{"f":{"operator":"and"}}}}')
add("query.range_bad_value_obj", "POST", S, '{"query":{"range":{"n":5}}}')
add("query.exists_no_field", "POST", S, '{"query":{"exists":{}}}')
add("query.ids_values_str", "POST", S, '{"query":{"ids":{"values":"x"}}}')
add("query.multi_match_bad_type", "POST", S, '{"query":{"multi_match":{"query":"x","type":"nope"}}}')
add("query.query_string_unknown_bad_type", "POST", S, '{"query":{"query_string":{"query":"x","default_operator":5}}}')

# --- D. aggregations -----------------------------------------------------
A = {
    "terms": '{"terms":{"field":"f",ZZ}}',
    "avg": '{"avg":{"field":"n",ZZ}}',
    "sum": '{"sum":{"field":"n",ZZ}}',
    "min": '{"min":{"field":"n",ZZ}}',
    "max": '{"max":{"field":"n",ZZ}}',
    "value_count": '{"value_count":{"field":"n",ZZ}}',
    "cardinality": '{"cardinality":{"field":"f",ZZ}}',
    "stats": '{"stats":{"field":"n",ZZ}}',
    "extended_stats": '{"extended_stats":{"field":"n",ZZ}}',
    "percentiles": '{"percentiles":{"field":"n",ZZ}}',
    "histogram": '{"histogram":{"field":"n","interval":1,ZZ}}',
    "date_histogram": '{"date_histogram":{"field":"d","calendar_interval":"day",ZZ}}',
    "range": '{"range":{"field":"n","ranges":[{"to":1}],ZZ}}',
    "range_inner": '{"range":{"field":"n","ranges":[{"to":1,ZZ}]}}',
    "date_range": '{"date_range":{"field":"d","ranges":[{"to":"now"}],ZZ}}',
    "filters": '{"filters":{"filters":{"a":{"match_all":{}}},ZZ}}',
    "missing": '{"missing":{"field":"f",ZZ}}',
    "top_hits": '{"top_hits":{"size":1,ZZ}}',
    "composite": '{"composite":{"sources":[{"a":{"terms":{"field":"f"}}}],ZZ}}',
    "nested": '{"nested":{"path":"o",ZZ}}',
    "significant_terms": '{"significant_terms":{"field":"f",ZZ}}',
    "global": '{"global":{ZZ}}',
    "sampler": '{"sampler":{ZZ}}',
    "geo_bounds": '{"geo_bounds":{"field":"g",ZZ}}',
    "bucket_sort": '{"bucket_sort":{ZZ}}',
}
for k, a in A.items():
    add("aggs.unknown_field." + k, "POST", S, '{"size":0,"aggs":{"a":' + a.replace("ZZ", '"zzz":1') + '}}')
add("aggs.unknown_type", "POST", S, '{"aggs":{"a":{"nope":{"field":"f"}}}}')
add("aggs.unknown_type_suggest", "POST", S, '{"aggs":{"a":{"term":{"field":"f"}}}}')
add("aggs.two_types", "POST", S, '{"aggs":{"a":{"terms":{"field":"f"},"avg":{"field":"n"}}}}')
add("aggs.missing_type", "POST", S, '{"aggs":{"a":{}}}')
add("aggs.missing_type_meta", "POST", S, '{"aggs":{"a":{"meta":{}}}}')
add("aggs.not_object", "POST", S, '{"aggs":{"a":1}}')
add("aggs.aggs_not_object", "POST", S, '{"aggs":1}')
add("aggs.body_not_object", "POST", S, '{"aggs":{"a":{"terms":1}}}')
add("aggs.sub_unknown_type", "POST", S, '{"aggs":{"a":{"terms":{"field":"f"},"aggs":{"b":{"nope":{}}}}}}')
add("aggs.sub_unknown_field", "POST", S, '{"aggs":{"a":{"terms":{"field":"f"},"aggs":{"b":{"avg":{"field":"n","zz":1}}}}}}')
add("aggs.terms_size_str", "POST", S, '{"aggs":{"a":{"terms":{"field":"f","size":"abc"}}}}')
add("aggs.terms_field_num", "POST", S, '{"aggs":{"a":{"terms":{"field":{}}}}}')
add("aggs.histogram_interval_str", "POST", S, '{"aggs":{"a":{"histogram":{"field":"n","interval":"x"}}}}')
add("aggs.filter_unknown_query", "POST", S, '{"aggs":{"a":{"filter":{"nope":{}}}}}')
add("aggs.filter_query_unknown_field", "POST", S, '{"aggs":{"a":{"filter":{"match_all":{"zz":1}}}}}')
add("aggs.bad_name", "POST", S, '{"aggs":{"a[b":{"avg":{"field":"n"}}}}')
add("aggs.multiline_unknown", "POST", S, '{\n "aggs": {\n  "a": {\n   "terms": {\n    "field": "f",\n    "zzz": 1\n   }\n  }\n }\n}')
add("aggs.metric_with_subaggs", "POST", S, '{"aggs":{"a":{"avg":{"field":"n"},"aggs":{"b":{"max":{"field":"n"}}}}}}')

# --- E. sort / highlight / suggest / knn / retriever / collapse / rescore --
E = {
    "sort.unknown_field": '{"sort":[{"n":{"order":"asc","zzz":1}}]}',
    "sort.bad_order": '{"sort":[{"n":{"order":"up"}}]}',
    "sort.bad_order_str": '{"sort":[{"n":"up"}]}',
    "sort.num": '{"sort":5}',
    "sort.nested_unknown": '{"sort":[{"n":{"nested":{"path":"o","zzz":1}}}]}',
    "sort.script_unknown": '{"sort":{"_script":{"type":"number","script":"1","zzz":1}}}',
    "sort.geo_unknown": '{"sort":[{"_geo_distance":{"g":[0,0],"zzz":1}}]}',
    "highlight.unknown_field": '{"highlight":{"fields":{"f":{}},"zzz":1}}',
    "highlight.field_unknown": '{"highlight":{"fields":{"f":{"zzz":1}}}}',
    "highlight.num": '{"highlight":5}',
    "highlight.bad_type": '{"highlight":{"type":"nope","fields":{"f":{}}}}',
    "highlight.fields_num": '{"highlight":{"fields":5}}',
    "suggest.unknown_suggester": '{"suggest":{"s":{"text":"x","nope":{"field":"f"}}}}',
    "suggest.term_unknown": '{"suggest":{"s":{"text":"x","term":{"field":"f","zzz":1}}}}',
    "suggest.phrase_unknown": '{"suggest":{"s":{"text":"x","phrase":{"field":"f","zzz":1}}}}',
    "suggest.completion_unknown": '{"suggest":{"s":{"prefix":"x","completion":{"field":"f","zzz":1}}}}',
    "suggest.not_obj": '{"suggest":{"s":5}}',
    "suggest.no_type": '{"suggest":{"s":{"text":"x"}}}',
    "knn.unknown_field": '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"zzz":1}}',
    "knn.no_field": '{"knn":{"query_vector":[1,1],"k":1,"num_candidates":1}}',
    "knn.k_str": '{"knn":{"field":"v","query_vector":[1,1],"k":"x","num_candidates":1}}',
    "knn.filter_unknown_query": '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"filter":{"nope":{}}}}',
    "knn.vector_str": '{"knn":{"field":"v","query_vector":"x","k":1,"num_candidates":1}}',
    "knn.num": '{"knn":5}',
    "retriever.unknown_type": '{"retriever":{"nope":{}}}',
    "retriever.standard_unknown": '{"retriever":{"standard":{"zzz":1}}}',
    "retriever.standard_query_unknown": '{"retriever":{"standard":{"query":{"nope":{}}}}}',
    "retriever.knn_unknown": '{"retriever":{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"zzz":1}}}',
    "retriever.rrf_unknown": '{"retriever":{"rrf":{"retrievers":[],"zzz":1}}}',
    "retriever.rrf_child_unknown": '{"retriever":{"rrf":{"retrievers":[{"standard":{"zzz":1}}]}}}',
    "retriever.two": '{"retriever":{"standard":{},"knn":{}}}',
    "retriever.num": '{"retriever":5}',
    "collapse.unknown": '{"collapse":{"field":"f","zzz":1}}',
    "collapse.num": '{"collapse":5}',
    "rescore.unknown": '{"rescore":{"window_size":10,"zzz":1}}',
    "rescore.query_unknown": '{"rescore":{"query":{"rescore_query":{"match_all":{}},"zzz":1}}}',
    "rescore.query_unknown_q": '{"rescore":{"query":{"rescore_query":{"nope":{}}}}}',
}
for k, b in E.items():
    add(k, "POST", S, b)

# --- F. opaque ids -------------------------------------------------------------
for k, i in {"foo": "foo", "b64_garbage": "Zm9vYmFy", "empty_like": "AA", "bad_char": "f.o", "short": "A", "pad": "Zm9v="}.items():
    add("id.scroll." + k, "POST", "/_search/scroll", json.dumps({"scroll_id": i}))
    add("id.clear_scroll." + k, "DELETE", "/_search/scroll", json.dumps({"scroll_id": i}))
    add("id.pit_search." + k, "POST", "/_search", json.dumps({"pit": {"id": i}}))
    add("id.pit_close." + k, "DELETE", "/_pit", json.dumps({"id": i}))
    add("id.async_get." + k, "GET", "/_async_search/" + i, None)
    add("id.async_status." + k, "GET", "/_async_search/status/" + i, None)
    add("id.async_delete." + k, "DELETE", "/_async_search/" + i, None)
add("id.scroll.path_form", "GET", "/_search/scroll/foo", None)
add("id.scroll.param", "GET", "/_search/scroll?scroll_id=foo", None)

# --- G. empty bodies (`{}`) and other body parsers ----------------------------
G = [
    ("POST", "/probe-alias/_rollover"),
    ("POST", "/probe-alias/_rollover/new-idx"),
    ("POST", "/_index_template/_simulate_index/probe-x"),
    ("POST", "/_index_template/_simulate"),
    ("POST", "/_validate/query"),
    ("POST", "/probe/_validate/query"),
    ("POST", "/probe/_termvectors"),
    ("PUT", "/_scripts/s1"),
    ("POST", "/_scripts/s1"),
    ("POST", "/probe/_knn_search"),
    ("PUT", "/_synonyms/syn1"),
    ("PUT", "/_synonyms/syn1/r1"),
    ("POST", "/_query_rules/rs1/_test"),
    ("POST", "/_watcher/watch/w1/_execute"),
    ("POST", "/missing-idx/_update_by_query"),
]
for m, p in G:
    add("empty_body." + m + p.replace("/", "_"), m, p, "{}")
add("body.count_unknown", "POST", "/probe/_count", '{"size":1}')
add("body.count_query_unknown", "POST", "/probe/_count", '{"query":{"match_all":{"zz":1}}}')
add("body.validate_unknown_field", "POST", "/probe/_validate/query", '{"query":{"match_all":{"zz":1}}}')
add("body.explain_unknown", "POST", "/probe/_explain/1", '{"query":{"nope":{}}}')
add("body.dbq_unknown_key", "POST", "/probe/_delete_by_query", '{"query":{"match_all":{}},"zz":1}')
add("body.dbq_query_unknown", "POST", "/probe/_delete_by_query", '{"query":{"match_all":{"zz":1}}}')
add("body.ubq_query_unknown", "POST", "/probe/_update_by_query", '{"query":{"nope":{}}}')
add("body.async_submit_unknown", "POST", "/probe/_async_search", '{"query":{"match_all":{"zz":1}}}')
add("body.async_submit_syntax", "POST", "/probe/_async_search", '{"size":}')
add("body.search_missing_index_bad_query", "POST", "/no-such-index/_search", '{"query":{"nope":{}}}')
add("body.search_get_source_param", "GET", "/probe/_search?source=%7B%22size%22%3A%22x%22%7D&source_content_type=application/json", None)
add("body.create_index_syntax", "PUT", "/probe-new", '{"settings":{}')
add("body.index_doc_syntax", "PUT", "/probe/_doc/9", '{"f":}')

# --- H. extra calibration cases ------------------------------------------------
H = {
    "syntax.eof_after_key": '{"size"',
    "syntax.eof_after_colon": '{"size":',
    "syntax.eof_after_comma": '{"size":1,',
    "syntax.eof_in_key": '{"si',
    "syntax.eof_in_array": '{"sort":[',
    "syntax.eof_after_array_value": '{"sort":["n"',
    "syntax.minus_only": '{"size":-}',
    "syntax.exp_no_digit": '{"size":1e}',
    "syntax.exp_sign_no_digit": '{"size":1e+}',
    "syntax.frac_then_letter": '{"size":1.5x}',
    "syntax.int_then_letter": '{"size":1x}',
    "syntax.true_then_letter": '{"explain":truex}',
    "syntax.ctrl_between_tokens": '{"size":\x011}',
    "syntax.missing_comma_obj": '{"size":1 "from":2}',
    "syntax.missing_comma_arr": '{"sort":["a" "b"]}',
    "syntax.double_comma": '{"size":1,,"from":2}',
    "syntax.key_not_string_in_arr": '{"sort":[1:2]}',
    "syntax.nested_unclosed_inner": '{"query":{"match_all":{}',
    "syntax.unicode_escape_bad": '{"query":{"match":{"f":"a\\u12"}}}',
    "syntax.nonascii_unexpected": '{"size":é}',
    "syntax.multibyte_before_error": '{"query":{"match":{"f":"ééé"}},"size":}',
    "syntax.long_body_truncate": '{"query":{"match":{"f":"' + "a" * 600 + '"}},"size":}',
    "syntax.top_true": 'true',
    "syntax.top_null": 'null',
    "syntax.top_object_then_object": '{"size":1}{"size":2}',
    "syntax.string_escape_tab": '{"query":{"match":{"f":"a\\tb"}},"size":}',
    "top.size_null": '{"size":null}',
    "top.size_str_num": '{"size":"5"}',
    "top.explain_num": '{"explain":1}',
    "top.timeout_obj": '{"timeout":{}}',
    "top.track_total_hits_obj": '{"track_total_hits":{}}',
    "top.min_score_arr": '{"min_score":[]}',
    "top.sort_str_bad": '{"sort":"nope_field"}',
    "top.aggs_arr": '{"aggs":[]}',
    "top.highlight_arr": '{"highlight":[]}',
    "top.suggest_num": '{"suggest":1}',
    "top.from_float": '{"from":1.7}',
    "top.version_num": '{"version":1}',
    "top.seq_no_primary_term_str": '{"seq_no_primary_term":"x"}',
    "top.track_scores_str": '{"track_scores":"x"}',
    "top.fields_num": '{"fields":5}',
    "top.fields_obj_unknown": '{"fields":[{"field":"f","zzz":1}]}',
    "top.ext_bad": '{"ext":5}',
    "top.rank_bad": '{"rank":5}',
    "top.sub_searches_bad": '{"sub_searches":5}',
    "top.name_str": '{"_name":"x"}',
    "top.pit_num": '{"pit":5}',
    "top.pit_keep_alive_bad": '{"pit":{"id":"Zm9vYmFy","keep_alive":"5x"}}',
    "top.pit_no_id": '{"pit":{"keep_alive":"1m"}}',
    "top.indices_boost_arr_unknown": '{"indices_boost":[{"probe":1,"zz":2}]}',
    "top.indices_boost_obj_str": '{"indices_boost":{"probe":"x"}}',
    "top.search_after_num": '{"search_after":5}',
    "top.explain_arr": '{"explain":[]}',
    "query.unknown_field.has_child": '{"query":{"has_child":{"type":"c","query":{"match_all":{}},"zzz":1}}}',
    "query.unknown_field.has_parent": '{"query":{"has_parent":{"parent_type":"p","query":{"match_all":{}},"zzz":1}}}',
    "query.unknown_field.parent_id": '{"query":{"parent_id":{"type":"c","id":"1","zzz":1}}}',
    "query.unknown_field.script_score": '{"query":{"script_score":{"query":{"match_all":{}},"script":"1","zzz":1}}}',
    "query.unknown_field.wrapper": '{"query":{"wrapper":{"query":"e30=","zzz":1}}}',
    "query.unknown_field.distance_feature": '{"query":{"distance_feature":{"field":"d","origin":"now","pivot":"1d","zzz":1}}}',
    "query.unknown_field.intervals": '{"query":{"intervals":{"t":{"match":{"query":"x"},"zzz":1}}}}',
    "query.unknown_field.intervals_top": '{"query":{"intervals":{"t":{"match":{"query":"x"}},"zzz":1}}}',
    "query.unknown_field.span_or": '{"query":{"span_or":{"clauses":[{"span_term":{"f":"x"}}],"zzz":1}}}',
    "query.unknown_field.span_not": '{"query":{"span_not":{"include":{"span_term":{"f":"x"}},"exclude":{"span_term":{"f":"y"}},"zzz":1}}}',
    "query.unknown_field.span_first": '{"query":{"span_first":{"match":{"span_term":{"f":"x"}},"end":1,"zzz":1}}}',
    "query.unknown_field.span_multi": '{"query":{"span_multi":{"match":{"prefix":{"f":"x"}},"zzz":1}}}',
    "query.unknown_field.span_containing": '{"query":{"span_containing":{"little":{"span_term":{"f":"x"}},"big":{"span_term":{"f":"y"}},"zzz":1}}}',
    "query.unknown_field.span_within": '{"query":{"span_within":{"little":{"span_term":{"f":"x"}},"big":{"span_term":{"f":"y"}},"zzz":1}}}',
    "query.unknown_field.span_field_masking": '{"query":{"span_field_masking":{"query":{"span_term":{"f":"x"}},"field":"t","zzz":1}}}',
    "query.unknown_field.geo_shape": '{"query":{"geo_shape":{"g":{"shape":{"type":"point","coordinates":[0,0]}},"zzz":1}}}',
    "query.unknown_field.geo_shape_inner": '{"query":{"geo_shape":{"g":{"shape":{"type":"point","coordinates":[0,0]},"zzz":1}}}}',
    "query.unknown_field.geo_polygon": '{"query":{"geo_polygon":{"g":{"points":[[0,0],[1,1],[0,1]]},"zzz":1}}}',
    "query.unknown_field.geo_grid": '{"query":{"geo_grid":{"g":{"geohash":"u0"},"zzz":1}}}',
    "query.unknown_field.shape": '{"query":{"shape":{"g":{"shape":{"type":"point","coordinates":[0,0]}},"zzz":1}}}',
    "query.unknown_field.text_expansion": '{"query":{"text_expansion":{"f":{"model_id":"m","model_text":"x","zzz":1}}}}',
    "query.unknown_field.weighted_tokens": '{"query":{"weighted_tokens":{"f":{"tokens":{"a":1},"zzz":1}}}}',
    "query.unknown_field.semantic": '{"query":{"semantic":{"field":"f","query":"x","zzz":1}}}',
    "query.unknown_field.rule": '{"query":{"rule":{"organic":{"match_all":{}},"ruleset_ids":["a"],"match_criteria":{},"zzz":1}}}',
    "query.unknown_field.match_inner_fieldless": '{"query":{"match":{"zzz":1}}}',
    "query.unknown_field.term_inner_obj_unknown": '{"query":{"term":{"f":{"zzz":1}}}}',
    "query.unknown_field.range_inner_only": '{"query":{"range":{"n":{"zzz":1}}}}',
    "query.unknown_field.prefix_short": '{"query":{"prefix":{"f":"x","zzz":1}}}',
    "query.unknown_field.wildcard_short": '{"query":{"wildcard":{"f":"x","zzz":1}}}',
    "query.unknown_field.fuzzy_short": '{"query":{"fuzzy":{"f":"x","zzz":1}}}',
    "query.unknown_field.regexp_short": '{"query":{"regexp":{"f":"x","zzz":1}}}',
    "query.unknown_field.match_phrase_short": '{"query":{"match_phrase":{"f":"x","zzz":1}}}',
    "query.unknown_field.ids_bool_prefix_short": '{"query":{"match_bool_prefix":{"f":"x","zzz":1}}}',
    "query.unknown_field.terms_set_short": '{"query":{"terms_set":{"f":{"terms":["x"]},"zzz":1}}}',
    "query.unknown_field.span_term_short": '{"query":{"span_term":{"f":"x","zzz":1}}}',
    "query.unknown_field.exists_str": '{"query":{"exists":{"field":"f","zzz":"q"}}}',
    "query.unknown_field.exists_obj": '{"query":{"exists":{"field":"f","zzz":{}}}}',
    "query.unknown_field.exists_arr": '{"query":{"exists":{"field":"f","zzz":[]}}}',
    "query.unknown_field.match_obj": '{"query":{"match":{"f":{"query":"x","zzz":{}}}}}',
    "query.unknown_field.match_arr": '{"query":{"match":{"f":{"query":"x","zzz":[]}}}}',
    "query.unknown_field.bool_obj": '{"query":{"bool":{"must":[],"zzz":{}}}}',
    "query.unknown_field.bool_arr": '{"query":{"bool":{"must":[],"zzz":[]}}}',
    "query.unknown_field.range_obj": '{"query":{"range":{"n":{"gte":1,"zzz":{}}}}}',
    "query.unknown_field.dis_max_obj": '{"query":{"dis_max":{"queries":[],"zzz":{}}}}',
    "query.unknown_field.function_score_obj": '{"query":{"function_score":{"query":{"match_all":{}},"zzz":{}}}}',
    "query.unknown_field.function_score_fn_unknown": '{"query":{"function_score":{"functions":[{"weight":2,"zzz":1}]}}}',
    "query.unknown_field.nested_obj": '{"query":{"nested":{"path":"o","query":{"match_all":{}},"zzz":{}}}}',
    "query.unknown_field.query_string_obj": '{"query":{"query_string":{"query":"x","zzz":{}}}}',
    "query.unknown_field.more_like_this_obj": '{"query":{"more_like_this":{"like":"x","zzz":{}}}}',
    "query.unknown_field.terms_arr_second": '{"query":{"terms":{"f":["x"],"zzz":["y"]}}}',
    "query.unknown_field.multi_match_obj": '{"query":{"multi_match":{"query":"x","zzz":{}}}}',
    "query.unknown_field.script_obj": '{"query":{"script":{"script":"true","zzz":{}}}}',
    "query.unknown_field.constant_score_arr": '{"query":{"constant_score":{"filter":{"match_all":{}},"zzz":[]}}}',
    "query.unknown_field.boosting_obj": '{"query":{"boosting":{"positive":{"match_all":{}},"negative":{"match_all":{}},"negative_boost":0.5,"zzz":{}}}}',
    "query.unknown_field.geo_distance_obj": '{"query":{"geo_distance":{"distance":"1km","g":[0,0],"zzz":{}}}}',
    "query.unknown_field.span_near_obj": '{"query":{"span_near":{"clauses":[{"span_term":{"f":"x"}}],"zzz":{}}}}',
    "query.unknown.inside_filter_agg_bool": '{"aggs":{"a":{"filter":{"bool":{"must":[{"nope":{}}]}}}}}',
    "query.unknown.deep_nested": '{"query":{"nested":{"path":"o","query":{"bool":{"should":[{"match_all":{}},{"nope":{}}]}}}}}',
    "query.unknown.suggest_2": '{"query":{"terms_se":{}}}',
    "query.unknown.suggest_3": '{"query":{"bol":{}}}',
    "query.unknown.suggest_4": '{"query":{"rang":{}}}',
    "query.unknown.suggest_5": '{"query":{"geo_distanc":{}}}',
    "query.unknown.suggest_6": '{"query":{"span_nea":{}}}',
    "query.unknown.suggest_7": '{"query":{"knn_":{}}}',
    "query.unknown.suggest_8": '{"query":{"sparse_vecto":{}}}',
    "query.unknown.suggest_9": '{"query":{"intervals_":{}}}',
    "query.unknown.suggest_10": '{"query":{"semanti":{}}}',
    "query.unknown.suggest_11": '{"query":{"wrappe":{}}}',
    "query.unknown.suggest_12": '{"query":{"percolat":{}}}',
    "query.unknown.suggest_13": '{"query":{"pinne":{}}}',
    "query.unknown.suggest_14": '{"query":{"rul":{}}}',
    "query.unknown.suggest_15": '{"query":{"text_expansio":{}}}',
    "query.unknown.suggest_16": '{"query":{"shap":{}}}',
    "query.unknown.suggest_17": '{"query":{"exac":{}}}',
    "query.unknown.suggest_18": '{"query":{"kq":{}}}',
    "query.unknown.suggest_19": '{"query":{"random_scor":{}}}',
    "query.unknown.suggest_20": '{"query":{"field_masking_spa":{}}}',
    "aggs.unknown_type_suggest_2": '{"aggs":{"a":{"avgg":{"field":"n"}}}}',
    "aggs.unknown_type_suggest_3": '{"aggs":{"a":{"date_histo":{"field":"d"}}}}',
    "aggs.unknown_type_suggest_4": '{"aggs":{"a":{"percentile":{"field":"n"}}}}',
    "aggs.unknown_type_suggest_5": '{"aggs":{"a":{"top_hit":{}}}}',
    "aggs.unknown_type_suggest_6": '{"aggs":{"a":{"geo":{}}}}',
    "aggs.unknown_type_suggest_7": '{"aggs":{"a":{"bucket_":{}}}}',
    "aggs.unknown_type_suggest_8": '{"aggs":{"a":{"sum_bucke":{}}}}',
    "aggs.unknown_type_suggest_9": '{"aggs":{"a":{"rare_term":{}}}}',
    "aggs.unknown_type_suggest_10": '{"aggs":{"a":{"string_stat":{}}}}',
    "aggs.unknown_field.filter": '{"aggs":{"a":{"filter":{"match_all":{}},"zzz":1}}}',
    "aggs.unknown_field.meta_bad": '{"aggs":{"a":{"meta":5,"avg":{"field":"n"}}}}',
    "aggs.unknown_field.terms_order_bad": '{"aggs":{"a":{"terms":{"field":"f","order":5}}}}',
    "aggs.unknown_field.terms_obj": '{"aggs":{"a":{"terms":{"field":"f","zzz":{}}}}}',
    "aggs.unknown_field.terms_arr": '{"aggs":{"a":{"terms":{"field":"f","zzz":[]}}}}',
    "aggs.unknown_field.avg_obj": '{"aggs":{"a":{"avg":{"field":"n","zzz":{}}}}}',
    "aggs.unknown_field.avg_bucket": '{"aggs":{"h":{"terms":{"field":"f"},"aggs":{"m":{"avg":{"field":"n"}}}},"b":{"avg_bucket":{"buckets_path":"h>m","zzz":1}}}}',
    "aggs.unknown_field.date_range_inner": '{"aggs":{"a":{"date_range":{"field":"d","ranges":[{"to":"now","zzz":1}]}}}}',
    "aggs.unknown_field.ip_range": '{"aggs":{"a":{"ip_range":{"field":"f","ranges":[{"to":"10.0.0.5"}],"zzz":1}}}}',
    "aggs.unknown_field.auto_date_histogram": '{"aggs":{"a":{"auto_date_histogram":{"field":"d","zzz":1}}}}',
    "aggs.unknown_field.variable_width_histogram": '{"aggs":{"a":{"variable_width_histogram":{"field":"n","zzz":1}}}}',
    "aggs.unknown_field.rare_terms": '{"aggs":{"a":{"rare_terms":{"field":"f","zzz":1}}}}',
    "aggs.unknown_field.multi_terms": '{"aggs":{"a":{"multi_terms":{"terms":[{"field":"f"}],"zzz":1}}}}',
    "aggs.unknown_field.multi_terms_inner": '{"aggs":{"a":{"multi_terms":{"terms":[{"field":"f","zzz":1}]}}}}',
    "aggs.unknown_field.percentile_ranks": '{"aggs":{"a":{"percentile_ranks":{"field":"n","values":[1],"zzz":1}}}}',
    "aggs.unknown_field.median_absolute_deviation": '{"aggs":{"a":{"median_absolute_deviation":{"field":"n","zzz":1}}}}',
    "aggs.unknown_field.weighted_avg": '{"aggs":{"a":{"weighted_avg":{"value":{"field":"n"},"weight":{"field":"n"},"zzz":1}}}}',
    "aggs.unknown_field.weighted_avg_inner": '{"aggs":{"a":{"weighted_avg":{"value":{"field":"n","zzz":1},"weight":{"field":"n"}}}}}',
    "aggs.unknown_field.string_stats": '{"aggs":{"a":{"string_stats":{"field":"f","zzz":1}}}}',
    "aggs.unknown_field.boxplot": '{"aggs":{"a":{"boxplot":{"field":"n","zzz":1}}}}',
    "aggs.unknown_field.t_test": '{"aggs":{"a":{"t_test":{"a":{"field":"n"},"b":{"field":"n"},"zzz":1}}}}',
    "aggs.unknown_field.rate": '{"aggs":{"a":{"date_histogram":{"field":"d","calendar_interval":"day"},"aggs":{"r":{"rate":{"unit":"day","zzz":1}}}}}}',
    "aggs.unknown_field.top_metrics": '{"aggs":{"a":{"top_metrics":{"metrics":{"field":"n"},"sort":"n","zzz":1}}}}',
    "aggs.unknown_field.scripted_metric": '{"aggs":{"a":{"scripted_metric":{"map_script":"1","zzz":1}}}}',
    "aggs.unknown_field.geo_centroid": '{"aggs":{"a":{"geo_centroid":{"field":"g","zzz":1}}}}',
    "aggs.unknown_field.geohash_grid": '{"aggs":{"a":{"geohash_grid":{"field":"g","zzz":1}}}}',
    "aggs.unknown_field.geotile_grid": '{"aggs":{"a":{"geotile_grid":{"field":"g","zzz":1}}}}',
    "aggs.unknown_field.geo_distance": '{"aggs":{"a":{"geo_distance":{"field":"g","origin":"0,0","ranges":[{"to":1}],"zzz":1}}}}',
    "aggs.unknown_field.reverse_nested": '{"aggs":{"a":{"nested":{"path":"o"},"aggs":{"r":{"reverse_nested":{"zzz":1}}}}}}',
    "aggs.unknown_field.children": '{"aggs":{"a":{"children":{"type":"c","zzz":1}}}}',
    "aggs.unknown_field.parent": '{"aggs":{"a":{"parent":{"type":"c","zzz":1}}}}',
    "aggs.unknown_field.diversified_sampler": '{"aggs":{"a":{"diversified_sampler":{"field":"f","zzz":1}}}}',
    "aggs.unknown_field.random_sampler": '{"aggs":{"a":{"random_sampler":{"probability":0.1,"zzz":1}}}}',
    "aggs.unknown_field.significant_text": '{"aggs":{"a":{"significant_text":{"field":"t","zzz":1}}}}',
    "aggs.unknown_field.adjacency_matrix": '{"aggs":{"a":{"adjacency_matrix":{"filters":{"x":{"match_all":{}}},"zzz":1}}}}',
    "aggs.unknown_field.matrix_stats": '{"aggs":{"a":{"matrix_stats":{"fields":["n"],"zzz":1}}}}',
    "aggs.unknown_field.derivative": '{"aggs":{"a":{"histogram":{"field":"n","interval":1},"aggs":{"m":{"avg":{"field":"n"}},"d":{"derivative":{"buckets_path":"m","zzz":1}}}}}}',
    "aggs.unknown_field.cumulative_sum": '{"aggs":{"a":{"histogram":{"field":"n","interval":1},"aggs":{"m":{"avg":{"field":"n"}},"d":{"cumulative_sum":{"buckets_path":"m","zzz":1}}}}}}',
    "aggs.unknown_field.bucket_script": '{"aggs":{"a":{"histogram":{"field":"n","interval":1},"aggs":{"m":{"avg":{"field":"n"}},"d":{"bucket_script":{"buckets_path":{"x":"m"},"script":"1","zzz":1}}}}}}',
    "aggs.unknown_field.bucket_selector": '{"aggs":{"a":{"histogram":{"field":"n","interval":1},"aggs":{"m":{"avg":{"field":"n"}},"d":{"bucket_selector":{"buckets_path":{"x":"m"},"script":"true","zzz":1}}}}}}',
    "aggs.unknown_field.moving_fn": '{"aggs":{"a":{"histogram":{"field":"n","interval":1},"aggs":{"m":{"avg":{"field":"n"}},"d":{"moving_fn":{"buckets_path":"m","window":2,"script":"1","zzz":1}}}}}}',
    "aggs.unknown_field.serial_diff": '{"aggs":{"a":{"histogram":{"field":"n","interval":1},"aggs":{"m":{"avg":{"field":"n"}},"d":{"serial_diff":{"buckets_path":"m","zzz":1}}}}}}',
    "aggs.unknown_field.max_bucket": '{"aggs":{"h":{"terms":{"field":"f"},"aggs":{"m":{"avg":{"field":"n"}}}},"b":{"max_bucket":{"buckets_path":"h>m","zzz":1}}}}',
    "aggs.unknown_field.stats_bucket": '{"aggs":{"h":{"terms":{"field":"f"},"aggs":{"m":{"avg":{"field":"n"}}}},"b":{"stats_bucket":{"buckets_path":"h>m","zzz":1}}}}',
    "aggs.unknown_field.percentiles_bucket": '{"aggs":{"h":{"terms":{"field":"f"},"aggs":{"m":{"avg":{"field":"n"}}}},"b":{"percentiles_bucket":{"buckets_path":"h>m","zzz":1}}}}',
    "aggs.unknown_field.inference": '{"aggs":{"h":{"terms":{"field":"f"},"aggs":{"m":{"avg":{"field":"n"}}}},"b":{"inference":{"model_id":"x","buckets_path":{"a":"h>m"},"zzz":1}}}}',
    "aggs.unknown_field.composite_source": '{"aggs":{"a":{"composite":{"sources":[{"a":{"terms":{"field":"f","zzz":1}}}]}}}}',
    "aggs.unknown_field.composite_source_type": '{"aggs":{"a":{"composite":{"sources":[{"a":{"nope":{"field":"f"}}}]}}}}',
    "aggs.unknown_field.filters_inner_unknown_q": '{"aggs":{"a":{"filters":{"filters":{"x":{"nope":{}}}}}}}',
    "aggs.unknown_field.top_hits_sort_unknown": '{"aggs":{"a":{"top_hits":{"sort":[{"n":{"zzz":1}}]}}}}',
    "aggs.unknown_field.histogram_obj": '{"aggs":{"a":{"histogram":{"field":"n","interval":1,"zzz":{}}}}}',
    "aggs.unknown_field.date_histogram_bad_interval": '{"aggs":{"a":{"date_histogram":{"field":"d","calendar_interval":"5x"}}}}',
    "aggs.unknown_field.terms_bad_exec_hint": '{"aggs":{"a":{"terms":{"field":"f","execution_hint":"nope"}}}}',
    "aggs.unknown_field.cardinality_precision_str": '{"aggs":{"a":{"cardinality":{"field":"f","precision_threshold":"x"}}}}',
    "aggs.unknown_field.percentiles_percents_str": '{"aggs":{"a":{"percentiles":{"field":"n","percents":"x"}}}}',
    "aggs.unknown_field.range_ranges_str": '{"aggs":{"a":{"range":{"field":"n","ranges":"x"}}}}',
    "aggs.unknown_field.missing_obj": '{"aggs":{"a":{"missing":{"field":"f","zzz":{}}}}}',
    "aggs.unknown_field.global_obj": '{"aggs":{"a":{"global":{"zzz":{}}}}}',
    "aggs.unknown_field.global_str": '{"aggs":{"a":{"global":{"zzz":"x"}}}}',
    "aggs.unknown_field.sampler_obj": '{"aggs":{"a":{"sampler":{"zzz":{}}}}}',
    "aggs.unknown_field.nested_obj": '{"aggs":{"a":{"nested":{"path":"o","zzz":{}}}}}',
    "aggs.unknown_field.nested_str": '{"aggs":{"a":{"nested":{"path":"o","zzz":"x"}}}}',
    "aggs.unknown_field.top_hits_obj": '{"aggs":{"a":{"top_hits":{"size":1,"zzz":{}}}}}',
    "aggs.unknown_field.filters_obj": '{"aggs":{"a":{"filters":{"filters":{"a":{"match_all":{}}},"zzz":{}}}}}',
    "aggs.unknown_field.filters_str": '{"aggs":{"a":{"filters":{"filters":{"a":{"match_all":{}}},"zzz":"x"}}}}',
    "aggs.sub_aggs_not_obj": '{"aggs":{"a":{"terms":{"field":"f"},"aggs":5}}}',
    "aggs.sub_aggs_both": '{"aggs":{"a":{"terms":{"field":"f"},"aggs":{},"aggregations":{}}}}',
    "aggs.meta_not_obj": '{"aggs":{"a":{"terms":{"field":"f"},"meta":5}}}',
    "aggs.empty_name": '{"aggs":{"":{"avg":{"field":"n"}}}}',
    "sort.obj_two_fields": '{"sort":[{"n":{"order":"asc"},"f":{"order":"asc"}}]}',
    "sort.unknown_obj": '{"sort":[{"n":{"zzz":{}}}]}',
    "sort.mode_bad": '{"sort":[{"n":{"mode":"nope"}}]}',
    "sort.missing_obj": '{"sort":[{"n":{"missing":{}}}]}',
    "sort.unmapped_type_num": '{"sort":[{"n":{"unmapped_type":5}}]}',
    "sort.numeric_type_bad": '{"sort":[{"n":{"numeric_type":"nope"}}]}',
    "sort.score_unknown": '{"sort":[{"_score":{"zzz":1}}]}',
    "sort.doc_unknown": '{"sort":[{"_doc":{"zzz":1}}]}',
    "sort.geo_unit_bad": '{"sort":[{"_geo_distance":{"g":[0,0],"unit":"nope"}}]}',
    "sort.geo_order_bad": '{"sort":[{"_geo_distance":{"g":[0,0],"order":"nope"}}]}',
    "sort.script_no_type": '{"sort":{"_script":{"script":"1"}}}',
    "sort.script_type_bad": '{"sort":{"_script":{"type":"nope","script":"1"}}}',
    "sort.arr_in_arr": '{"sort":[[1]]}',
    "sort.nested_not_obj": '{"sort":[{"n":{"nested":5}}]}',
    "sort.nested_filter_unknown_q": '{"sort":[{"n":{"nested":{"path":"o","filter":{"nope":{}}}}}]}',
    "highlight.fields_arr_unknown": '{"highlight":{"fields":[{"f":{"zzz":1}}]}}',
    "highlight.order_bad": '{"highlight":{"order":"nope","fields":{"f":{}}}}',
    "highlight.encoder_bad": '{"highlight":{"encoder":"nope","fields":{"f":{}}}}',
    "highlight.fragment_size_str": '{"highlight":{"fragment_size":"x","fields":{"f":{}}}}',
    "highlight.pre_tags_num": '{"highlight":{"pre_tags":5,"fields":{"f":{}}}}',
    "highlight.highlight_query_unknown": '{"highlight":{"highlight_query":{"nope":{}},"fields":{"f":{}}}}',
    "highlight.field_highlight_query_unknown": '{"highlight":{"fields":{"f":{"highlight_query":{"nope":{}}}}}}',
    "highlight.field_obj_unknown": '{"highlight":{"fields":{"f":{"zzz":{}}}}}',
    "highlight.fields_obj_not_obj": '{"highlight":{"fields":{"f":5}}}',
    "highlight.unknown_obj": '{"highlight":{"fields":{"f":{}},"zzz":{}}}',
    "suggest.term_unknown_obj": '{"suggest":{"s":{"text":"x","term":{"field":"f","zzz":{}}}}}',
    "suggest.phrase_unknown_obj": '{"suggest":{"s":{"text":"x","phrase":{"field":"f","zzz":{}}}}}',
    "suggest.phrase_collate_unknown": '{"suggest":{"s":{"text":"x","phrase":{"field":"f","collate":{"zzz":1}}}}}',
    "suggest.phrase_smoothing_unknown": '{"suggest":{"s":{"text":"x","phrase":{"field":"f","smoothing":{"nope":{}}}}}}',
    "suggest.phrase_generator_unknown": '{"suggest":{"s":{"text":"x","phrase":{"field":"f","direct_generator":[{"field":"f","zzz":1}]}}}}',
    "suggest.completion_contexts_bad": '{"suggest":{"s":{"prefix":"x","completion":{"field":"f","contexts":5}}}}',
    "suggest.completion_fuzzy_unknown": '{"suggest":{"s":{"prefix":"x","completion":{"field":"f","fuzzy":{"zzz":1}}}}}',
    "suggest.top_unknown": '{"suggest":{"s":{"zzz":"x","term":{"field":"f"}}}}',
    "suggest.two_suggesters": '{"suggest":{"s":{"text":"x","term":{"field":"f"},"phrase":{"field":"f"}}}}',
    "suggest.text_num": '{"suggest":{"s":{"text":5,"term":{"field":"f"}}}}',
    "suggest.global_text_num": '{"suggest":{"text":5,"s":{"term":{"field":"f"}}}}',
    "suggest.suggester_not_obj": '{"suggest":{"s":{"text":"x","term":5}}}',
    "knn.k_obj": '{"knn":{"field":"v","query_vector":[1,1],"k":{},"num_candidates":1}}',
    "knn.similarity_str": '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"similarity":"x"}}',
    "knn.vector_obj": '{"knn":{"field":"v","query_vector":{},"k":1,"num_candidates":1}}',
    "knn.vector_arr_str": '{"knn":{"field":"v","query_vector":["a"],"k":1,"num_candidates":1}}',
    "knn.boost_str": '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"boost":"x"}}',
    "knn.qvb_unknown": '{"knn":{"field":"v","query_vector_builder":{"nope":{}},"k":1,"num_candidates":1}}',
    "knn.qvb_text_embedding_unknown": '{"knn":{"field":"v","query_vector_builder":{"text_embedding":{"model_id":"m","model_text":"x","zzz":1}},"k":1,"num_candidates":1}}',
    "knn.inner_hits_unknown": '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"inner_hits":{"zzz":1}}}',
    "knn.arr_unknown": '{"knn":[{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"zzz":1}]}',
    "knn.arr_num": '{"knn":[5]}',
    "knn.no_vector": '{"knn":{"field":"v","k":1,"num_candidates":1}}',
    "knn.rescore_vector_unknown": '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"rescore_vector":{"zzz":1}}}',
    "retriever.standard_filter_unknown": '{"retriever":{"standard":{"filter":{"nope":{}}}}}',
    "retriever.standard_sort_unknown": '{"retriever":{"standard":{"sort":[{"n":{"zzz":1}}]}}}',
    "retriever.standard_collapse_unknown": '{"retriever":{"standard":{"collapse":{"field":"f","zzz":1}}}}',
    "retriever.knn_k_str": '{"retriever":{"knn":{"field":"v","query_vector":[1,1],"k":"x","num_candidates":1}}}',
    "retriever.rrf_rank_window_str": '{"retriever":{"rrf":{"retrievers":[],"rank_window_size":"x"}}}',
    "retriever.rrf_child_unknown_type": '{"retriever":{"rrf":{"retrievers":[{"nope":{}}]}}}',
    "retriever.rrf_child_num": '{"retriever":{"rrf":{"retrievers":[5]}}}',
    "retriever.text_similarity_unknown": '{"retriever":{"text_similarity_reranker":{"retriever":{"standard":{}},"field":"f","inference_text":"x","zzz":1}}}',
    "retriever.rule_unknown": '{"retriever":{"rule":{"retriever":{"standard":{}},"ruleset_ids":["a"],"match_criteria":{},"zzz":1}}}',
    "retriever.linear_unknown": '{"retriever":{"linear":{"retrievers":[],"zzz":1}}}',
    "retriever.rescorer_unknown": '{"retriever":{"rescorer":{"retriever":{"standard":{}},"rescore":{"query":{"rescore_query":{"match_all":{}}}},"zzz":1}}}',
    "retriever.pinned_unknown": '{"retriever":{"pinned":{"retriever":{"standard":{}},"ids":["1"],"zzz":1}}}',
    "retriever.standard_unknown_obj": '{"retriever":{"standard":{"zzz":{}}}}',
    "retriever.min_score_str": '{"retriever":{"standard":{"min_score":"x"}}}',
    "retriever.name_num": '{"retriever":{"standard":{"_name":5}}}',
    "retriever.with_query": '{"retriever":{"standard":{}},"query":{"match_all":{}}}',
    "retriever.with_knn": '{"retriever":{"standard":{}},"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1}}',
    "collapse.inner_hits_unknown": '{"collapse":{"field":"f","inner_hits":{"zzz":1}}}',
    "collapse.inner_hits_arr_unknown": '{"collapse":{"field":"f","inner_hits":[{"zzz":1}]}}',
    "collapse.no_field": '{"collapse":{"inner_hits":{"name":"x"}}}',
    "collapse.field_num": '{"collapse":{"field":5}}',
    "collapse.max_concurrent_str": '{"collapse":{"field":"f","max_concurrent_group_searches":"x"}}',
    "collapse.collapse_obj": '{"collapse":{"field":"f","collapse":{"field":"n"}}}',
    "rescore.arr_unknown": '{"rescore":[{"window_size":10,"zzz":1}]}',
    "rescore.unknown_type": '{"rescore":{"nope":{}}}',
    "rescore.query_weight_str": '{"rescore":{"query":{"rescore_query":{"match_all":{}},"query_weight":"x"}}}',
    "rescore.score_mode_bad": '{"rescore":{"query":{"rescore_query":{"match_all":{}},"score_mode":"nope"}}}',
    "rescore.window_size_str": '{"rescore":{"window_size":"x","query":{"rescore_query":{"match_all":{}}}}}',
    "rescore.no_query": '{"rescore":{"window_size":10}}',
    "rescore.learning_to_rank_unknown": '{"rescore":{"learning_to_rank":{"model_id":"m","zzz":1}}}',
    "rescore.query_obj_unknown": '{"rescore":{"query":{"rescore_query":{"match_all":{}},"zzz":{}}}}',
    "rescore.num": '{"rescore":5}',
    "rescore.unknown_obj": '{"rescore":{"window_size":10,"zzz":{}}}',
}
for k, b in H.items():
    add(k, "POST", S, b)
# positions in re-serialised bodies (by_query, count, explain) and multiline / spaced variants
add("body.dbq_unknown_key_2", "POST", "/probe/_delete_by_query", '{"zz":1}')
add("body.dbq_unknown_key_3", "POST", "/probe/_delete_by_query", '{"query":{"match_all":{}},"zzzzzzzz":1}')
add("body.dbq_unknown_key_4", "POST", "/probe/_delete_by_query", '{"query":{"match_all":{}},   "zz":1}')
add("body.dbq_unknown_key_5", "POST", "/probe/_delete_by_query", '{"query": {"match_all": {}}, "zz": {"a": 1}}')
add("body.dbq_unknown_key_6", "POST", "/probe/_delete_by_query", '{\n  "query": {"match_all": {}},\n  "zz": 1\n}')
add("body.dbq_query_unknown_2", "POST", "/probe/_delete_by_query", '{"query": {"match_all": {"zz": 1}}}')
add("body.dbq_unknown_q", "POST", "/probe/_delete_by_query", '{"query":{"nope":{}}}')
add("body.dbq_size_and_max_docs", "POST", "/probe/_delete_by_query", '{"query":{"match_all":{}},"size":1,"max_docs":1}')
add("body.dbq_syntax", "POST", "/probe/_delete_by_query", '{"query":}')
add("body.ubq_unknown_key", "POST", "/probe/_update_by_query", '{"query":{"match_all":{}},"zz":1}')
add("body.ubq_script_bad", "POST", "/probe/_update_by_query", '{"script":5}')
add("body.reindex_unknown_key", "POST", "/_reindex", '{"source":{"index":"probe"},"dest":{"index":"probe-copy"},"zz":1}')
add("body.reindex_source_unknown", "POST", "/_reindex", '{"source":{"index":"probe","zz":1},"dest":{"index":"probe-copy"}}')
add("body.reindex_query_unknown", "POST", "/_reindex", '{"source":{"index":"probe","query":{"match_all":{"zz":1}}},"dest":{"index":"probe-copy"}}')
add("body.count_unknown_spaced", "POST", "/probe/_count", '{"query": {"match_all": {}}, "size": 1}')
add("body.count_multiline", "POST", "/probe/_count", '{\n  "query": {"match_all": {}},\n  "size": 1\n}')
add("body.count_query_unknown_q", "POST", "/probe/_count", '{"query":{"nope":{}}}')
add("body.count_syntax", "POST", "/probe/_count", '{"query":}')
add("body.explain_unknown_key", "POST", "/probe/_explain/1", '{"query":{"match_all":{}},"zz":1}')
add("body.explain_query_unknown_field", "POST", "/probe/_explain/1", '{"query":{"match_all":{"zz":1}}}')
add("body.explain_syntax", "POST", "/probe/_explain/1", '{"query":}')
add("body.validate_unknown_q", "POST", "/probe/_validate/query", '{"query":{"nope":{}}}')
add("body.validate_syntax", "POST", "/probe/_validate/query", '{"query":}')
add("body.msearch_unknown", "POST", "/probe/_msearch", '{}\n{"query":{"match_all":{"zz":1}}}\n')
add("body.msearch_unknown_key", "POST", "/probe/_msearch", '{}\n{"zz":1}\n')
add("body.msearch_header_unknown", "POST", "/probe/_msearch", '{"zz":1}\n{"query":{"match_all":{}}}\n')
add("body.msearch_syntax", "POST", "/probe/_msearch", '{}\n{"query":}\n')
add("body.async_submit_unknown_key", "POST", "/probe/_async_search", '{"zz":1}')
add("body.async_submit_unknown_q", "POST", "/probe/_async_search", '{"query":{"nope":{}}}')
add("body.search_multiline_unknown_key", "POST", S, '{\n  "query": {"match_all": {}},\n  "zz": 1\n}')
add("body.search_spaced_unknown_q", "POST", S, '{"query": {"nope": {}}}')
add("body.search_spaced_unknown_field", "POST", S, '{"query": {"match_all": {"foo": 1}}}')
add("body.search_source_param_syntax", "GET", "/probe/_search?source=%7B%22size%22%3A%7D&source_content_type=application/json", None)
add("body.search_source_param_unknown", "GET", "/probe/_search?source=%7B%22zz%22%3A1%7D&source_content_type=application/json", None)
add("body.search_tsv_unknown_nested", "POST", S, '{"query":{"bool":{"must":{"bool":{"filter":{"term":{"f":{"value":"x","zz":1}}}}}}}}')
add("body.search_term_boost_str", "POST", S, '{"query":{"term":{"f":{"value":"x","boost":"abc"}}}}')
add("body.search_match_boost_str", "POST", S, '{"query":{"match":{"f":{"query":"x","boost":"abc"}}}}')
add("body.search_match_operator_bad", "POST", S, '{"query":{"match":{"f":{"query":"x","operator":"nope"}}}}')
add("body.search_match_fuzziness_bad", "POST", S, '{"query":{"match":{"f":{"query":"x","fuzziness":"nope"}}}}')
add("body.search_range_relation_bad", "POST", S, '{"query":{"range":{"n":{"gte":1,"relation":"nope"}}}}')
add("body.search_bool_msm_obj", "POST", S, '{"query":{"bool":{"minimum_should_match":{}}}}')
add("body.search_bool_must_num", "POST", S, '{"query":{"bool":{"must":5}}}')
add("body.search_bool_must_arr_num", "POST", S, '{"query":{"bool":{"must":[5]}}}')
add("body.search_bool_must_arr_arr", "POST", S, '{"query":{"bool":{"must":[[]]}}}')
add("body.search_terms_two_fields", "POST", S, '{"query":{"terms":{"f":["x"],"t":["y"]}}}')
add("body.search_terms_no_field", "POST", S, '{"query":{"terms":{}}}')
add("body.search_term_no_field", "POST", S, '{"query":{"term":{}}}')
add("body.search_match_no_field", "POST", S, '{"query":{"match":{}}}')
add("body.search_range_no_field", "POST", S, '{"query":{"range":{}}}')
add("body.search_prefix_no_field", "POST", S, '{"query":{"prefix":{}}}')
add("body.search_ids_values_obj", "POST", S, '{"query":{"ids":{"values":{}}}}')
add("body.search_ids_values_arr_obj", "POST", S, '{"query":{"ids":{"values":[{}]}}}')
add("body.search_nested_no_path", "POST", S, '{"query":{"nested":{"query":{"match_all":{}}}}}')
add("body.search_nested_no_query", "POST", S, '{"query":{"nested":{"path":"o"}}}')
add("body.search_nested_score_mode_bad", "POST", S, '{"query":{"nested":{"path":"o","query":{"match_all":{}},"score_mode":"nope"}}}')
add("body.search_dis_max_queries_obj", "POST", S, '{"query":{"dis_max":{"queries":{}}}}')
add("body.search_dis_max_tie_str", "POST", S, '{"query":{"dis_max":{"queries":[],"tie_breaker":"x"}}}')
add("body.search_constant_score_no_filter", "POST", S, '{"query":{"constant_score":{}}}')
add("body.search_constant_score_two_filters", "POST", S, '{"query":{"constant_score":{"filter":{"match_all":{}},"filter2":{"match_all":{}}}}}')
add("body.search_boosting_no_positive", "POST", S, '{"query":{"boosting":{"negative":{"match_all":{}},"negative_boost":0.5}}}')
add("body.search_boosting_no_negative_boost", "POST", S, '{"query":{"boosting":{"positive":{"match_all":{}},"negative":{"match_all":{}}}}}')
add("body.search_function_score_boost_mode_bad", "POST", S, '{"query":{"function_score":{"query":{"match_all":{}},"boost_mode":"nope"}}}')
add("body.search_function_score_two_fns", "POST", S, '{"query":{"function_score":{"query":{"match_all":{}},"functions":[{"weight":1,"random_score":{},"script_score":{"script":"1"}}]}}}')
add("body.search_function_score_fn_filter_unknown", "POST", S, '{"query":{"function_score":{"functions":[{"filter":{"nope":{}},"weight":1}]}}}')
add("body.search_function_score_gauss_unknown", "POST", S, '{"query":{"function_score":{"functions":[{"gauss":{"n":{"origin":1,"scale":1,"zzz":1}}}]}}}')
add("body.search_function_score_random_unknown", "POST", S, '{"query":{"function_score":{"functions":[{"random_score":{"zzz":1}}]}}}')
add("body.search_function_score_fvf_unknown", "POST", S, '{"query":{"function_score":{"functions":[{"field_value_factor":{"field":"n","zzz":1}}]}}}')
add("body.search_function_score_script_score_unknown", "POST", S, '{"query":{"function_score":{"functions":[{"script_score":{"script":"1","zzz":1}}]}}}')
add("body.search_multi_match_fields_num", "POST", S, '{"query":{"multi_match":{"query":"x","fields":5}}}')
add("body.search_multi_match_no_query", "POST", S, '{"query":{"multi_match":{"fields":["f"]}}}')
add("body.search_query_string_no_query", "POST", S, '{"query":{"query_string":{"fields":["f"]}}}')
add("body.search_query_string_bad_syntax", "POST", S, '{"query":{"query_string":{"query":"f:(x"}}}')
add("body.search_sqs_no_query", "POST", S, '{"query":{"simple_query_string":{"fields":["f"]}}}')
add("body.search_sqs_flags_bad", "POST", S, '{"query":{"simple_query_string":{"query":"x","flags":"NOPE"}}}')
add("body.search_wildcard_no_value", "POST", S, '{"query":{"wildcard":{"f":{"boost":1}}}}')
add("body.search_regexp_flags_bad", "POST", S, '{"query":{"regexp":{"f":{"value":"x","flags":"NOPE"}}}}')
add("body.search_fuzzy_no_value", "POST", S, '{"query":{"fuzzy":{"f":{"boost":1}}}}')
add("body.search_fuzzy_fuzziness_bad", "POST", S, '{"query":{"fuzzy":{"f":{"value":"x","fuzziness":"nope"}}}}')
add("body.search_script_no_script", "POST", S, '{"query":{"script":{}}}')
add("body.search_script_script_num", "POST", S, '{"query":{"script":{"script":5}}}')
add("body.search_script_script_obj_unknown", "POST", S, '{"query":{"script":{"script":{"source":"true","zzz":1}}}}')
add("body.search_mlt_no_like", "POST", S, '{"query":{"more_like_this":{"fields":["t"]}}}')
add("body.search_mlt_like_obj_unknown", "POST", S, '{"query":{"more_like_this":{"like":[{"_index":"probe","_id":"1","zzz":1}]}}}')
add("body.search_span_near_no_clauses", "POST", S, '{"query":{"span_near":{"slop":1}}}')
add("body.search_span_near_clause_not_span", "POST", S, '{"query":{"span_near":{"clauses":[{"term":{"f":"x"}}]}}}')
add("body.search_span_term_obj_unknown", "POST", S, '{"query":{"span_term":{"f":{"value":"x","zzz":1}}}}')
add("body.search_geo_distance_no_distance", "POST", S, '{"query":{"geo_distance":{"g":[0,0]}}}')
add("body.search_geo_distance_bad_distance", "POST", S, '{"query":{"geo_distance":{"distance":"x","g":[0,0]}}}')
add("body.search_geo_bbox_unknown_corner", "POST", S, '{"query":{"geo_bounding_box":{"g":{"top_left":[0,1],"bottom_right":[1,0],"zzz":1}}}}')
add("body.search_geo_bbox_no_field", "POST", S, '{"query":{"geo_bounding_box":{}}}')
add("body.search_knn_query_no_vector", "POST", S, '{"query":{"knn":{"field":"v"}}}')
add("body.search_knn_query_k_str", "POST", S, '{"query":{"knn":{"field":"v","query_vector":[1,1],"k":"x"}}}')
add("body.search_knn_query_filter_unknown", "POST", S, '{"query":{"knn":{"field":"v","query_vector":[1,1],"filter":{"nope":{}}}}}')
add("body.search_pinned_ids_num", "POST", S, '{"query":{"pinned":{"ids":5,"organic":{"match_all":{}}}}}')
add("body.search_pinned_no_organic", "POST", S, '{"query":{"pinned":{"ids":["1"]}}}')
add("body.search_pinned_organic_unknown", "POST", S, '{"query":{"pinned":{"ids":["1"],"organic":{"nope":{}}}}}')
add("body.search_terms_set_no_terms", "POST", S, '{"query":{"terms_set":{"f":{}}}}')
add("body.search_rank_feature_no_field", "POST", S, '{"query":{"rank_feature":{}}}')
add("body.search_rank_feature_two_fns", "POST", S, '{"query":{"rank_feature":{"field":"n","saturation":{},"log":{"scaling_factor":1}}}}')
add("body.search_rank_feature_saturation_unknown", "POST", S, '{"query":{"rank_feature":{"field":"n","saturation":{"zzz":1}}}}')
add("body.search_percolate_no_field", "POST", S, '{"query":{"percolate":{"document":{}}}}')
add("body.search_percolate_doc_num", "POST", S, '{"query":{"percolate":{"field":"p","document":5}}}')
add("body.search_sparse_vector_no_field", "POST", S, '{"query":{"sparse_vector":{"query_vector":{"a":1}}}}')
add("body.search_sparse_vector_qv_num", "POST", S, '{"query":{"sparse_vector":{"field":"f","query_vector":5}}}')
add("body.search_sparse_vector_prune_str", "POST", S, '{"query":{"sparse_vector":{"field":"f","query_vector":{"a":1},"prune":"x"}}}')
add("body.search_exists_field_num", "POST", S, '{"query":{"exists":{"field":5}}}')
add("body.search_exists_field_arr", "POST", S, '{"query":{"exists":{"field":["f"]}}}')
add("body.search_ids_unknown_obj", "POST", S, '{"query":{"ids":{"values":["1"],"zzz":{}}}}')
add("body.search_match_all_unknown_obj", "POST", S, '{"query":{"match_all":{"zzz":{}}}}')
add("body.search_match_all_unknown_arr", "POST", S, '{"query":{"match_all":{"zzz":[]}}}')
add("body.search_match_all_unknown_str", "POST", S, '{"query":{"match_all":{"zzz":"x"}}}')
add("body.search_match_all_name_num", "POST", S, '{"query":{"match_all":{"_name":5}}}')
add("body.search_match_none_boost_str", "POST", S, '{"query":{"match_none":{"boost":"x"}}}')
add("body.search_term_value_obj", "POST", S, '{"query":{"term":{"f":{"value":{}}}}}')
add("body.search_term_no_value", "POST", S, '{"query":{"term":{"f":{"boost":1}}}}')
add("body.search_term_case_insensitive_str", "POST", S, '{"query":{"term":{"f":{"value":"x","case_insensitive":"x"}}}}')
add("body.search_range_gte_obj", "POST", S, '{"query":{"range":{"n":{"gte":{}}}}}')
add("body.search_range_boost_str", "POST", S, '{"query":{"range":{"n":{"gte":1,"boost":"x"}}}}')
add("body.search_range_format_num", "POST", S, '{"query":{"range":{"d":{"gte":1,"format":5}}}}')
add("body.search_match_phrase_slop_str", "POST", S, '{"query":{"match_phrase":{"f":{"query":"x","slop":"x"}}}}')
add("body.search_match_zero_terms_bad", "POST", S, '{"query":{"match":{"f":{"query":"x","zero_terms_query":"nope"}}}}')
add("body.search_match_prefix_length_str", "POST", S, '{"query":{"match":{"f":{"query":"x","prefix_length":"x"}}}}')
add("body.search_match_lenient_str", "POST", S, '{"query":{"match":{"f":{"query":"x","lenient":"x"}}}}')
add("body.search_match_query_obj", "POST", S, '{"query":{"match":{"f":{"query":{}}}}}')
add("body.search_match_query_arr", "POST", S, '{"query":{"match":{"f":{"query":[]}}}}')
add("body.search_match_value_arr", "POST", S, '{"query":{"match":{"f":["x"]}}}')
add("body.search_bool_should_obj_unknown_field", "POST", S, '{"query":{"bool":{"should":{"match_all":{"zz":1}}}}}')
add("body.search_bool_adjust_str", "POST", S, '{"query":{"bool":{"adjust_pure_negative":"x"}}}')
add("body.search_bool_name_num", "POST", S, '{"query":{"bool":{"_name":5}}}')
add("body.search_bool_two_unknown", "POST", S, '{"query":{"bool":{"zz":1,"yy":2}}}')
add("body.search_range_two_fields", "POST", S, '{"query":{"range":{"n":{"gte":1},"d":{"gte":1}}}}')
add("body.search_prefix_two_fields", "POST", S, '{"query":{"prefix":{"f":"x","t":"y"}}}')
add("body.search_match_phrase_two_fields", "POST", S, '{"query":{"match_phrase":{"f":"x","t":"y"}}}')
add("body.search_multi_match_two_fields", "POST", S, '{"query":{"multi_match":{"query":"x","fields":["f","f"]}}}')
add("body.search_prefix_value_num", "POST", S, '{"query":{"prefix":{"f":5}}}')
add("body.search_wildcard_value_obj", "POST", S, '{"query":{"wildcard":{"f":{"wildcard":{}}}}}')
add("body.search_regexp_max_states_str", "POST", S, '{"query":{"regexp":{"f":{"value":"x","max_determinized_states":"x"}}}}')
add("body.search_query_string_boost_str", "POST", S, '{"query":{"query_string":{"query":"x","boost":"x"}}}')
add("body.search_query_string_fields_num", "POST", S, '{"query":{"query_string":{"query":"x","fields":5}}}')
add("body.search_query_string_fields_arr_num", "POST", S, '{"query":{"query_string":{"query":"x","fields":[5]}}}')
add("body.search_query_string_query_num", "POST", S, '{"query":{"query_string":{"query":5}}}')
add("body.search_query_string_type_bad", "POST", S, '{"query":{"query_string":{"query":"x","type":"nope"}}}')
add("body.search_query_string_rewrite_bad", "POST", S, '{"query":{"query_string":{"query":"x","rewrite":"nope"}}}')
add("body.search_query_string_fuzzy_rewrite_bad", "POST", S, '{"query":{"query_string":{"query":"x","fuzzy_rewrite":"nope"}}}')
add("body.search_query_string_tz_bad", "POST", S, '{"query":{"query_string":{"query":"x","time_zone":"nope"}}}')
add("body.search_query_string_analyzer_bad", "POST", S, '{"query":{"query_string":{"query":"x","analyzer":"nope"}}}')
add("body.search_match_analyzer_bad", "POST", S, '{"query":{"match":{"f":{"query":"x","analyzer":"nope"}}}}')
add("body.search_match_minimum_should_match_obj", "POST", S, '{"query":{"match":{"f":{"query":"x","minimum_should_match":{}}}}}')
add("body.search_multi_match_type_num", "POST", S, '{"query":{"multi_match":{"query":"x","type":5}}}')
add("body.search_multi_match_tie_str", "POST", S, '{"query":{"multi_match":{"query":"x","tie_breaker":"x"}}}')
add("body.search_multi_match_operator_bad", "POST", S, '{"query":{"multi_match":{"query":"x","operator":"nope"}}}')
add("body.search_multi_match_query_obj", "POST", S, '{"query":{"multi_match":{"query":{}}}}')
add("body.search_terms_boost_str", "POST", S, '{"query":{"terms":{"f":["x"],"boost":"x"}}}')
add("body.search_terms_lookup_no_index", "POST", S, '{"query":{"terms":{"f":{"id":"1","path":"f"}}}}')
add("body.search_terms_lookup_unknown", "POST", S, '{"query":{"terms":{"f":{"index":"probe","id":"1","path":"f","zzz":1}}}}')
add("body.search_terms_arr_obj", "POST", S, '{"query":{"terms":{"f":[{}]}}}')
add("body.search_terms_str", "POST", S, '{"query":{"terms":{"f":"x"}}}')
add("body.search_nested_inner_hits_unknown", "POST", S, '{"query":{"nested":{"path":"o","query":{"match_all":{}},"inner_hits":{"zzz":1}}}}')
add("body.search_nested_ignore_unmapped_str", "POST", S, '{"query":{"nested":{"path":"o","query":{"match_all":{}},"ignore_unmapped":"x"}}}')
add("body.search_dis_max_boost_str", "POST", S, '{"query":{"dis_max":{"queries":[],"boost":"x"}}}')
add("body.search_dis_max_queries_arr_num", "POST", S, '{"query":{"dis_max":{"queries":[5]}}}')
add("body.search_boosting_negative_boost_str", "POST", S, '{"query":{"boosting":{"positive":{"match_all":{}},"negative":{"match_all":{}},"negative_boost":"x"}}}')
add("body.search_boosting_negative_boost_neg", "POST", S, '{"query":{"boosting":{"positive":{"match_all":{}},"negative":{"match_all":{}},"negative_boost":-1}}}')
add("body.search_constant_score_boost_str", "POST", S, '{"query":{"constant_score":{"filter":{"match_all":{}},"boost":"x"}}}')
add("body.search_constant_score_filter_num", "POST", S, '{"query":{"constant_score":{"filter":5}}}')
add("body.search_function_score_functions_obj", "POST", S, '{"query":{"function_score":{"functions":{}}}}')
add("body.search_function_score_fn_num", "POST", S, '{"query":{"function_score":{"functions":[5]}}}')
add("body.search_function_score_weight_str", "POST", S, '{"query":{"function_score":{"functions":[{"weight":"x"}]}}}')
add("body.search_function_score_max_boost_str", "POST", S, '{"query":{"function_score":{"query":{"match_all":{}},"max_boost":"x"}}}')
add("body.search_function_score_min_score_str", "POST", S, '{"query":{"function_score":{"query":{"match_all":{}},"min_score":"x"}}}')
add("body.search_function_score_score_mode_bad", "POST", S, '{"query":{"function_score":{"query":{"match_all":{}},"score_mode":"nope"}}}')
add("body.search_function_score_query_num", "POST", S, '{"query":{"function_score":{"query":5}}}')
add("body.search_function_score_weight_and_fns", "POST", S, '{"query":{"function_score":{"query":{"match_all":{}},"weight":1,"functions":[{"weight":1}]}}}')
add("body.search_script_score_no_script", "POST", S, '{"query":{"script_score":{"query":{"match_all":{}}}}}')
add("body.search_script_score_min_score_str", "POST", S, '{"query":{"script_score":{"query":{"match_all":{}},"script":"1","min_score":"x"}}}')
add("body.search_wrapper_bad", "POST", S, '{"query":{"wrapper":{"query":"notb64!"}}}')
add("body.search_wrapper_no_query", "POST", S, '{"query":{"wrapper":{}}}')
add("body.search_wrapper_inner_unknown", "POST", S, '{"query":{"wrapper":{"query":"eyJub3BlIjp7fX0="}}}')
add("body.search_ids_values_num", "POST", S, '{"query":{"ids":{"values":[5]}}}')
add("body.search_span_multi_not_multi", "POST", S, '{"query":{"span_multi":{"match":{"term":{"f":"x"}}}}}')
add("body.search_span_first_no_end", "POST", S, '{"query":{"span_first":{"match":{"span_term":{"f":"x"}}}}}')
add("body.search_span_or_clauses_obj", "POST", S, '{"query":{"span_or":{"clauses":{}}}}')
add("body.search_intervals_unknown_rule", "POST", S, '{"query":{"intervals":{"t":{"nope":{}}}}}')
add("body.search_intervals_match_unknown", "POST", S, '{"query":{"intervals":{"t":{"match":{"query":"x","zzz":1}}}}}')
add("body.search_intervals_all_of_unknown", "POST", S, '{"query":{"intervals":{"t":{"all_of":{"intervals":[],"zzz":1}}}}}')
add("body.search_intervals_filter_unknown", "POST", S, '{"query":{"intervals":{"t":{"match":{"query":"x","filter":{"zzz":{}}}}}}}')
add("body.search_distance_feature_no_origin", "POST", S, '{"query":{"distance_feature":{"field":"d","pivot":"1d"}}}')
add("body.search_geo_shape_relation_bad", "POST", S, '{"query":{"geo_shape":{"g":{"shape":{"type":"point","coordinates":[0,0]},"relation":"nope"}}}}')
add("body.search_geo_shape_no_shape", "POST", S, '{"query":{"geo_shape":{"g":{}}}}')
add("body.search_geo_shape_indexed_unknown", "POST", S, '{"query":{"geo_shape":{"g":{"indexed_shape":{"index":"probe","id":"1","zzz":1}}}}}')
add("body.search_has_child_no_type", "POST", S, '{"query":{"has_child":{"query":{"match_all":{}}}}}')
add("body.search_has_child_score_mode_bad", "POST", S, '{"query":{"has_child":{"type":"c","query":{"match_all":{}},"score_mode":"nope"}}}')
add("body.search_has_parent_no_type", "POST", S, '{"query":{"has_parent":{"query":{"match_all":{}}}}}')
add("body.search_parent_id_no_type", "POST", S, '{"query":{"parent_id":{"id":"1"}}}')
add("body.search_percolate_unknown_doc_type", "POST", S, '{"query":{"percolate":{"field":"p","document":{},"document_type":"x","zzz":1}}}')
add("body.search_semantic_no_query", "POST", S, '{"query":{"semantic":{"field":"f"}}}')
add("body.search_text_expansion_no_model", "POST", S, '{"query":{"text_expansion":{"f":{"model_text":"x"}}}}')
add("body.search_rule_no_organic", "POST", S, '{"query":{"rule":{"ruleset_ids":["a"],"match_criteria":{}}}}')
add("body.search_combined_fields_no_fields", "POST", S, '{"query":{"combined_fields":{"query":"x"}}}')
add("body.search_combined_fields_operator_bad", "POST", S, '{"query":{"combined_fields":{"query":"x","fields":["f"],"operator":"nope"}}}')
add("body.search_combined_fields_fields_num", "POST", S, '{"query":{"combined_fields":{"query":"x","fields":5}}}')
add("body.search_match_bool_prefix_slop", "POST", S, '{"query":{"match_bool_prefix":{"f":{"query":"x","slop":1}}}}')
add("body.search_sqs_query_num", "POST", S, '{"query":{"simple_query_string":{"query":5}}}')
add("body.search_sqs_fields_obj", "POST", S, '{"query":{"simple_query_string":{"query":"x","fields":{}}}}')
add("body.search_sqs_default_operator_bad", "POST", S, '{"query":{"simple_query_string":{"query":"x","default_operator":"nope"}}}')
add("body.search_exists_boost_str", "POST", S, '{"query":{"exists":{"field":"f","boost":"x"}}}')
add("body.search_ids_boost_str", "POST", S, '{"query":{"ids":{"values":["1"],"boost":"x"}}}')
add("body.search_knn_query_num_cands_str", "POST", S, '{"query":{"knn":{"field":"v","query_vector":[1,1],"num_candidates":"x"}}}')
add("body.search_knn_query_vector_str", "POST", S, '{"query":{"knn":{"field":"v","query_vector":"x"}}}')
add("body.search_knn_query_unknown_obj", "POST", S, '{"query":{"knn":{"field":"v","query_vector":[1,1],"zzz":{}}}}')
add("body.search_knn_query_no_field", "POST", S, '{"query":{"knn":{"query_vector":[1,1]}}}')
add("body.search_mlt_like_num", "POST", S, '{"query":{"more_like_this":{"like":5}}}')
add("body.search_mlt_fields_num", "POST", S, '{"query":{"more_like_this":{"like":"x","fields":5}}}')
add("body.search_mlt_min_term_freq_str", "POST", S, '{"query":{"more_like_this":{"like":"x","min_term_freq":"x"}}}')
add("body.search_geo_distance_two_fields", "POST", S, '{"query":{"geo_distance":{"distance":"1km","g":[0,0],"g2":[0,0]}}}')
add("body.search_geo_distance_validation_bad", "POST", S, '{"query":{"geo_distance":{"distance":"1km","g":[0,0],"validation_method":"nope"}}}')
add("body.search_geo_distance_type_bad", "POST", S, '{"query":{"geo_distance":{"distance":"1km","g":[0,0],"distance_type":"nope"}}}')
add("body.search_geo_bbox_type_bad", "POST", S, '{"query":{"geo_bounding_box":{"g":{"top_left":[0,1],"bottom_right":[1,0]},"type":"nope"}}}')
add("body.search_geo_bbox_corner_bad", "POST", S, '{"query":{"geo_bounding_box":{"g":{"top_left":"x","bottom_right":[1,0]}}}}')
add("body.search_span_near_slop_str", "POST", S, '{"query":{"span_near":{"clauses":[{"span_term":{"f":"x"}}],"slop":"x"}}}')
add("body.search_span_near_in_order_str", "POST", S, '{"query":{"span_near":{"clauses":[{"span_term":{"f":"x"}}],"in_order":"x"}}}')
add("body.search_span_near_clauses_num", "POST", S, '{"query":{"span_near":{"clauses":5}}}')
add("body.search_span_not_no_include", "POST", S, '{"query":{"span_not":{"exclude":{"span_term":{"f":"y"}}}}}')
add("body.search_span_not_dist_and_pre", "POST", S, '{"query":{"span_not":{"include":{"span_term":{"f":"x"}},"exclude":{"span_term":{"f":"y"}},"dist":1,"pre":1}}}')
add("body.search_span_term_two_fields", "POST", S, '{"query":{"span_term":{"f":"x","t":"y"}}}')
add("body.search_span_term_value_obj", "POST", S, '{"query":{"span_term":{"f":{"value":{}}}}}')
add("body.search_terms_set_msm_both", "POST", S, '{"query":{"terms_set":{"f":{"terms":["x"],"minimum_should_match_field":"n","minimum_should_match_script":"1"}}}}')
add("body.search_terms_set_two_fields", "POST", S, '{"query":{"terms_set":{"f":{"terms":["x"]},"t":{"terms":["y"]}}}}')
add("body.search_terms_set_terms_str", "POST", S, '{"query":{"terms_set":{"f":{"terms":"x"}}}}')
add("body.search_rank_feature_unknown_fn", "POST", S, '{"query":{"rank_feature":{"field":"n","nope":{}}}}')
add("body.search_rank_feature_field_num", "POST", S, '{"query":{"rank_feature":{"field":5}}}')
add("body.search_rank_feature_log_unknown", "POST", S, '{"query":{"rank_feature":{"field":"n","log":{"scaling_factor":1,"zzz":1}}}}')
add("body.search_rank_feature_sigmoid_unknown", "POST", S, '{"query":{"rank_feature":{"field":"n","sigmoid":{"pivot":1,"exponent":1,"zzz":1}}}}')
add("body.search_rank_feature_linear_unknown", "POST", S, '{"query":{"rank_feature":{"field":"n","linear":{"zzz":1}}}}')
add("body.search_pinned_two_fields", "POST", S, '{"query":{"pinned":{"ids":["1"],"docs":[{"_index":"probe","_id":"1"}],"organic":{"match_all":{}}}}}')
add("body.search_pinned_docs_unknown", "POST", S, '{"query":{"pinned":{"docs":[{"_index":"probe","_id":"1","zzz":1}],"organic":{"match_all":{}}}}}')
add("body.search_sparse_vector_two_methods", "POST", S, '{"query":{"sparse_vector":{"field":"f","query_vector":{"a":1},"inference_id":"x","query":"y"}}}')
add("body.search_sparse_vector_pruning_unknown", "POST", S, '{"query":{"sparse_vector":{"field":"f","query_vector":{"a":1},"pruning_config":{"zzz":1}}}}')
add("body.search_shape_relation_bad", "POST", S, '{"query":{"shape":{"g":{"shape":{"type":"point","coordinates":[0,0]},"relation":"nope"}}}}')
add("body.search_match_phrase_prefix_max_exp_str", "POST", S, '{"query":{"match_phrase_prefix":{"f":{"query":"x","max_expansions":"x"}}}}')
add("body.search_match_phrase_prefix_zero_terms_bad", "POST", S, '{"query":{"match_phrase_prefix":{"f":{"query":"x","zero_terms_query":"nope"}}}}')
add("body.search_match_bool_prefix_operator_bad", "POST", S, '{"query":{"match_bool_prefix":{"f":{"query":"x","operator":"nope"}}}}')
add("body.search_exists_two_fields", "POST", S, '{"query":{"exists":{"field":"f","field2":"t"}}}')
add("body.search_bool_must_not_arr_unknown_q", "POST", S, '{"query":{"bool":{"must_not":[{"nope":{}}]}}}')
add("body.search_bool_filter_obj_unknown_q", "POST", S, '{"query":{"bool":{"filter":{"nope":{}}}}}')
add("body.search_bool_msm_bad_str", "POST", S, '{"query":{"bool":{"minimum_should_match":"nope"}}}')
add("body.search_bool_boost_obj", "POST", S, '{"query":{"bool":{"boost":{}}}}')
add("body.search_bool_must_arr_empty_obj", "POST", S, '{"query":{"bool":{"must":[{}]}}}')
add("body.search_bool_must_arr_two_names", "POST", S, '{"query":{"bool":{"must":[{"match_all":{},"match_none":{}}]}}}')
add("body.search_range_lte_and_lt", "POST", S, '{"query":{"range":{"n":{"lt":1,"lte":1}}}}')
add("body.search_range_time_zone_bad", "POST", S, '{"query":{"range":{"d":{"gte":"now","time_zone":"nope"}}}}')
add("body.search_range_format_bad", "POST", S, '{"query":{"range":{"d":{"gte":"2020","format":"nope"}}}}')
add("body.search_range_from_obj", "POST", S, '{"query":{"range":{"n":{"from":{}}}}}')
add("body.search_range_include_lower_str", "POST", S, '{"query":{"range":{"n":{"from":1,"include_lower":"x"}}}}')
add("body.search_fuzzy_value_obj", "POST", S, '{"query":{"fuzzy":{"f":{"value":{}}}}}')
add("body.search_fuzzy_prefix_length_str", "POST", S, '{"query":{"fuzzy":{"f":{"value":"x","prefix_length":"x"}}}}')
add("body.search_fuzzy_transpositions_str", "POST", S, '{"query":{"fuzzy":{"f":{"value":"x","transpositions":"x"}}}}')
add("body.search_fuzzy_rewrite_bad", "POST", S, '{"query":{"fuzzy":{"f":{"value":"x","rewrite":"nope"}}}}')
add("body.search_prefix_rewrite_bad", "POST", S, '{"query":{"prefix":{"f":{"value":"x","rewrite":"nope"}}}}')
add("body.search_prefix_case_insensitive_str", "POST", S, '{"query":{"prefix":{"f":{"value":"x","case_insensitive":"x"}}}}')
add("body.search_wildcard_rewrite_bad", "POST", S, '{"query":{"wildcard":{"f":{"value":"x","rewrite":"nope"}}}}')
add("body.search_wildcard_boost_str", "POST", S, '{"query":{"wildcard":{"f":{"value":"x","boost":"x"}}}}')
add("body.search_regexp_value_obj", "POST", S, '{"query":{"regexp":{"f":{"value":{}}}}}')
add("body.search_regexp_no_value", "POST", S, '{"query":{"regexp":{"f":{"boost":1}}}}')
add("body.search_regexp_rewrite_bad", "POST", S, '{"query":{"regexp":{"f":{"value":"x","rewrite":"nope"}}}}')
add("body.search_regexp_case_insensitive_str", "POST", S, '{"query":{"regexp":{"f":{"value":"x","case_insensitive":"x"}}}}')
add("body.search_query_string_df_num", "POST", S, '{"query":{"query_string":{"query":"x","default_field":5}}}')
add("body.search_query_string_lenient_str", "POST", S, '{"query":{"query_string":{"query":"x","lenient":"x"}}}')
add("body.search_query_string_fuzziness_bad", "POST", S, '{"query":{"query_string":{"query":"x","fuzziness":"nope"}}}')
add("body.search_query_string_msm_obj", "POST", S, '{"query":{"query_string":{"query":"x","minimum_should_match":{}}}}')
add("body.search_query_string_phrase_slop_str", "POST", S, '{"query":{"query_string":{"query":"x","phrase_slop":"x"}}}')
add("body.search_query_string_max_states_str", "POST", S, '{"query":{"query_string":{"query":"x","max_determinized_states":"x"}}}')
add("body.search_query_string_quote_analyzer_bad", "POST", S, '{"query":{"query_string":{"query":"x","quote_analyzer":"nope"}}}')
add("body.search_query_string_escape_str", "POST", S, '{"query":{"query_string":{"query":"x","escape":"x"}}}')
add("body.search_query_string_analyze_wildcard_str", "POST", S, '{"query":{"query_string":{"query":"x","analyze_wildcard":"x"}}}')
add("body.search_sqs_analyzer_bad", "POST", S, '{"query":{"simple_query_string":{"query":"x","analyzer":"nope"}}}')
add("body.search_sqs_lenient_str", "POST", S, '{"query":{"simple_query_string":{"query":"x","lenient":"x"}}}')
add("body.search_sqs_msm_obj", "POST", S, '{"query":{"simple_query_string":{"query":"x","minimum_should_match":{}}}}')
add("body.search_sqs_boost_str", "POST", S, '{"query":{"simple_query_string":{"query":"x","boost":"x"}}}')
add("body.search_sqs_fields_arr_num", "POST", S, '{"query":{"simple_query_string":{"query":"x","fields":[5]}}}')
add("body.search_sqs_fuzzy_prefix_length_str", "POST", S, '{"query":{"simple_query_string":{"query":"x","fuzzy_prefix_length":"x"}}}')
add("body.search_match_phrase_analyzer_bad", "POST", S, '{"query":{"match_phrase":{"f":{"query":"x","analyzer":"nope"}}}}')
add("body.search_match_phrase_boost_str", "POST", S, '{"query":{"match_phrase":{"f":{"query":"x","boost":"x"}}}}')
add("body.search_match_phrase_zero_terms_bad", "POST", S, '{"query":{"match_phrase":{"f":{"query":"x","zero_terms_query":"nope"}}}}')
add("body.search_match_phrase_query_obj", "POST", S, '{"query":{"match_phrase":{"f":{"query":{}}}}}')
add("body.search_match_phrase_no_query", "POST", S, '{"query":{"match_phrase":{"f":{"slop":1}}}}')
add("body.search_match_phrase_prefix_no_query", "POST", S, '{"query":{"match_phrase_prefix":{"f":{"slop":1}}}}')
add("body.search_match_bool_prefix_no_query", "POST", S, '{"query":{"match_bool_prefix":{"f":{"analyzer":"standard"}}}}')
add("body.search_match_bool_prefix_fuzziness_bad", "POST", S, '{"query":{"match_bool_prefix":{"f":{"query":"x","fuzziness":"nope"}}}}')
add("body.search_match_bool_prefix_msm_obj", "POST", S, '{"query":{"match_bool_prefix":{"f":{"query":"x","minimum_should_match":{}}}}}')
add("body.search_multi_match_fuzziness_bad", "POST", S, '{"query":{"multi_match":{"query":"x","fuzziness":"nope"}}}')
add("body.search_multi_match_msm_obj", "POST", S, '{"query":{"multi_match":{"query":"x","minimum_should_match":{}}}}')
add("body.search_multi_match_slop_str", "POST", S, '{"query":{"multi_match":{"query":"x","slop":"x"}}}')
add("body.search_multi_match_boost_str", "POST", S, '{"query":{"multi_match":{"query":"x","boost":"x"}}}')
add("body.search_multi_match_analyzer_bad", "POST", S, '{"query":{"multi_match":{"query":"x","analyzer":"nope"}}}')
add("body.search_multi_match_zero_terms_bad", "POST", S, '{"query":{"multi_match":{"query":"x","zero_terms_query":"nope"}}}')
add("body.search_multi_match_fields_arr_num", "POST", S, '{"query":{"multi_match":{"query":"x","fields":[5]}}}')
add("body.search_multi_match_lenient_str", "POST", S, '{"query":{"multi_match":{"query":"x","lenient":"x"}}}')
add("body.search_multi_match_query_arr", "POST", S, '{"query":{"multi_match":{"query":[]}}}')
add("body.search_match_cutoff_str", "POST", S, '{"query":{"match":{"f":{"query":"x","cutoff_frequency":"x"}}}}')
add("body.search_match_cutoff", "POST", S, '{"query":{"match":{"f":{"query":"x","cutoff_frequency":0.1}}}}')
add("body.search_match_type", "POST", S, '{"query":{"match":{"f":{"query":"x","type":"phrase"}}}}')
add("body.search_match_fuzzy_transpositions_str", "POST", S, '{"query":{"match":{"f":{"query":"x","fuzzy_transpositions":"x"}}}}')
add("body.search_match_auto_generate_str", "POST", S, '{"query":{"match":{"f":{"query":"x","auto_generate_synonyms_phrase_query":"x"}}}}')
add("body.search_match_fuzzy_rewrite_bad", "POST", S, '{"query":{"match":{"f":{"query":"x","fuzzy_rewrite":"nope"}}}}')
add("body.search_match_max_expansions_str", "POST", S, '{"query":{"match":{"f":{"query":"x","max_expansions":"x"}}}}')
add("body.search_match_name_num", "POST", S, '{"query":{"match":{"f":{"query":"x","_name":5}}}}')
add("body.search_match_query_null", "POST", S, '{"query":{"match":{"f":{"query":null}}}}')
add("body.search_match_null", "POST", S, '{"query":{"match":{"f":null}}}')
add("body.search_term_null", "POST", S, '{"query":{"term":{"f":null}}}')
add("body.search_term_value_null", "POST", S, '{"query":{"term":{"f":{"value":null}}}}')
add("body.search_term_value_arr", "POST", S, '{"query":{"term":{"f":{"value":[]}}}}')
add("body.search_term_boost_obj", "POST", S, '{"query":{"term":{"f":{"value":"x","boost":{}}}}}')
add("body.search_term_name_num", "POST", S, '{"query":{"term":{"f":{"value":"x","_name":5}}}}')
add("body.search_range_null", "POST", S, '{"query":{"range":{"n":null}}}')
add("body.search_range_arr", "POST", S, '{"query":{"range":{"n":[]}}}')
add("body.search_range_name_num", "POST", S, '{"query":{"range":{"n":{"gte":1,"_name":5}}}}')
add("body.search_exists_null", "POST", S, '{"query":{"exists":{"field":null}}}')
add("body.search_ids_null", "POST", S, '{"query":{"ids":{"values":null}}}')
add("body.search_ids_values_arr_null", "POST", S, '{"query":{"ids":{"values":[null]}}}')
add("body.search_ids_values_arr_arr", "POST", S, '{"query":{"ids":{"values":[[]]}}}')
add("body.search_bool_null", "POST", S, '{"query":{"bool":{"must":null}}}')
add("body.search_bool_must_arr_null", "POST", S, '{"query":{"bool":{"must":[null]}}}')
add("body.search_bool_boost_null", "POST", S, '{"query":{"bool":{"boost":null}}}')
add("body.search_query_null", "POST", S, '{"query":null}')
add("body.search_match_all_null", "POST", S, '{"query":{"match_all":null}}')
add("body.search_size_null_from_null", "POST", S, '{"size":null,"from":null}')
add("body.search_sort_null", "POST", S, '{"sort":null}')
add("body.search_sort_arr_null", "POST", S, '{"sort":[null]}')
add("body.search_aggs_null", "POST", S, '{"aggs":null}')
add("body.search_aggs_a_null", "POST", S, '{"aggs":{"a":null}}')
add("body.search_aggs_terms_null", "POST", S, '{"aggs":{"a":{"terms":null}}}')
add("body.search_aggs_terms_field_null", "POST", S, '{"aggs":{"a":{"terms":{"field":null}}}}')
add("body.search_highlight_null", "POST", S, '{"highlight":null}')
add("body.search_source_null", "POST", S, '{"_source":null}')
add("body.search_explain_null", "POST", S, '{"explain":null}')
add("body.search_timeout_null", "POST", S, '{"timeout":null}')
add("body.search_knn_null", "POST", S, '{"knn":null}')
add("body.search_suggest_null", "POST", S, '{"suggest":null}')
add("body.search_collapse_null", "POST", S, '{"collapse":null}')
add("body.search_rescore_null", "POST", S, '{"rescore":null}')
add("body.search_retriever_null", "POST", S, '{"retriever":null}')
add("body.search_pit_null", "POST", S, '{"pit":null}')
add("body.search_post_filter_null", "POST", S, '{"post_filter":null}')
add("body.search_fields_null", "POST", S, '{"fields":null}')
add("body.search_stored_fields_null", "POST", S, '{"stored_fields":null}')
add("body.search_docvalue_fields_null", "POST", S, '{"docvalue_fields":null}')
add("body.search_script_fields_null", "POST", S, '{"script_fields":null}')
add("body.search_track_total_hits_null", "POST", S, '{"track_total_hits":null}')
add("body.search_search_after_null", "POST", S, '{"search_after":null}')
add("body.search_min_score_null", "POST", S, '{"min_score":null}')
add("body.search_terminate_after_null", "POST", S, '{"terminate_after":null}')
add("body.search_version_null", "POST", S, '{"version":null}')
add("body.search_profile_null", "POST", S, '{"profile":null}')
add("body.search_runtime_mappings_null", "POST", S, '{"runtime_mappings":null}')
add("body.search_indices_boost_null", "POST", S, '{"indices_boost":null}')
add("body.search_stats_null", "POST", S, '{"stats":null}')
add("body.search_ext_null", "POST", S, '{"ext":null}')
add("body.search_slice_null", "POST", S, '{"slice":null}')
add("body.search_seq_no_null", "POST", S, '{"seq_no_primary_term":null}')
add("body.search_track_scores_null", "POST", S, '{"track_scores":null}')
add("body.search_rank_null", "POST", S, '{"rank":null}')
add("body.search_sub_searches_null", "POST", S, '{"sub_searches":null}')
add("body.search_name_null", "POST", S, '{"_name":null}')
add("empty_body.POST_missing-alias__rollover", "POST", "/no-such-alias/_rollover", "{}")
add("empty_body.POST_missing-alias__rollover_new", "POST", "/no-such-alias/_rollover/new-idx", "{}")
add("empty_body.POST_probe__rollover_concrete", "POST", "/probe/_rollover", "{}")
add("empty_body.POST_probe-alias__rollover_unknown", "POST", "/probe-alias/_rollover", '{"zz":1}')
add("empty_body.POST_probe-alias__rollover_cond_unknown", "POST", "/probe-alias/_rollover", '{"conditions":{"zz":1}}')
add("empty_body.POST__index_template__simulate_unknown", "POST", "/_index_template/_simulate", '{"index_patterns":["x*"],"zz":1}')
add("empty_body.POST__index_template__simulate_index_unknown", "POST", "/_index_template/_simulate_index/probe-x", '{"index_patterns":["x*"],"zz":1}')
add("empty_body.POST__index_template__simulate_template_unknown", "POST", "/_index_template/_simulate", '{"index_patterns":["x*"],"template":{"zz":1}}')
add("empty_body.PUT__scripts_s1_unknown", "PUT", "/_scripts/s1", '{"script":{"lang":"painless","source":"1"},"zz":1}')
add("empty_body.PUT__scripts_s1_script_unknown", "PUT", "/_scripts/s1", '{"script":{"lang":"painless","source":"1","zz":1}}')
add("empty_body.PUT__scripts_s1_no_lang", "PUT", "/_scripts/s1", '{"script":{"source":"1"}}')
add("empty_body.PUT__scripts_s1_no_source", "PUT", "/_scripts/s1", '{"script":{"lang":"painless"}}')
add("empty_body.PUT__scripts_s1_script_num", "PUT", "/_scripts/s1", '{"script":5}')
add("empty_body.PUT__scripts_s1_empty_script", "PUT", "/_scripts/s1", '{"script":{}}')
add("empty_body.PUT__scripts_s1_syntax", "PUT", "/_scripts/s1", '{"script":}')
add("empty_body.POST_probe__knn_search_unknown", "POST", "/probe/_knn_search", '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1},"zz":1}')
add("empty_body.POST_probe__knn_search_knn_unknown", "POST", "/probe/_knn_search", '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1,"zz":1}}')
add("empty_body.POST_probe__knn_search_knn_num", "POST", "/probe/_knn_search", '{"knn":5}')
add("empty_body.POST_probe__knn_search_filter_unknown", "POST", "/probe/_knn_search", '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1},"filter":{"nope":{}}}')
add("empty_body.POST_probe__knn_search_no_k", "POST", "/probe/_knn_search", '{"knn":{"field":"v","query_vector":[1,1]}}')
add("empty_body.POST_probe__knn_search_syntax", "POST", "/probe/_knn_search", '{"knn":}')
add("empty_body.PUT__synonyms_syn1_unknown", "PUT", "/_synonyms/syn1", '{"synonyms_set":[{"synonyms":"a,b"}],"zz":1}')
add("empty_body.PUT__synonyms_syn1_rule_unknown", "PUT", "/_synonyms/syn1", '{"synonyms_set":[{"synonyms":"a,b","zz":1}]}')
add("empty_body.PUT__synonyms_syn1_set_num", "PUT", "/_synonyms/syn1", '{"synonyms_set":5}')
add("empty_body.PUT__synonyms_syn1_syntax", "PUT", "/_synonyms/syn1", '{"synonyms_set":}')
add("empty_body.PUT__synonyms_syn1_r1_unknown", "PUT", "/_synonyms/syn1/r1", '{"synonyms":"a,b","zz":1}')
add("empty_body.PUT__synonyms_syn1_r1_num", "PUT", "/_synonyms/syn1/r1", '{"synonyms":5}')
add("empty_body.PUT__synonyms_nosuchset_r1", "PUT", "/_synonyms/no-such-set/r1", '{"synonyms":"a,b"}')
add("empty_body.POST__query_rules_rs1__test_unknown", "POST", "/_query_rules/rs1/_test", '{"match_criteria":{"a":"b"},"zz":1}')
add("empty_body.POST__query_rules_rs1__test_mc_num", "POST", "/_query_rules/rs1/_test", '{"match_criteria":5}')
add("empty_body.POST__query_rules_rs1__test_missing", "POST", "/_query_rules/no-such-rs/_test", '{"match_criteria":{"a":"b"}}')
add("empty_body.POST__query_rules_rs1__test_syntax", "POST", "/_query_rules/rs1/_test", '{"match_criteria":}')
add("empty_body.POST__validate_query_unknown", "POST", "/probe/_validate/query", '{"zz":1}')
add("empty_body.POST__validate_query_syntax", "POST", "/probe/_validate/query", '{"query":}')
add("empty_body.POST__validate_query_null", "POST", "/probe/_validate/query", '{"query":null}')
add("empty_body.POST_probe__termvectors_unknown", "POST", "/probe/_termvectors", '{"doc":{"f":"a"},"zz":1}')
add("empty_body.POST_probe__termvectors_doc_num", "POST", "/probe/_termvectors", '{"doc":5}')
add("empty_body.POST_probe__termvectors_fields_num", "POST", "/probe/_termvectors", '{"doc":{"f":"a"},"fields":5}')
add("empty_body.POST_probe__termvectors_filter_unknown", "POST", "/probe/_termvectors", '{"doc":{"f":"a"},"filter":{"zz":1}}')
add("empty_body.POST_probe__termvectors_id_in_body", "POST", "/probe/_termvectors", '{"_id":"1"}')
add("empty_body.POST_probe__termvectors_syntax", "POST", "/probe/_termvectors", '{"doc":}')
add("empty_body.POST_probe__termvectors_1_unknown", "POST", "/probe/_termvectors/1", '{"zz":1}')
add("empty_body.POST_probe__termvectors_1_doc", "POST", "/probe/_termvectors/1", '{"doc":{"f":"a"}}')
add("empty_body.POST__mtermvectors_unknown", "POST", "/probe/_mtermvectors", '{"docs":[{"_id":"1"}],"zz":1}')
add("empty_body.POST__mtermvectors_doc_unknown", "POST", "/probe/_mtermvectors", '{"docs":[{"_id":"1","zz":1}]}')
add("empty_body.POST__mtermvectors_empty", "POST", "/probe/_mtermvectors", '{}')
add("empty_body.POST__mtermvectors_docs_num", "POST", "/probe/_mtermvectors", '{"docs":5}')
add("empty_body.POST__watcher_watch__execute_unknown", "POST", "/_watcher/watch/_execute", '{"zz":1}')
add("empty_body.POST__watcher_watch__execute_empty", "POST", "/_watcher/watch/_execute", '{}')
add("empty_body.POST__watcher_watch__execute_watch_num", "POST", "/_watcher/watch/_execute", '{"watch":5}')
add("empty_body.POST__watcher_watch__execute_watch_unknown", "POST", "/_watcher/watch/_execute", '{"watch":{"trigger":{"schedule":{"interval":"10s"}},"input":{"none":{}},"actions":{},"zz":1}}')
add("empty_body.POST__watcher_watch__execute_modes_bad", "POST", "/_watcher/watch/_execute", '{"action_modes":{"a":"nope"},"watch":{"trigger":{"schedule":{"interval":"10s"}},"input":{"none":{}},"actions":{}}}')
add("empty_body.POST__watcher_watch_w1__execute_unknown", "POST", "/_watcher/watch/w1/_execute", '{"zz":1}')
add("empty_body.POST__watcher_watch_w1__execute_syntax", "POST", "/_watcher/watch/w1/_execute", '{"zz":}')
add("empty_body.POST_missing-idx__update_by_query_unknown", "POST", "/missing-idx/_update_by_query", '{"zz":1}')
add("empty_body.POST_missing-idx__update_by_query_syntax", "POST", "/missing-idx/_update_by_query", '{"query":}')
add("empty_body.POST_missing-idx__delete_by_query", "POST", "/missing-idx/_delete_by_query", '{}')
add("empty_body.POST_missing-idx__delete_by_query_q", "POST", "/missing-idx/_delete_by_query", '{"query":{"match_all":{}}}')
add("empty_body.POST_probe__update_by_query_empty", "POST", "/probe/_update_by_query", '{}')
add("empty_body.POST_probe__delete_by_query_empty", "POST", "/probe/_delete_by_query", '{}')
add("empty_body.POST_missing-idx__search", "POST", "/missing-idx/_search", '{}')
add("empty_body.POST_missing-idx__search_unknown_key", "POST", "/missing-idx/_search", '{"zz":1}')
add("empty_body.POST_missing-idx__search_syntax", "POST", "/missing-idx/_search", '{"zz":}')
add("empty_body.POST_missing-idx__count_unknown", "POST", "/missing-idx/_count", '{"zz":1}')
add("empty_body.POST_missing-idx__explain_unknown", "POST", "/missing-idx/_explain/1", '{"query":{"nope":{}}}')
add("empty_body.POST_missing-idx__termvectors", "POST", "/missing-idx/_termvectors", '{"doc":{"f":"a"}}')
add("empty_body.POST_missing-idx__validate_query", "POST", "/missing-idx/_validate/query", '{"query":{"match_all":{}}}')
add("empty_body.POST_missing-idx__validate_query_unknown_q", "POST", "/missing-idx/_validate/query", '{"query":{"nope":{}}}')
add("empty_body.POST_missing-idx__knn_search", "POST", "/missing-idx/_knn_search", '{"knn":{"field":"v","query_vector":[1,1],"k":1,"num_candidates":1}}')
add("empty_body.POST_missing-idx__async_search_unknown", "POST", "/missing-idx/_async_search", '{"zz":1}')
add("empty_body.POST_missing-idx__async_search", "POST", "/missing-idx/_async_search", '{}')
add("empty_body.POST_missing-idx__msearch_unknown", "POST", "/missing-idx/_msearch", '{}\n{"zz":1}\n')
add("empty_body.POST__search_scroll_body_unknown", "POST", "/_search/scroll", '{"scroll_id":"foo","zz":1}')
add("empty_body.POST__search_scroll_body_empty", "POST", "/_search/scroll", '{}')
add("empty_body.POST__search_scroll_body_syntax", "POST", "/_search/scroll", '{"scroll_id":}')
add("empty_body.POST__search_scroll_body_num", "POST", "/_search/scroll", '{"scroll_id":5}')
add("empty_body.POST__search_scroll_body_arr", "POST", "/_search/scroll", '{"scroll_id":["foo"]}')
add("empty_body.POST__search_scroll_scroll_bad", "POST", "/_search/scroll", '{"scroll_id":"foo","scroll":"5x"}')
add("empty_body.DELETE__search_scroll_body_unknown", "DELETE", "/_search/scroll", '{"scroll_id":["foo"],"zz":1}')
add("empty_body.DELETE__search_scroll_body_empty", "DELETE", "/_search/scroll", '{}')
add("empty_body.DELETE__search_scroll_body_num", "DELETE", "/_search/scroll", '{"scroll_id":5}')
add("empty_body.DELETE__search_scroll_body_arr_num", "DELETE", "/_search/scroll", '{"scroll_id":[5]}')
add("empty_body.DELETE__pit_body_unknown", "DELETE", "/_pit", '{"id":"foo","zz":1}')
add("empty_body.DELETE__pit_body_empty", "DELETE", "/_pit", '{}')
add("empty_body.DELETE__pit_body_num", "DELETE", "/_pit", '{"id":5}')
add("empty_body.DELETE__pit_body_arr", "DELETE", "/_pit", '{"id":["foo"]}')
add("empty_body.DELETE__pit_no_body", "DELETE", "/_pit", None)
add("empty_body.POST_probe__pit_no_keep_alive", "POST", "/probe/_pit", None)
add("empty_body.POST_probe__pit_bad_keep_alive", "POST", "/probe/_pit?keep_alive=5x", None)
add("empty_body.POST_probe__pit_body_unknown", "POST", "/probe/_pit?keep_alive=1m", '{"zz":1}')
add("empty_body.POST_probe__pit_index_filter_unknown_q", "POST", "/probe/_pit?keep_alive=1m", '{"index_filter":{"nope":{}}}')
add("empty_body.GET__async_search_status_missing", "GET", "/_async_search/status/FmRkZWZkZWZkZWZkZWZkZWZkZWZkZWYWZGVmZGVmZGVmZGVmOjEyMzQ1", None)
add("empty_body.GET__async_search_missing", "GET", "/_async_search/FmRkZWZkZWZkZWZkZWZkZWZkZWZkZWYWZGVmZGVmZGVmZGVmOjEyMzQ1", None)
add("empty_body.DELETE__async_search_missing", "DELETE", "/_async_search/FmRkZWZkZWZkZWZkZWZkZWZkZWZkZWYWZGVmZGVmZGVmZGVmOjEyMzQ1", None)
add("empty_body.GET__async_search_unknown_ka", "GET", "/_async_search/FmRkZWZkZWZkZWZkZWZkZWZkZWZkZWYWZGVmZGVmZGVmZGVmOjEyMzQ1?keep_alive=5x", None)
add("empty_body.POST_probe__async_search_bad_ka", "POST", "/probe/_async_search?keep_alive=5x", '{}')
add("empty_body.POST_probe__async_search_short_ka", "POST", "/probe/_async_search?keep_alive=1ms", '{}')
add("empty_body.POST_probe__async_search_bad_wait", "POST", "/probe/_async_search?wait_for_completion_timeout=5x", '{}')
add("empty_body.POST_probe__async_search_scroll", "POST", "/probe/_async_search?scroll=1m", '{}')
add("empty_body.POST_probe__async_search_keep_on_bad", "POST", "/probe/_async_search?keep_on_completion=x", '{}')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.jsonl"), "w") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")
print(len(cases), "cases")
