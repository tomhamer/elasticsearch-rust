//! `_mget` request body, parsed the way ES 9.5.5's `MultiGetRequest#add`
//! streams it: the top level takes only the `docs` and `ids` arrays
//! (`_source`, `stored_fields`, ... there are errors, not defaults: those
//! come from the URL), and each `docs` element runs `parseDocuments`'
//! token loop, including what it does with tokens it does not consume
//! (an unknown key's array elements are read as values of that key, an
//! unknown key's object ends the element early).

use elasticsearch_engine::error::EsError;
use elasticsearch_engine::search::SourceFilter;
use serde_json::{Map, Value};

/// One item to get.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Item {
    pub index: Option<String>,
    pub id: Option<String>,
    pub routing: Option<String>,
    /// The item's own `_source` (ES `fetchSourceContext`); `None`: the
    /// request's default.
    pub source: Option<SourceFilter>,
    /// The item's own `stored_fields`; `None`: the request's default.
    pub stored: Option<Vec<String>>,
}

/// Jackson's name of the token a value starts with.
fn token(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Number(_) => "VALUE_NUMBER",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

/// `parser.text()` of a scalar (`None` for `null`).
fn text(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        x => Some(x.to_string()),
    }
}

/// How many fields named `key` come before the field holding `target`
/// in document order (the position hint's `nth`).
fn rank(root: &Value, target: &Value, key: &str) -> usize {
    fn walk(v: &Value, target: &Value, key: &str, n: &mut usize) -> bool {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    if std::ptr::eq(x, target) {
                        return true;
                    }
                    if k == key {
                        *n += 1;
                    }
                    if walk(x, target, key, n) {
                        return true;
                    }
                }
                false
            }
            Value::Array(a) => a.iter().any(|x| walk(x, target, key, n)),
            _ => false,
        }
    }
    let mut n = 0;
    walk(root, target, key, &mut n);
    n
}

/// ES `ParsingException` at the value of field `key` (`line` / `col`).
fn parsing_at(root: &Value, key: &str, value: &Value, reason: String) -> EsError {
    EsError::parsing(reason).at_key(key, None).nth_key(rank(root, value, key)).with_lc()
}

/// The `FetchSourceContext` being built for one element.
#[derive(Clone, Default)]
struct Fetch {
    fetch: bool,
    includes: Vec<String>,
    excludes: Vec<String>,
}

impl Fetch {
    fn filter(self) -> SourceFilter {
        if !self.fetch {
            SourceFilter::None
        } else if self.includes.is_empty() && self.excludes.is_empty() {
            SourceFilter::All
        } else {
            SourceFilter::Filter { includes: self.includes, excludes: self.excludes }
        }
    }
}

/// One `docs` element's fields as `parseDocuments` reads them.
struct Doc<'r> {
    root: &'r Value,
    item: Item,
    fetch: Option<Fetch>,
}

/// `Err(None)`: the element ended early at an unconsumed object's end
/// (the next token is then not an element start).
type Step = Result<(), Option<EsError>>;

impl Doc<'_> {
    fn value(&mut self, key: &str, v: &Value) -> Step {
        match key {
            "_index" => self.item.index = text(v),
            "_id" => self.item.id = text(v),
            "routing" => self.item.routing = text(v),
            // `_slice` routes like `routing` (ES `SliceIndexing`).
            "_slice" => self.item.routing = text(v),
            "fields" => return Err(Some(parsing_at(self.root, key, v, "Unsupported field [fields] used, expected [stored_fields] instead".into()))),
            "stored_fields" => self.item.stored = Some(text(v).into_iter().collect()),
            "version" | "version_type" => {}
            "_source" => {
                let prev = self.fetch.take();
                self.fetch = Some(match v {
                    Value::Bool(b) => Fetch { fetch: *b, ..prev.unwrap_or_default() },
                    Value::String(s) => Fetch { fetch: prev.as_ref().is_none_or(|p| p.fetch), includes: vec![s.clone()], excludes: prev.map(|p| p.excludes).unwrap_or_default() },
                    other => return Err(Some(EsError::new(400, "parse_exception", format!("illegal type for _source: [{}]", token(other))))),
                });
            }
            _ => return Err(Some(EsError::new(400, "parse_exception", format!("failed to parse multi get request. unknown field [{key}]")))),
        }
        Ok(())
    }

    /// Tokens the loop does not consume: array elements are values of the
    /// same key; an object's fields are read as the element's own fields,
    /// and its end ends the element.
    fn unconsumed(&mut self, key: &str, v: &Value) -> Step {
        match v {
            Value::Array(a) => {
                for x in a {
                    self.unconsumed(key, x)?;
                }
                Ok(())
            }
            Value::Object(m) => {
                self.fields(m)?;
                Err(None)
            }
            x => self.value(key, x),
        }
    }

    fn fields(&mut self, m: &Map<String, Value>) -> Step {
        for (k, v) in m {
            match v {
                Value::Array(a) => match k.as_str() {
                    "fields" => return Err(Some(parsing_at(self.root, k, v, "Unsupported field [fields] used, expected [stored_fields] instead".into()))),
                    "stored_fields" => self.item.stored = Some(a.iter().filter_map(text).collect()),
                    "_source" => {
                        let prev = self.fetch.take();
                        self.fetch = Some(Fetch { fetch: prev.as_ref().is_none_or(|p| p.fetch), includes: a.iter().filter_map(text).collect(), excludes: prev.map(|p| p.excludes).unwrap_or_default() });
                    }
                    _ => self.unconsumed(k, v)?,
                },
                Value::Object(o) if k == "_source" => {
                    let (mut includes, mut excludes) = (Vec::new(), Vec::new());
                    for (sk, sv) in o {
                        let list = match sk.as_str() {
                            "includes" | "include" => &mut includes,
                            "excludes" | "exclude" => &mut excludes,
                            // `parser.text()` on the FIELD_NAME token.
                            _ => return Err(Some(EsError::illegal_argument("Expected text at @POS@ but found FIELD_NAME").at_key(sk, Some("_source")).cursor("key"))),
                        };
                        match sv {
                            Value::Array(xs) => list.extend(xs.iter().filter_map(text)),
                            Value::Object(_) => return Err(Some(EsError::new(400, "parse_exception", "unexpected token while parsing source settings"))),
                            x => list.extend(text(x)),
                        }
                    }
                    let prev = self.fetch.take();
                    self.fetch = Some(Fetch { fetch: prev.is_none_or(|p| p.fetch), includes, excludes });
                }
                Value::Object(_) => self.unconsumed(k, v)?,
                x => self.value(k, x)?,
            }
        }
        Ok(())
    }
}

/// The items of an `_mget` body (`index`: the path's index, `routing`:
/// the URL's). Per-item `_source` / `stored_fields` stay `None` when the
/// element does not set them.
pub fn parse(body: &Value, index: Option<&str>, routing: Option<&str>) -> Result<Vec<Item>, EsError> {
    let top = match body {
        Value::Object(m) => m,
        // No body: the request validation reports "no documents to get".
        Value::Null => return Ok(Vec::new()),
        other => return Err(EsError::parsing(format!("unexpected token [{}], expected [START_OBJECT]", token(other))).cursor("root").with_lc()),
    };
    let mut items = Vec::new();
    for (k, v) in top {
        let Value::Array(a) = v else {
            return Err(parsing_at(body, k, v, format!("unexpected token [{}], expected [FIELD_NAME] or [START_ARRAY]", token(v))));
        };
        match k.as_str() {
            "docs" => {
                for d in a {
                    let Value::Object(m) = d else {
                        return Err(EsError::illegal_argument("docs array element should include an object"));
                    };
                    let mut doc = Doc { root: body, item: Item { index: index.map(str::to_string), routing: routing.map(str::to_string), ..Default::default() }, fetch: None };
                    match doc.fields(m) {
                        Ok(()) => {}
                        Err(Some(e)) => return Err(e),
                        Err(None) => return Err(EsError::illegal_argument("docs array element should include an object")),
                    }
                    doc.item.source = doc.fetch.map(Fetch::filter);
                    items.push(doc.item);
                }
            }
            "ids" => {
                for id in a {
                    if id.is_array() || id.is_object() {
                        return Err(EsError::illegal_argument("ids array element should only contain ids"));
                    }
                    items.push(Item { index: index.map(str::to_string), id: text(id), routing: routing.map(str::to_string), ..Default::default() });
                }
            }
            _ => return Err(parsing_at(body, k, v, format!("unknown key [{k}] for a START_ARRAY, expected [docs] or [ids]"))),
        }
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn err(body: Value) -> (String, String) {
        let e = parse(&body, Some("i"), None).unwrap_err();
        (e.kind.clone(), e.reason.clone())
    }

    #[test]
    fn top_level_takes_only_docs_and_ids() {
        assert_eq!(err(json!({"ids": ["1"], "_source": false})).1, "unexpected token [VALUE_BOOLEAN], expected [FIELD_NAME] or [START_ARRAY]");
        assert_eq!(err(json!({"ids": ["1"], "_source": {"includes": ["a"]}})).1, "unexpected token [START_OBJECT], expected [FIELD_NAME] or [START_ARRAY]");
        assert_eq!(err(json!({"ids": ["1"], "_source": ["a"]})).1, "unknown key [_source] for a START_ARRAY, expected [docs] or [ids]");
        assert_eq!(err(json!({"ids": ["1"], "stored_fields": ["a"]})).1, "unknown key [stored_fields] for a START_ARRAY, expected [docs] or [ids]");
        assert_eq!(err(json!([1])).1, "unexpected token [START_ARRAY], expected [START_OBJECT]");
        assert_eq!(err(json!({"ids": [{}]})).1, "ids array element should only contain ids");
        assert_eq!(err(json!({"docs": [1]})).1, "docs array element should include an object");
    }

    #[test]
    fn element_fields_follow_parse_documents() {
        let items = parse(&json!({"docs": [
            {"_id": "1", "_source": false},
            {"_id": "2", "_source": "a"},
            {"_id": "3", "_source": ["a", "o.x"]},
            {"_id": "4", "_source": {"includes": "o.*", "excludes": ["o.y"]}},
            {"_id": "5", "stored_fields": "a", "routing": "r"},
            {"_id": "6", "stored_fields": ["a", "b"], "_source": true},
            {"_id": ["7"], "nothing": []},
            {"_id": "8"}
        ]}), Some("i"), Some("u")).unwrap();
        assert_eq!(items[0].source, Some(SourceFilter::None));
        assert_eq!(items[1].source, Some(SourceFilter::Filter { includes: vec!["a".into()], excludes: vec![] }));
        assert_eq!(items[2].source, Some(SourceFilter::Filter { includes: vec!["a".into(), "o.x".into()], excludes: vec![] }));
        assert_eq!(items[3].source, Some(SourceFilter::Filter { includes: vec!["o.*".into()], excludes: vec!["o.y".into()] }));
        assert_eq!((items[4].stored.clone(), items[4].routing.as_deref()), (Some(vec!["a".to_string()]), Some("r")));
        assert_eq!((items[5].stored.clone().unwrap().len(), items[5].source.clone()), (2, Some(SourceFilter::All)));
        // An unknown key's array elements are values of that key; `[]` is nothing.
        assert_eq!(items[6].id.as_deref(), Some("7"));
        assert_eq!((items[7].source.clone(), items[7].stored.clone(), items[7].routing.as_deref()), (None, None, Some("u")));
        assert_eq!(err(json!({"docs": [{"_id": "1", "foo": "x"}]})), ("parse_exception".into(), "failed to parse multi get request. unknown field [foo]".into()));
        assert_eq!(err(json!({"docs": [{"_id": "1", "foo": [1]}]})).1, "failed to parse multi get request. unknown field [foo]");
        assert_eq!(err(json!({"docs": [{"_id": "1", "foo": {}}]})).1, "docs array element should include an object");
        assert_eq!(err(json!({"docs": [{"_id": "1", "fields": ["a"]}]})), ("parsing_exception".into(), "Unsupported field [fields] used, expected [stored_fields] instead".into()));
        assert_eq!(err(json!({"docs": [{"_id": "1", "_source": 5}]})), ("parse_exception".into(), "illegal type for _source: [VALUE_NUMBER]".into()));
        assert_eq!(err(json!({"docs": [{"_id": "1", "_source": {"includes": ["a"], "bad": ["x"]}}]})).0, "illegal_argument_exception");
    }
}
