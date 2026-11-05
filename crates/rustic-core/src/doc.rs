//! Document storage and retrieval.
//!
//! This module handles the stored fields (original document data)
//! that are returned in search results.

use crate::types::Document;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A stored document with its metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredDoc {
    /// The document data.
    pub doc: Document,
    /// The document's unique ID (if provided).
    pub doc_id: Option<String>,
    /// The index name.
    pub index: String,
    /// The segment ID.
    pub segment_id: u32,
    /// The local doc ID within the segment.
    pub local_doc_id: u32,
}

impl StoredDoc {
    pub fn new(doc: Document, index: &str, segment_id: u32, local_doc_id: u32) -> Self {
        Self {
            doc,
            doc_id: None,
            index: index.to_string(),
            segment_id,
            local_doc_id,
        }
    }

    /// Convert to a JSON-compatible map.
    pub fn to_json_map(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        for (field_id, value) in &self.doc.fields {
            let key = format!("field_{}", field_id);
            map.insert(
                key,
                serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
            );
        }
        map
    }
}

/// A document store that can retrieve stored documents by address.
pub struct DocStore {
    /// Per-segment stored documents.
    segments: Vec<Vec<Document>>,
}

impl DocStore {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Add a segment's stored documents.
    pub fn add_segment(&mut self, docs: Vec<Document>) {
        self.segments.push(docs);
    }

    /// Get a document by segment ID and local doc ID.
    pub fn get(&self, segment_id: u32, local_doc_id: u32) -> Option<&Document> {
        self.segments
            .get(segment_id as usize)
            .and_then(|docs| docs.get(local_doc_id as usize))
    }

    /// Get the number of segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Get the total number of documents.
    pub fn num_docs(&self) -> u32 {
        self.segments.iter().map(|s| s.len() as u32).sum()
    }
}

impl Default for DocStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Document, FieldValue};

    #[test]
    fn test_doc_store() {
        let mut store = DocStore::new();

        let mut doc1 = Document::new();
        doc1.fields.push((0, FieldValue::Text("hello".into())));
        let mut doc2 = Document::new();
        doc2.fields.push((0, FieldValue::Text("world".into())));

        store.add_segment(vec![doc1, doc2]);

        assert_eq!(store.num_segments(), 1);
        assert_eq!(store.num_docs(), 2);
        assert!(store.get(0, 0).is_some());
        assert!(store.get(0, 1).is_some());
        assert!(store.get(0, 2).is_none());
        assert!(store.get(1, 0).is_none());
    }
}
