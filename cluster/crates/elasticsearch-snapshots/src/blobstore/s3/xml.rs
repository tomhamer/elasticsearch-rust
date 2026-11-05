//! Just enough XML for S3 responses: extract the text of (non-nested,
//! same-named) elements and (un)escape entities.

/// Inner content of every `<tag>...</tag>` (raw, not unescaped).
pub fn tags<'a>(s: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = s[pos..].find(&open) {
        let start = pos + i + open.len();
        match s[start..].find(&close) {
            Some(j) => {
                out.push(&s[start..start + j]);
                pos = start + j + close.len();
            }
            None => break,
        }
    }
    out
}

/// Unescaped text of the first `<tag>`.
pub fn tag(s: &str, tag_name: &str) -> Option<String> {
    tags(s, tag_name).first().map(|t| unescape(t))
}

pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let Some(end) = rest.find(';') else {
            break;
        };
        let ent = &rest[1..end];
        let rep = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ if ent.starts_with("#x") => u32::from_str_radix(&ent[2..], 16).ok().and_then(char::from_u32),
            _ if ent.starts_with('#') => ent[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match rep {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_list_response() {
        let x = r#"<?xml version="1.0"?><ListBucketResult><IsTruncated>true</IsTruncated>
<Contents><Key>a/b&amp;c</Key><Size>12</Size></Contents><Contents><Key>a/d</Key><Size>3</Size></Contents>
<CommonPrefixes><Prefix>a/x/</Prefix></CommonPrefixes><NextContinuationToken>tok==</NextContinuationToken></ListBucketResult>"#;
        let c = tags(x, "Contents");
        assert_eq!(c.len(), 2);
        assert_eq!(tag(c[0], "Key").unwrap(), "a/b&c");
        assert_eq!(tag(c[1], "Size").unwrap(), "3");
        assert_eq!(tag(x, "NextContinuationToken").unwrap(), "tok==");
        assert_eq!(tag(tags(x, "CommonPrefixes")[0], "Prefix").unwrap(), "a/x/");
        assert_eq!(unescape("&#65;&#x42;&lt;"), "AB<");
        assert_eq!(escape("\"a&b\""), "&quot;a&amp;b&quot;");
    }
}
