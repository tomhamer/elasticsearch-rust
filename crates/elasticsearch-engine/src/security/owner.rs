//! Identities bound to long-lived resources.
//!
//! * [`ResourceOwner`]: who created a retained result (async search,
//!   ES|QL / SQL / EQL async results, SQL cursors). ES stores the creating
//!   request's security headers with the result and only hands it back to a
//!   caller that `Authentication.canAccessResourcesOf` the creator
//!   (`AsyncTaskIndexService`, `AsyncSearchSecurity`).
//! * [`StoredAuthn`]: the authentication a background job runs as (ES
//!   persists the `_xpack_security_authentication` header with a watch and
//!   with the job, and restores it for every execution): the subject, its
//!   realm, role names, API key id and run-as target, serializable so it
//!   survives restarts and travels to other nodes.

use super::{Authn, Principal};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The subject that owns a retained result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceOwner {
    /// Effective (run-as target, if any) user name.
    pub username: String,
    pub realm_name: String,
    pub realm_type: String,
    /// The authenticating API key (API keys own their own results).
    #[serde(default)]
    pub api_key_id: Option<String>,
}

impl ResourceOwner {
    pub fn of(a: &Authn) -> ResourceOwner {
        let e = a.effective();
        ResourceOwner {
            username: e.username.clone(),
            realm_name: e.realm_name.clone(),
            realm_type: e.realm_type.clone(),
            api_key_id: if a.run_as.is_none() { a.api_key_id().map(str::to_string) } else { None },
        }
    }

    /// ES `Authentication.canAccessResourcesOf`: two API key authentications
    /// match on the key id only; an API key never matches a user; users
    /// match on the effective principal and its realm (file and native
    /// realms by type, as ES does, others by name and type).
    pub fn can_access(&self, creator: &ResourceOwner) -> bool {
        match (&self.api_key_id, &creator.api_key_id) {
            (Some(mine), Some(theirs)) => mine == theirs,
            (None, None) => {
                if self.username != creator.username || self.realm_type != creator.realm_type {
                    return false;
                }
                matches!(self.realm_type.as_str(), "file" | "native") || self.realm_name == creator.realm_name
            }
            _ => false,
        }
    }
}

/// An API key as [`StoredAuthn`] records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredApiKey {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub metadata: Value,
}

/// A serializable [`Authn`]: what a background execution runs as.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredAuthn {
    pub username: String,
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub full_name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub metadata: Value,
    pub realm_name: String,
    pub realm_type: String,
    pub auth_type: String,
    #[serde(default)]
    pub api_key: Option<StoredApiKey>,
    /// Service account token `(name, source)`.
    #[serde(default)]
    pub service_token: Option<(String, String)>,
    #[serde(default)]
    pub anonymous: bool,
    #[serde(default)]
    pub run_as: Option<Box<StoredAuthn>>,
    /// A remote cluster's user over API key based remote cluster security:
    /// the cross-cluster API key id and the granted role descriptors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_cluster: Option<(String, Vec<Value>)>,
}

impl StoredAuthn {
    pub fn of(a: &Authn) -> StoredAuthn {
        let (api_key, service_token, anonymous) = match &a.principal {
            Principal::ApiKey { id, name, metadata } => (Some(StoredApiKey { id: id.clone(), name: name.clone(), metadata: metadata.clone() }), None, false),
            Principal::ServiceAccount { token_name, source } => (None, Some((token_name.clone(), source.to_string())), false),
            Principal::Anonymous => (None, None, true),
            Principal::User | Principal::CrossCluster { .. } => (None, None, false),
        };
        let cross_cluster = match &a.principal {
            Principal::CrossCluster { key_id, descriptors } => Some((key_id.clone(), (**descriptors).clone())),
            _ => None,
        };
        StoredAuthn {
            username: a.username.clone(),
            roles: a.roles.clone(),
            full_name: a.full_name.clone(),
            email: a.email.clone(),
            metadata: a.metadata.clone(),
            realm_name: a.realm_name.clone(),
            realm_type: a.realm_type.clone(),
            auth_type: a.auth_type.to_string(),
            api_key,
            service_token,
            anonymous,
            run_as: a.run_as.as_deref().map(|r| Box::new(StoredAuthn::of(r))),
            cross_cluster,
        }
    }

    /// The authentication to evaluate privileges with (roles resolve at
    /// execution time, as ES resolves a stored authentication's roles).
    pub fn to_authn(&self) -> Authn {
        let principal = if let Some((key_id, descriptors)) = &self.cross_cluster {
            Principal::CrossCluster { key_id: key_id.clone(), descriptors: std::sync::Arc::new(descriptors.clone()) }
        } else if let Some(k) = &self.api_key {
            Principal::ApiKey { id: k.id.clone(), name: k.name.clone(), metadata: k.metadata.clone() }
        } else if let Some((name, source)) = &self.service_token {
            Principal::ServiceAccount { token_name: name.clone(), source: if source == "file" { "file" } else { "index" } }
        } else if self.anonymous {
            Principal::Anonymous
        } else {
            Principal::User
        };
        let auth_type: &'static str = match self.auth_type.as_str() {
            "api_key" => "api_key",
            "token" => "token",
            "anonymous" => "anonymous",
            "internal" => "internal",
            _ => "realm",
        };
        Authn {
            username: self.username.clone(),
            roles: self.roles.clone(),
            full_name: self.full_name.clone(),
            email: self.email.clone(),
            metadata: if self.metadata.is_null() { json!({}) } else { self.metadata.clone() },
            enabled: true,
            realm_name: self.realm_name.clone(),
            realm_type: self.realm_type.clone(),
            auth_type,
            principal,
            run_as: self.run_as.as_deref().map(|r| Box::new(r.to_authn())),
        }
    }

    /// The effective user name (run-as target, if any).
    pub fn principal(&self) -> &str {
        self.run_as.as_deref().map(|r| r.username.as_str()).unwrap_or(&self.username)
    }

    /// Base64 of the JSON form (the `_xpack_security_authentication` header value).
    pub fn encode(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(serde_json::to_vec(self).unwrap_or_default())
    }

    pub fn decode(s: &str) -> Option<StoredAuthn> {
        use base64::Engine;
        let b = base64::engine::general_purpose::STANDARD.decode(s.trim()).ok()?;
        serde_json::from_slice(&b).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(name: &str, realm: &str, ty: &str) -> Authn {
        Authn {
            username: name.into(),
            roles: vec!["r1".into()],
            full_name: None,
            email: None,
            metadata: json!({}),
            enabled: true,
            realm_name: realm.into(),
            realm_type: ty.into(),
            auth_type: "realm",
            principal: Principal::User,
            run_as: None,
        }
    }

    fn key(id: &str, owner: &str) -> Authn {
        let mut a = user(owner, "_es_api_key", "_es_api_key");
        a.auth_type = "api_key";
        a.principal = Principal::ApiKey { id: id.into(), name: "k".into(), metadata: json!({}) };
        a
    }

    #[test]
    fn users_match_on_principal_and_realm() {
        let alice = ResourceOwner::of(&user("alice", "native1", "native"));
        assert!(alice.can_access(&ResourceOwner::of(&user("alice", "native1", "native"))));
        // Native and file realms compare by type (ES: they may be renamed).
        assert!(alice.can_access(&ResourceOwner::of(&user("alice", "native2", "native"))));
        assert!(!alice.can_access(&ResourceOwner::of(&user("bob", "native1", "native"))));
        assert!(!alice.can_access(&ResourceOwner::of(&user("alice", "file1", "file"))));
        let ldap = ResourceOwner::of(&user("alice", "ldap1", "ldap"));
        assert!(!ldap.can_access(&ResourceOwner::of(&user("alice", "ldap2", "ldap"))));
    }

    #[test]
    fn api_keys_match_on_key_id_only() {
        let k1 = ResourceOwner::of(&key("k1", "alice"));
        assert!(k1.can_access(&ResourceOwner::of(&key("k1", "alice"))));
        assert!(!k1.can_access(&ResourceOwner::of(&key("k2", "alice"))));
        // A key never reads its owner's results, nor the owner the key's.
        assert!(!k1.can_access(&ResourceOwner::of(&user("alice", "native1", "native"))));
        assert!(!ResourceOwner::of(&user("alice", "native1", "native")).can_access(&k1));
    }

    #[test]
    fn run_as_uses_the_effective_user() {
        let mut a = user("admin", "file1", "file");
        a.run_as = Some(Box::new(user("alice", "native1", "native")));
        let o = ResourceOwner::of(&a);
        assert_eq!(o.username, "alice");
        assert!(o.can_access(&ResourceOwner::of(&user("alice", "native1", "native"))));
        assert!(!o.can_access(&ResourceOwner::of(&user("admin", "file1", "file"))));
    }

    #[test]
    fn stored_authn_round_trips() {
        let mut a = key("k9", "carol");
        a.run_as = None;
        let s = StoredAuthn::of(&a);
        let back = StoredAuthn::decode(&s.encode()).unwrap();
        assert_eq!(back, s);
        let b = back.to_authn();
        assert_eq!(b.api_key_id(), Some("k9"));
        assert_eq!(b.auth_type, "api_key");
        let mut u = user("dave", "native1", "native");
        u.run_as = Some(Box::new(user("erin", "file1", "file")));
        let s = StoredAuthn::of(&u);
        assert_eq!(s.principal(), "erin");
        let b = StoredAuthn::decode(&s.encode()).unwrap().to_authn();
        assert_eq!(b.effective().username, "erin");
        assert_eq!(b.roles, vec!["r1".to_string()]);
        assert!(StoredAuthn::decode("not base64 json").is_none());
    }
}
