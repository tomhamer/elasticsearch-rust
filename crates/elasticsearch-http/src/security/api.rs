//! `/_security/*` REST APIs (security enabled only).

use super::query;
use super::{check_cluster, denied};
use crate::{blocking, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::{Node, NODE_ID};
use elasticsearch_engine::security::hash::{self, Hasher};
use elasticsearch_engine::security::{
    self as sec, now_ms, parse_time_ms, privilege, reserved_user_roles, roles, service_account, ApiKey, Authn, NativeUser, Permission, Principal,
    ReservedState, ServiceToken, RESERVED_USERS, SERVICE_ACCOUNTS,
};
use serde_json::{json, Map, Value};
use std::sync::Arc;

const P: &str = "cluster:admin/xpack/security";

pub async fn handle(node: Arc<Node>, req: Req, authn: Authn) -> R {
    blocking(move || {
        let perm = node.security.permission(&authn);
        let ctx = Ctx { node: &node, req: &req, a: &authn, perm: &perm };
        ctx.route()
    })
    .await
}

struct Ctx<'a> {
    node: &'a Node,
    req: &'a Req,
    a: &'a Authn,
    perm: &'a Permission,
}

fn validation(msg: &str) -> EsError {
    EsError::action_validation(&[msg])
}

fn license_error(feature: &str) -> EsError {
    EsError::new(403, "security_exception", format!("current license is non-compliant for [{feature}]")).with("license.expired.feature", feature)
}

fn parse_error(msg: String) -> EsError {
    EsError::new(400, "parse_exception", msg)
}

fn names_of(s: &str) -> Vec<String> {
    s.split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::to_string).collect()
}

fn str_list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        _ => vec![],
    }
}

fn check_metadata(md: &Value) -> Result<(), EsError> {
    if let Some(o) = md.as_object() {
        if o.keys().any(|k| k.starts_with('_')) {
            return Err(validation("metadata keys may not start with [_]"));
        }
    }
    Ok(())
}

impl<'a> Ctx<'a> {
    fn seg(&self, i: usize) -> &str {
        self.req.seg(i)
    }

    fn need(&self, action: &str) -> Result<(), EsError> {
        check_cluster(self.node, self.a, self.perm, action)
    }

    /// Cluster privilege, or `global.application.manage` covering every app.
    fn need_apps(&self, action: &str, apps: &[String]) -> Result<(), EsError> {
        if self.perm.check_cluster(action) {
            return Ok(());
        }
        let covers = |r: &roles::Role| !apps.is_empty() && apps.iter().all(|a| r.manage_apps.iter().any(|p| privilege::glob(p, a)));
        if covers(&self.perm.role) && self.perm.limit.as_ref().map(|l| covers(l)).unwrap_or(true) {
            return Ok(());
        }
        Err(denied(self.node, self.a, self.perm, action, None))
    }

    fn nodes_response(&self) -> Value {
        json!({"_nodes": {"total": 1, "successful": 1, "failed": 0}, "cluster_name": self.node.cluster_name, "nodes": {NODE_ID: {"name": self.node.name}}})
    }

    fn body(&self) -> Result<Value, EsError> {
        self.req.json()
    }

    fn route(&self) -> R {
        let m = &self.req.method;
        let (s1, s2, s3) = (self.seg(1).to_string(), self.seg(2).to_string(), self.seg(3).to_string());
        let n = self.req.segs.len();
        let get = *m == Method::GET || *m == Method::HEAD;
        let post = *m == Method::POST;
        let put = *m == Method::PUT;
        let del = *m == Method::DELETE;
        match s1.as_str() {
            "_authenticate" if get => Ok(Resp::ok(self.a.to_json())),
            // `RestSecurityStatsAction` (9.2+): per-node role store stats.
            "stats" if get && n == 2 => {
                self.need("cluster:monitor/xpack/security/stats")?;
                let dls = json!({"count": 0, "memory": "0b", "memory_in_bytes": 0, "hits": 0, "misses": 0, "evictions": 0, "hits_time_in_millis": 0, "misses_time_in_millis": 0});
                Ok(Resp::ok(json!({"nodes": {elasticsearch_engine::node::NODE_ID: {"roles": {"dls": {"bit_set_cache": dls}}}}})))
            }
            "_query" => match s2.as_str() {
                "user" if get || post => self.query_users(),
                "api_key" if get || post => self.query_api_keys(),
                "role" if get || post => self.query_roles(),
                _ => crate::not_found_handler(self.req),
            },
            "user" => match (n, s2.as_str(), s3.as_str()) {
                (2, _, _) if get => self.get_users(None),
                (3, "_privileges", _) if get => self.get_user_privileges(),
                (3, "_has_privileges", _) if get || post => self.has_privileges(None),
                (3, "_password", _) if put || post => self.change_password(None),
                (3, name, _) if put || post => self.put_user(name),
                (3, name, _) if get => self.get_users(Some(name)),
                (3, name, _) if del => self.delete_user(name),
                (4, name, "_password") if put || post => self.change_password(Some(name)),
                (4, name, "_enable") if put || post => self.set_enabled(name, true),
                (4, name, "_disable") if put || post => self.set_enabled(name, false),
                (4, name, "_has_privileges") if get || post => self.has_privileges(Some(name)),
                _ => crate::not_found_handler(self.req),
            },
            "role" => match (n, s2.as_str(), s3.as_str()) {
                (2, _, _) if get => self.get_roles(None),
                (2, _, _) if post => self.bulk_put_roles(),
                (2, _, _) if del => self.bulk_delete_roles(),
                (3, name, _) if put || post => self.put_role(name),
                (3, name, _) if get => self.get_roles(Some(name)),
                (3, name, _) if del => self.delete_role(name),
                (4, _, "_clear_cache") if post => {
                    self.need(&format!("{P}/roles/cache/clear"))?;
                    Ok(Resp::ok(self.nodes_response()))
                }
                _ => crate::not_found_handler(self.req),
            },
            "role_mapping" => match (n, s2.as_str()) {
                (2, _) if get => self.get_role_mappings(None),
                (3, name) if put || post => self.put_role_mapping(name),
                (3, name) if get => self.get_role_mappings(Some(name)),
                (3, name) if del => self.delete_role_mapping(name),
                _ => crate::not_found_handler(self.req),
            },
            "privilege" => match (n, s2.as_str(), s3.as_str()) {
                (3, "_builtin", _) if get => self.builtin_privileges(),
                (2, _, _) if put || post => self.put_privileges(),
                (2, _, _) if get => self.get_privileges(None, None),
                (3, app, _) if get => self.get_privileges(Some(app), None),
                (4, app, _) if get => self.get_privileges(Some(app), Some(&s3)),
                (4, app, names) if del => self.delete_privileges(app, names),
                (4, _, "_clear_cache") if post => {
                    self.need(&format!("{P}/privilege/cache/clear"))?;
                    Ok(Resp::ok(self.nodes_response()))
                }
                _ => crate::not_found_handler(self.req),
            },
            "api_key" => match (n, s2.as_str(), s3.as_str()) {
                (2, _, _) if put || post => self.create_api_key(),
                (2, _, _) if get => self.get_api_keys(),
                (2, _, _) if del => self.invalidate_api_keys(),
                (3, "grant", _) if post => self.grant_api_key(),
                (3, "clone", _) if post || put => self.clone_api_key(),
                (3, "_bulk_update", _) if post => self.bulk_update_api_keys(),
                (3, id, _) if put => self.update_api_key(id),
                (4, _, "_clear_cache") if post => {
                    self.need(&format!("{P}/api_key/clear_cache"))?;
                    Ok(Resp::ok(self.nodes_response()))
                }
                _ => crate::not_found_handler(self.req),
            },
            "cross_cluster" => match (n, s2.as_str()) {
                (3, "api_key") if post => self.create_cross_cluster_key(),
                (4, "api_key") if put => self.update_cross_cluster_key(&s3),
                _ => crate::not_found_handler(self.req),
            },
            "oauth2" if s2 == "token" => {
                if post {
                    self.create_token()
                } else if del {
                    self.invalidate_token()
                } else {
                    crate::not_found_handler(self.req)
                }
            }
            "service" => self.service(),
            "realm" if s3 == "_clear_cache" && post => {
                self.need(&format!("{P}/realm/cache/clear"))?;
                Ok(Resp::ok(self.nodes_response()))
            }
            "profile" => self.profile(),
            "settings" => {
                if get {
                    self.need(&format!("{P}/settings/get"))?;
                    Ok(Resp::ok(self.security_settings()))
                } else if put {
                    self.need(&format!("{P}/settings/update"))?;
                    self.update_security_settings()
                } else {
                    crate::not_found_handler(self.req)
                }
            }
            "enroll" if get && n == 3 && matches!(s2.as_str(), "node" | "kibana") => self.enroll(&s2),
            "enroll" => crate::not_found_handler(self.req),
            "saml" => Err(license_error("saml")),
            "oidc" => Err(license_error("oidc")),
            "delegate_pki" => Err(license_error("pki")),
            _ => crate::not_found_handler(self.req),
        }
    }

    // --------------------------------------------------------- enrollment

    /// `GET /_security/enroll/node` (`RestNodeEnrollmentAction`) and
    /// `GET /_security/enroll/kibana` (`RestKibanaEnrollAction`): ES's
    /// `EnrollmentBaseRestHandler` refuses them unless
    /// `xpack.security.enrollment.enabled`, then the transport action is
    /// authorized (`cluster:admin/xpack/security/enroll/{node,kibana}`, the
    /// only privilege of an enrollment token's API key).
    fn enroll(&self, kind: &str) -> R {
        let enrollment = self.req.extensions.get::<crate::Enrollment>().filter(|e| e.enabled).cloned().ok_or_else(|| {
            EsError::new(403, "security_exception", "Enrollment mode is not enabled. Set [xpack.security.enrollment.enabled] to true, in order to use this API.")
        })?;
        self.need(&format!("{P}/enroll/{kind}"))?;
        if kind == "node" {
            return Ok(Resp::ok((enrollment.node)()?));
        }
        let http_ca = (enrollment.http_ca)()?;
        // ES creates the token as the security origin: no further privilege.
        let account = "elastic/kibana";
        let name = format!("enroll-process-token-{}", now_ms());
        let secret = hash::random_b64url(16);
        let sec = &self.node.security;
        {
            let mut st = sec.state.write();
            let tokens = st.service_tokens.entry(account.to_string()).or_default();
            if tokens.contains_key(&name) {
                let cause = EsError::new(409, "version_conflict_engine_exception", format!("[service_account_token-{account}/{name}]: version conflict, document already exists (current version [1])"));
                return Err(EsError::new(500, "exception", "Failed to create token for the [elastic/kibana] service account").caused_by(cause));
            }
            tokens.insert(name.clone(), ServiceToken { hash: Hasher::Ssha256.hash(&secret), creation: now_ms() });
        }
        sec.save().map_err(|e| EsError::new(500, "exception", "Failed to create token for the [elastic/kibana] service account").caused_by(e))?;
        Ok(Resp::ok(json!({"token": {"name": name, "value": sec::service_token_value(account, &name, &secret)}, "http_ca": http_ca})))
    }

    // ---------------------------------------------------------- settings

    fn security_settings(&self) -> Value {
        let st = self.node.security.state.read();
        let mut out = Map::new();
        for k in ["security", "security-tokens", "security-profile"] {
            let flat = st.index_settings.get(k).cloned().unwrap_or_else(|| {
                let mut m = Map::new();
                m.insert("index.auto_expand_replicas".into(), json!("0-1"));
                m.insert("index.number_of_replicas".into(), json!("0"));
                m
            });
            let mut idx = Map::new();
            for (fk, v) in flat {
                idx.insert(fk.trim_start_matches("index.").to_string(), v);
            }
            out.insert(k.to_string(), json!({ "index": idx }));
        }
        Value::Object(out)
    }

    fn update_security_settings(&self) -> R {
        let body = self.body()?;
        let sec = &self.node.security;
        let mut st = sec.state.write();
        for (k, v) in body.as_object().into_iter().flatten() {
            if !matches!(k.as_str(), "security" | "security-tokens" | "security-profile") {
                return Err(EsError::illegal_argument(format!("unknown security index [{k}]")));
            }
            let entry = st.index_settings.entry(k.clone()).or_insert_with(|| {
                let mut m = Map::new();
                m.insert("index.auto_expand_replicas".into(), json!("0-1"));
                m.insert("index.number_of_replicas".into(), json!("0"));
                m
            });
            let flat = elasticsearch_engine::settings::flatten(v);
            for (fk, fv) in flat {
                let key = if fk.starts_with("index.") { fk.clone() } else { format!("index.{fk}") };
                match fv {
                    Value::Null => {
                        entry.remove(&key);
                    }
                    Value::String(x) => {
                        entry.insert(key, json!(x));
                    }
                    other => {
                        entry.insert(key, json!(other.to_string()));
                    }
                }
            }
        }
        drop(st);
        sec.save()?;
        Ok(Resp::ack())
    }

    fn profile_uid(&self, username: &str) -> Option<String> {
        let st = self.node.security.state.read();
        st.profiles.iter().find(|(_, p)| p["user"]["username"] == json!(username)).map(|(k, _)| k.clone())
    }

    // ------------------------------------------------------------- users

    fn user_json(&self, name: &str) -> Option<Value> {
        let sec = &self.node.security;
        if let Some(roles) = reserved_user_roles(name) {
            let st = sec.state.read();
            let rs = st.reserved.get(name).cloned().unwrap_or_default();
            let md = sec.lookup_user(name).map(|u| u.metadata).unwrap_or_else(|| json!({"_reserved": true}));
            return Some(json!({"username": name, "roles": roles, "full_name": null, "email": null, "metadata": md, "enabled": rs.enabled}));
        }
        let st = sec.state.read();
        st.users.get(name).map(|u| {
            json!({"username": name, "roles": u.roles, "full_name": u.full_name, "email": u.email,
                   "metadata": if u.metadata.is_null() { json!({}) } else { u.metadata.clone() }, "enabled": u.enabled})
        })
    }

    fn get_users(&self, names: Option<&str>) -> R {
        self.need(&format!("{P}/user/get"))?;
        let with_uid = self.req.flag("with_profile_uid")?;
        let r = self.get_users_inner(names)?;
        if !with_uid {
            return Ok(r);
        }
        let mut r = r;
        if let crate::RespBody::Json(Value::Object(m)) = &mut r.body {
            for (n, u) in m.iter_mut() {
                if let Some(uid) = self.profile_uid(n) {
                    u["profile_uid"] = json!(uid);
                }
            }
        }
        Ok(r)
    }

    fn get_users_inner(&self, names: Option<&str>) -> R {
        let mut out = Map::new();
        match names {
            None => {
                for (n, _) in RESERVED_USERS {
                    if let Some(v) = self.user_json(n) {
                        out.insert(n.to_string(), v);
                    }
                }
                let users: Vec<String> = self.node.security.state.read().users.keys().cloned().collect();
                for n in users {
                    if let Some(v) = self.user_json(&n) {
                        out.insert(n, v);
                    }
                }
            }
            Some(list) => {
                for n in names_of(list) {
                    if let Some(v) = self.user_json(&n) {
                        out.insert(n, v);
                    }
                }
                if out.is_empty() {
                    return Ok(Resp::json(404, json!({})));
                }
            }
        }
        Ok(Resp::ok(Value::Object(out)))
    }

    fn mirror(&self, id: &str, doc: Option<Value>) -> Result<(), EsError> {
        if let Some(writer) = self.req.extensions.get::<crate::NativeSecurityMirror>() {
            return (writer.0)(id, doc);
        }
        // Keep `.security-7` (alias `.security`) in step for direct reads.
        let node = self.node;
        let svc = match node.get_index(".security-7") {
            Some(s) => s,
            None => {
                if doc.is_none() {
                    return Ok(());
                }
                let body = json!({"settings": {"index": {"hidden": true, "number_of_shards": 1, "number_of_replicas": 0, "auto_expand_replicas": "0-1"}},
                                  "aliases": {".security": {"is_hidden": true}}});
                match node.create_index(".security-7", &body) {
                    Ok(s) => s,
                    Err(error) => return Err(error),
                }
            }
        };
        let w = match doc {
            Some(d) => WriteReq::Index { id: Some(id.to_string()), source: d.to_string(), routing: None, cond: Default::default() },
            None => WriteReq::Delete { id: id.to_string(), routing: None, cond: Default::default() },
        };
        svc.write(vec![w], false).remove(0)?;
        // Put/delete user default to `refresh=true` in ES
        // (`PutUserRequest` / `DeleteUserRequest`: `RefreshPolicy.IMMEDIATE`).
        svc.refresh_doc_shard(id, None)?;
        Ok(())
    }

    fn mirror_user(&self, name: &str) -> Result<(), EsError> {
        let user = self.node.security.state.read().users.get(name).cloned();
        if let Some(user) = user {
            self.mirror(&format!("user-{name}"), Some(json!({"username":name,"password":user.password_hash,
                "roles":user.roles,"full_name":user.full_name,"email":user.email,"metadata":user.metadata,
                "enabled":user.enabled,"type":"user"})))?;
        }
        Ok(())
    }

    fn put_user(&self, name: &str) -> R {
        self.need(&format!("{P}/user/put"))?;
        if reserved_user_roles(name).is_some() || name == self.node.security.cfg.read().anonymous_username {
            return Err(validation(&format!("Username [{name}] is reserved and may not be used.")));
        }
        validate_name("User", name, 507)?;
        let body = self.body()?;
        let obj = body.as_object().cloned().unwrap_or_default();
        let hasher = self.node.security.cfg.read().password_hashing;
        let mut password_hash: Option<String> = None;
        let mut roles_v: Option<Vec<String>> = None;
        let (mut full_name, mut email, mut metadata, mut enabled): (Option<Value>, Option<Value>, Option<Value>, Option<bool>) = (None, None, None, None);
        for (k, v) in &obj {
            match k.as_str() {
                "username" => {
                    if v.as_str() != Some(name) {
                        return Err(EsError::illegal_argument(format!("[username] in source does not match the username provided [{name}]")));
                    }
                }
                "password" => {
                    let p = v.as_str().ok_or_else(|| parse_error(format!("expected field [password] to be of type string, but found [{}] instead", token_name(v))))?;
                    if p.chars().count() < 6 {
                        return Err(validation("passwords must be at least [6] characters long"));
                    }
                    if password_hash.is_some() {
                        return Err(validation("password_hash has already been set"));
                    }
                    password_hash = Some(hasher.hash(p));
                }
                "password_hash" => {
                    let h = v.as_str().unwrap_or("");
                    let resolved = Hasher::from_hash(h);
                    if resolved != hasher && !resolved.is_stored_password_algo() {
                        return Err(EsError::illegal_argument(format!(
                            "The provided password hash is not a hash or it could not be resolved to a supported hash algorithm. The supported password hash algorithms are {}",
                            hash::stored_algos_display()
                        )));
                    }
                    if password_hash.is_some() {
                        return Err(validation("password_hash has already been set"));
                    }
                    password_hash = Some(h.to_string());
                }
                "roles" => roles_v = Some(str_list(v)),
                "full_name" => full_name = Some(v.clone()),
                "email" => email = Some(v.clone()),
                "metadata" => metadata = Some(v.clone()),
                "enabled" => enabled = v.as_bool(),
                other => return Err(parse_error(format!("failed to parse add user request. unexpected field [{other}]"))),
            }
        }
        let roles_v = roles_v.ok_or_else(|| validation("roles are missing"))?;
        if let Some(md) = &metadata {
            check_metadata(md)?;
        }
        let sec = &self.node.security;
        let mut st = sec.state.write();
        let existing = st.users.get(name).cloned();
        let created = existing.is_none();
        let password_hash = match (password_hash, &existing) {
            (Some(h), _) => h,
            (None, Some(e)) => e.password_hash.clone(),
            (None, None) => return Err(validation("password must be specified unless you are updating an existing user")),
        };
        let opt_str = |v: Option<Value>, old: Option<String>| -> Option<String> {
            match v {
                Some(Value::String(s)) => Some(s),
                Some(_) => None,
                None => old,
            }
        };
        let user = NativeUser {
            password_hash,
            roles: roles_v,
            full_name: opt_str(full_name, existing.as_ref().and_then(|e| e.full_name.clone())),
            email: opt_str(email, existing.as_ref().and_then(|e| e.email.clone())),
            metadata: metadata.unwrap_or_else(|| existing.as_ref().map(|e| e.metadata.clone()).unwrap_or_else(|| json!({}))),
            enabled: enabled.unwrap_or_else(|| existing.as_ref().map(|e| e.enabled).unwrap_or(true)),
        };
        st.users.insert(name.to_string(), user.clone());
        drop(st);
        sec.save()?;
        self.mirror(
            &format!("user-{name}"),
            Some(json!({"username": name, "password": user.password_hash, "roles": user.roles, "full_name": user.full_name, "email": user.email,
                        "metadata": user.metadata, "enabled": user.enabled, "type": "user"})),
        )?;
        Ok(Resp::ok(json!({"created": created})))
    }

    fn delete_user(&self, name: &str) -> R {
        self.need(&format!("{P}/user/delete"))?;
        if reserved_user_roles(name).is_some() {
            return Err(EsError::illegal_argument(format!("user [{name}] is reserved and cannot be deleted")));
        }
        let sec = &self.node.security;
        let found = sec.state.write().users.remove(name).is_some();
        if found {
            sec.save()?;
            self.mirror(&format!("user-{name}"), None)?;
        }
        Ok(Resp::json(if found { 200 } else { 404 }, json!({"found": found})))
    }

    fn change_password(&self, name: Option<&str>) -> R {
        let me = self.a.effective();
        let target = name.unwrap_or(&me.username).to_string();
        let action = format!("{P}/user/change_password");
        let same_user = target == me.username && self.a.auth_type == "realm" && matches!(me.realm_type.as_str(), "native" | "reserved") && self.a.run_as.is_none();
        if !same_user {
            self.need(&action)?;
        }
        let body = self.body()?;
        let hasher = self.node.security.cfg.read().password_hashing;
        let mut new_hash: Option<String> = None;
        for (k, v) in body.as_object().into_iter().flatten() {
            match k.as_str() {
                "password" => {
                    let p = v.as_str().unwrap_or("");
                    if p.chars().count() < 6 {
                        return Err(validation("passwords must be at least [6] characters long"));
                    }
                    new_hash = Some(hasher.hash(p));
                }
                "password_hash" => {
                    let h = v.as_str().unwrap_or("");
                    let resolved = Hasher::from_hash(h);
                    if resolved != hasher && !resolved.is_stored_password_algo() {
                        return Err(EsError::illegal_argument(format!(
                            "The provided password hash is not a hash or it could not be resolved to a supported hash algorithm. The supported password hash algorithms are {}",
                            hash::stored_algos_display()
                        )));
                    }
                    new_hash = Some(h.to_string());
                }
                other => return Err(parse_error(format!("failed to parse change password request. unexpected field [{other}]"))),
            }
        }
        let new_hash = new_hash.ok_or_else(|| validation("password is missing"))?;
        let sec = &self.node.security;
        let mut st = sec.state.write();
        if reserved_user_roles(&target).is_some() {
            st.reserved.entry(target.clone()).or_insert_with(|| ReservedState { password_hash: None, enabled: true }).password_hash = Some(new_hash);
        } else if let Some(u) = st.users.get_mut(&target) {
            u.password_hash = new_hash;
        } else if sec.cfg.read().file_users.contains_key(&target) {
            return Err(validation(&format!("user [{target}] does not belong to the native realm")));
        } else {
            return Err(validation("user must exist in order to change password"));
        }
        drop(st);
        sec.save()?;
        self.mirror_user(&target)?;
        Ok(Resp::ok(json!({})))
    }

    fn set_enabled(&self, name: &str, enabled: bool) -> R {
        self.need(&format!("{P}/user/set_enabled"))?;
        if name == self.a.effective().username {
            return Err(validation("users may not update the enabled status of their own account"));
        }
        let sec = &self.node.security;
        let mut st = sec.state.write();
        if reserved_user_roles(name).is_some() {
            st.reserved.entry(name.to_string()).or_insert_with(|| ReservedState { password_hash: None, enabled: true }).enabled = enabled;
        } else if let Some(u) = st.users.get_mut(name) {
            u.enabled = enabled;
        } else {
            return Err(EsError::resource_not_found(format!("could not find user [{name}] to {}", if enabled { "enable" } else { "disable" })));
        }
        drop(st);
        sec.save()?;
        self.mirror_user(name)?;
        Ok(Resp::ok(json!({})))
    }

    fn query_users(&self) -> R {
        self.need(&format!("{P}/user/query"))?;
        let body = self.body()?;
        // Native users only (the query runs over the security index).
        let names: Vec<String> = self.node.security.state.read().users.keys().cloned().collect();
        let with_uid = self.req.flag("with_profile_uid")?;
        let docs: Vec<Value> = names
            .iter()
            .filter_map(|n| {
                let mut u = self.user_json(n)?;
                if with_uid {
                    if let Some(uid) = self.profile_uid(n) {
                        u["profile_uid"] = json!(uid);
                    }
                }
                Some(u)
            })
            .collect();
        let map = |f: &str| -> String { f.to_string() };
        let page = query::run(docs, &body, &map)?;
        let users: Vec<Value> = page
            .hits
            .into_iter()
            .map(|(mut d, s)| {
                if let Some(s) = s {
                    d["_sort"] = json!(s);
                }
                d
            })
            .collect();
        Ok(Resp::ok(json!({"total": page.total, "count": users.len(), "users": users})))
    }

    // ------------------------------------------------------- privileges

    fn get_user_privileges(&self) -> R {
        let descs: Vec<Value> = match &self.a.principal {
            Principal::ApiKey { id, .. } => {
                let st = self.node.security.state.read();
                st.api_keys.get(id).map(|k| if k.role_descriptors.is_empty() { k.limited_by.values().cloned().collect() } else { k.role_descriptors.values().cloned().collect() }).unwrap_or_default()
            }
            Principal::ServiceAccount { .. } => service_account(&self.a.username).into_iter().collect(),
            _ => self.node.security.assigned_roles(self.a).iter().filter_map(|n| self.node.security.role_descriptor(n)).collect(),
        };
        let mut cluster: Vec<String> = Vec::new();
        let mut global: Vec<Value> = Vec::new();
        let mut indices: Vec<(Vec<String>, bool, Vec<Value>, Vec<Value>, Vec<String>)> = Vec::new();
        let mut apps: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
        let mut run_as: Vec<String> = Vec::new();
        let mut remote_indices: Vec<Value> = Vec::new();
        let mut remote_cluster: Vec<Value> = Vec::new();
        for d in &descs {
            for g in d["remote_indices"].as_array().into_iter().flatten() {
                let mut g = g.clone();
                if let Some(o) = g.as_object_mut() {
                    for k in ["names", "privileges", "clusters"] {
                        if let Some(Value::Array(a)) = o.get_mut(k) {
                            a.sort_by(|x, y| x.as_str().cmp(&y.as_str()));
                        }
                    }
                }
                remote_indices.push(g);
            }
            for g in d["remote_cluster"].as_array().into_iter().flatten() {
                remote_cluster.push(g.clone());
            }
            for c in str_list(&d["cluster"]) {
                if !cluster.contains(&c) {
                    cluster.push(c);
                }
            }
            if !d["global"].is_null() && !global.contains(&d["global"]) {
                global.push(d["global"].clone());
            }
            for g in d["indices"].as_array().into_iter().flatten() {
                let mut names = str_list(&g["names"]);
                names.sort();
                let restricted = g["allow_restricted_indices"].as_bool().unwrap_or(false);
                let fls: Vec<Value> = if g["field_security"].is_null() { vec![] } else { vec![g["field_security"].clone()] };
                let q: Vec<Value> = if g["query"].is_null() { vec![] } else { vec![g["query"].clone()] };
                let privs = str_list(&g["privileges"]);
                // Groups on the same names merge: privileges union, field_security / query accumulate.
                match indices.iter_mut().find(|e| e.0 == names && e.1 == restricted) {
                    Some(e) => {
                        for p in privs {
                            if !e.4.contains(&p) {
                                e.4.push(p);
                            }
                        }
                        for f in fls {
                            if !e.2.contains(&f) {
                                e.2.push(f);
                            }
                        }
                        for x in q {
                            if !e.3.contains(&x) {
                                e.3.push(x);
                            }
                        }
                    }
                    None => indices.push((names, restricted, fls, q, privs)),
                }
            }
            for ap in d["applications"].as_array().into_iter().flatten() {
                let app = ap["application"].as_str().unwrap_or("").to_string();
                let mut res = str_list(&ap["resources"]);
                res.sort();
                let privs = str_list(&ap["privileges"]);
                match apps.iter_mut().find(|e| e.0 == app && e.1 == res) {
                    Some(e) => {
                        for p in privs {
                            if !e.2.contains(&p) {
                                e.2.push(p);
                            }
                        }
                    }
                    None => apps.push((app, res, privs)),
                }
            }
            for r in str_list(&d["run_as"]) {
                if !run_as.contains(&r) {
                    run_as.push(r);
                }
            }
        }
        cluster.sort();
        run_as.sort();
        let indices: Vec<Value> = indices
            .into_iter()
            .map(|(names, restricted, fls, q, mut privs)| {
                privs.sort();
                let mut m = Map::new();
                m.insert("names".into(), json!(names));
                m.insert("privileges".into(), json!(privs));
                if !fls.is_empty() {
                    m.insert("field_security".into(), json!(fls));
                }
                if !q.is_empty() {
                    m.insert("query".into(), json!(q));
                }
                m.insert("allow_restricted_indices".into(), json!(restricted));
                Value::Object(m)
            })
            .collect();
        // ES merges twice: privileges of grants on the same resources, then
        // resources of grants with the same privileges.
        let mut merged: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
        for (app, res, mut privs) in apps {
            privs.sort();
            match merged.iter_mut().find(|e| e.0 == app && e.2 == privs) {
                Some(e) => {
                    for r in res {
                        if !e.1.contains(&r) {
                            e.1.push(r);
                        }
                    }
                    e.1.sort();
                }
                None => merged.push((app, res, privs)),
            }
        }
        let apps: Vec<Value> = merged.into_iter().map(|(app, res, privs)| json!({"application": app, "privileges": privs, "resources": res})).collect();
        let mut out = json!({"cluster": cluster, "global": global, "indices": indices, "applications": apps, "run_as": run_as});
        if !remote_indices.is_empty() {
            out["remote_indices"] = json!(remote_indices);
        }
        if !remote_cluster.is_empty() {
            out["remote_cluster"] = json!(remote_cluster);
        }
        Ok(Resp::ok(out))
    }

    /// Actions granted by application privilege `p` of `app`.
    fn app_actions(&self, app: &str, p: &str) -> Vec<String> {
        if p == "*" {
            return vec!["*".into()];
        }
        if p.contains('/') || p.contains(':') || p.contains('*') {
            return vec![p.to_string()];
        }
        let st = self.node.security.state.read();
        for (a, privs) in &st.app_privileges {
            if privilege::glob(a, app) || a == app {
                if let Some(d) = privs.get(p) {
                    return str_list(&d["actions"]);
                }
            }
        }
        vec![]
    }

    fn role_has_app(&self, role: &roles::Role, app: &str, resource: &str, wanted: &str) -> bool {
        let want_actions = if wanted.contains('/') || wanted.contains(':') || wanted.contains('*') { vec![wanted.to_string()] } else { self.app_actions(app, wanted) };
        if want_actions.is_empty() {
            // Unknown privilege name: only an explicit grant of that name (or `*`) satisfies it.
            return role.apps.iter().any(|g| privilege::glob(&g.app, app) && g.resources.iter().any(|r| cover(r, resource)) && g.privileges.iter().any(|p| p == "*" || p == wanted));
        }
        // Match resource groups once and resolve each grant once. A named grant
        // may expand to hundreds of actions; rebuilding that list for every
        // wanted action makes large Kibana privilege requests quadratic in
        // allocation work. Keep this request-local so role/privilege updates
        // require no additional cache invalidation.
        let mut seen = std::collections::HashSet::new();
        let granted: Vec<(&str, std::cell::OnceCell<Vec<String>>)> = role.apps.iter()
            .filter(|g| privilege::glob(&g.app, app) && g.resources.iter().any(|r| cover(r, resource)))
            .flat_map(|g| g.privileges.iter().map(String::as_str))
            .filter(|p| seen.insert(*p))
            .map(|p| (p, std::cell::OnceCell::new()))
            .collect();
        want_actions.iter().all(|w| granted.iter().any(|(p, actions)| {
            actions.get_or_init(|| self.app_actions(app, p)).iter().any(|ga| cover(ga, w))
        }))
    }

    fn has_privileges(&self, user: Option<&str>) -> R {
        if let Some(u) = user {
            if u != self.a.effective().username {
                return Err(EsError::new(403, "security_exception", format!("action [{P}/user/has_privileges] is unauthorized for {}, users may only check the privileges of their own account", super::subject_text(self.a))));
            }
        }
        let body = self.body()?;
        if body.is_null() || body.as_object().map(|o| o.is_empty()).unwrap_or(true) {
            return Err(validation("clusterPrivileges must not be null"));
        }
        let mut all = true;
        let mut cluster = Map::new();
        for c in str_list(&body["cluster"]) {
            let ok = self.perm.has_cluster_privilege(&c);
            all &= ok;
            cluster.insert(c, json!(ok));
        }
        let mut index = Map::new();
        for e in body["index"].as_array().into_iter().flatten() {
            for n in str_list(&e["names"]) {
                let entry = index.entry(n.clone()).or_insert_with(|| json!({}));
                for p in str_list(&e["privileges"]) {
                    let ok = self.perm.has_index_privilege(&n, &p);
                    all &= ok;
                    entry[p.as_str()] = json!(ok);
                }
            }
        }
        let mut application = Map::new();
        for e in body["application"].as_array().into_iter().flatten() {
            let app = e["application"].as_str().unwrap_or("").to_string();
            let app_entry = application.entry(app.clone()).or_insert_with(|| json!({}));
            for r in str_list(&e["resources"]) {
                if app_entry.get(&r).is_none() {
                    app_entry[r.as_str()] = json!({});
                }
                for p in str_list(&e["privileges"]) {
                    let ok = self.role_has_app(&self.perm.role, &app, &r, &p) && self.perm.limit.as_ref().map(|l| self.role_has_app(l, &app, &r, &p)).unwrap_or(true);
                    all &= ok;
                    app_entry[r.as_str()][p.as_str()] = json!(ok);
                }
            }
        }
        Ok(Resp::ok(json!({"username": self.a.effective().username, "has_all_requested": all, "cluster": cluster, "index": index, "application": application})))
    }

    fn builtin_privileges(&self) -> R {
        self.need(&format!("{P}/privilege/builtin/get"))?;
        let mut c: Vec<&str> = privilege::CLUSTER_PRIVS.iter().map(|d| d.name).collect();
        c.sort();
        let mut i: Vec<&str> = privilege::INDEX_PRIVS.iter().map(|(d, _)| d.name).collect();
        i.sort();
        Ok(Resp::ok(json!({"cluster": c, "index": i, "remote_cluster": ["monitor_enrich", "monitor_stats"]})))
    }

    fn put_privileges(&self) -> R {
        let body = self.body()?;
        let obj = body.as_object().ok_or_else(|| parse_error("expected an object".into()))?;
        let apps: Vec<String> = obj.keys().cloned().collect();
        self.need_apps(&format!("{P}/privilege/put"), &apps)?;
        let mut out = Map::new();
        let mut updates: Vec<(String, String, Value)> = Vec::new();
        for (app, privs) in obj {
            validate_app_name(app)?;
            let po = privs.as_object().ok_or_else(|| parse_error(format!("failed to parse application privileges for [{app}]")))?;
            for (name, def) in po {
                if let Some(a) = def["application"].as_str() {
                    if a != app {
                        return Err(EsError::illegal_argument(format!("application [{a}] in source does not match the application provided [{app}]")));
                    }
                }
                if let Some(nm) = def["name"].as_str() {
                    if nm != name {
                        return Err(EsError::illegal_argument(format!("privilege name [{nm}] in source does not match the privilege name provided [{name}]")));
                    }
                }
                let actions = str_list(&def["actions"]);
                for a in &actions {
                    if !(a.contains('/') || a.contains('*') || a.contains(':')) {
                        return Err(validation(&format!("action [{a}] must contain one of [/*:]")));
                    }
                }
                let md = if def["metadata"].is_object() { def["metadata"].clone() } else { json!({}) };
                check_metadata(&md)?;
                updates.push((app.clone(), name.clone(), json!({"application": app, "name": name, "actions": actions, "metadata": md})));
            }
        }
        let sec = &self.node.security;
        let mut st = sec.state.write();
        for (app, name, doc) in updates {
            let created = st.app_privileges.entry(app.clone()).or_default().insert(name.clone(), doc).is_none();
            out.entry(app).or_insert_with(|| json!({}))[name.as_str()] = json!({"created": created});
        }
        drop(st);
        sec.save()?;
        Ok(Resp::ok(Value::Object(out)))
    }

    fn get_privileges(&self, app: Option<&str>, names: Option<&str>) -> R {
        let apps: Vec<String> = app.map(names_of).unwrap_or_default();
        self.need_apps(&format!("{P}/privilege/get"), &apps)?;
        let st = self.node.security.state.read();
        let names: Vec<String> = names.map(names_of).unwrap_or_default();
        let mut out = Map::new();
        for (a, privs) in &st.app_privileges {
            if !apps.is_empty() && !apps.iter().any(|p| privilege::glob(p, a)) {
                continue;
            }
            for (n, d) in privs {
                if !names.is_empty() && !names.contains(n) {
                    continue;
                }
                out.entry(a.clone()).or_insert_with(|| json!({}))[n.as_str()] = d.clone();
            }
        }
        // ES `RestGetPrivilegesAction`: 404 only when specific privilege
        // names were asked for and none was found; an application without
        // privileges is `{}` with 200 (Kibana registers its privileges
        // after reading them so: a 404 there failed every registration).
        let missing = out.is_empty() && !names.is_empty();
        Ok(Resp::json(if missing { 404 } else { 200 }, Value::Object(out)))
    }

    fn delete_privileges(&self, app: &str, names: &str) -> R {
        self.need_apps(&format!("{P}/privilege/delete"), &[app.to_string()])?;
        let sec = &self.node.security;
        let mut st = sec.state.write();
        let mut res = Map::new();
        let mut any = false;
        for n in names_of(names) {
            let found = st.app_privileges.get_mut(app).map(|m| m.remove(&n).is_some()).unwrap_or(false);
            any |= found;
            res.insert(n, json!({"found": found}));
        }
        if st.app_privileges.get(app).map(|m| m.is_empty()).unwrap_or(false) {
            st.app_privileges.remove(app);
        }
        drop(st);
        if any {
            sec.save()?;
        }
        Ok(Resp::json(if any { 200 } else { 404 }, json!({ app: res })))
    }

    // ------------------------------------------------------------- roles

    fn get_roles(&self, names: Option<&str>) -> R {
        self.need(&format!("{P}/role/get"))?;
        let sec = &self.node.security;
        let mut out = Map::new();
        match names {
            None => {
                for (n, d) in roles::reserved_roles() {
                    out.insert(n.clone(), d.clone());
                }
                for (n, d) in sec.cfg.read().file_roles.iter() {
                    out.insert(n.clone(), d.clone());
                }
                for (n, d) in sec.state.read().roles.iter() {
                    out.insert(n.clone(), d.clone());
                }
            }
            Some(list) => {
                for n in names_of(list) {
                    if let Some(d) = sec.role_descriptor(&n) {
                        out.insert(n, d);
                    }
                }
                if out.is_empty() {
                    return Ok(Resp::json(404, json!({})));
                }
            }
        }
        Ok(Resp::ok(Value::Object(out)))
    }

    fn validate_role(&self, name: &str, body: &Value) -> Result<Value, EsError> {
        if roles::reserved_roles().contains_key(name) {
            return Err(validation(&format!("Role [{name}] is reserved and may not be used.")));
        }
        validate_name("Role", name, 507)?;
        let d = roles::parse_role(name, body, false)?;
        for (i, g) in d["indices"].as_array().into_iter().flatten().enumerate() {
            if let Some(q) = g["query"].as_str() {
                if let Ok(qv) = serde_json::from_str::<Value>(q) {
                    if has_terms_lookup(&qv) {
                        return Err(parse_error(format!("failed to parse field 'query' from indices[{i}]"))
                            .caused_by(EsError::illegal_argument("terms query with terms lookup isn't supported as part of a role query")));
                    }
                }
            }
        }
        check_metadata(&d["metadata"])?;
        Ok(d)
    }

    fn put_role(&self, name: &str) -> R {
        self.need(&format!("{P}/role/put"))?;
        let body = self.body()?;
        let d = self.validate_role(name, &body)?;
        let sec = &self.node.security;
        let created = sec.state.write().roles.insert(name.to_string(), d).is_none();
        sec.save()?;
        Ok(Resp::ok(json!({"role": {"created": created}})))
    }

    fn delete_role(&self, name: &str) -> R {
        self.need(&format!("{P}/role/delete"))?;
        if roles::reserved_roles().contains_key(name) {
            return Err(EsError::illegal_argument(format!("role [{name}] is reserved and cannot be deleted")));
        }
        let sec = &self.node.security;
        let found = sec.state.write().roles.remove(name).is_some();
        if found {
            sec.save()?;
        }
        Ok(Resp::json(if found { 200 } else { 404 }, json!({"found": found})))
    }

    fn bulk_put_roles(&self) -> R {
        self.need(&format!("{P}/role/bulk_put"))?;
        let body = self.body()?;
        let mut created = Vec::new();
        let mut updated = Vec::new();
        let mut noop = Vec::new();
        let mut errors = Map::new();
        let sec = &self.node.security;
        for (name, d) in body["roles"].as_object().into_iter().flatten() {
            match self.validate_role(name, d) {
                Ok(d) => {
                    let mut st = sec.state.write();
                    match st.roles.get(name) {
                        Some(old) if *old == d => noop.push(name.clone()),
                        Some(_) => {
                            st.roles.insert(name.clone(), d);
                            updated.push(name.clone());
                        }
                        None => {
                            st.roles.insert(name.clone(), d);
                            created.push(name.clone());
                        }
                    }
                }
                Err(e) => {
                    errors.insert(name.clone(), json!({"type": e.kind, "reason": e.reason}));
                }
            }
        }
        sec.save()?;
        let mut out = Map::new();
        if !created.is_empty() {
            out.insert("created".into(), json!(created));
        }
        if !updated.is_empty() {
            out.insert("updated".into(), json!(updated));
        }
        if !noop.is_empty() {
            out.insert("noop".into(), json!(noop));
        }
        if !errors.is_empty() {
            out.insert("errors".into(), json!({"count": errors.len(), "details": errors}));
        }
        Ok(Resp::ok(Value::Object(out)))
    }

    fn bulk_delete_roles(&self) -> R {
        self.need(&format!("{P}/role/bulk_delete"))?;
        let body = self.body()?;
        let sec = &self.node.security;
        let mut deleted = Vec::new();
        let mut not_found = Vec::new();
        let mut errors = Map::new();
        for n in str_list(&body["names"]) {
            if roles::reserved_roles().contains_key(&n) {
                errors.insert(n.clone(), json!({"type": "illegal_argument_exception", "reason": format!("role [{n}] is reserved and cannot be deleted")}));
            } else if sec.state.write().roles.remove(&n).is_some() {
                deleted.push(n);
            } else {
                not_found.push(n);
            }
        }
        sec.save()?;
        let mut out = Map::new();
        if !deleted.is_empty() {
            out.insert("deleted".into(), json!(deleted));
        }
        if !not_found.is_empty() {
            out.insert("not_found".into(), json!(not_found));
        }
        if !errors.is_empty() {
            out.insert("errors".into(), json!({"count": errors.len(), "details": errors}));
        }
        Ok(Resp::ok(Value::Object(out)))
    }

    fn query_roles(&self) -> R {
        self.need(&format!("{P}/role/query"))?;
        let body = self.body()?;
        let docs: Vec<Value> = self
            .node
            .security
            .state
            .read()
            .roles
            .iter()
            .map(|(n, d)| {
                let mut v = json!({"name": n});
                for (k, x) in d.as_object().into_iter().flatten() {
                    if k != "transient_metadata" {
                        v[k.as_str()] = x.clone();
                    }
                }
                v
            })
            .collect();
        let map = |f: &str| -> String { f.to_string() };
        let mut b = body.clone();
        if b["sort"].is_null() {
            b["sort"] = json!(["name"]);
        }
        let page = query::run(docs, &b, &map)?;
        let roles: Vec<Value> = page
            .hits
            .into_iter()
            .map(|(mut d, s)| {
                if !body["sort"].is_null() {
                    if let Some(s) = s {
                        d["_sort"] = json!(s);
                    }
                }
                d["transient_metadata"] = json!({"enabled": true});
                d
            })
            .collect();
        Ok(Resp::ok(json!({"total": page.total, "count": roles.len(), "roles": roles})))
    }

    // ----------------------------------------------------- role mappings

    fn put_role_mapping(&self, name: &str) -> R {
        self.need(&format!("{P}/role_mapping/put"))?;
        let body = self.body()?;
        let mut m = Map::new();
        for (k, v) in body.as_object().into_iter().flatten() {
            match k.as_str() {
                "enabled" | "roles" | "role_templates" | "rules" | "metadata" => {
                    m.insert(k.clone(), v.clone());
                }
                other => return Err(parse_error(format!("failed to parse role-mapping [{name}]. unexpected field [{other}]"))),
            }
        }
        if m.get("roles").is_none() && m.get("role_templates").is_none() {
            return Err(validation("must specify at least one role or role template"));
        }
        if m.get("rules").is_none() {
            return Err(validation("must specify rules"));
        }
        if m.get("enabled").is_none() {
            m.insert("enabled".into(), json!(true));
        }
        if m.get("metadata").is_none() {
            m.insert("metadata".into(), json!({}));
        }
        check_metadata(&m["metadata"])?;
        let mut out = Map::new();
        out.insert("enabled".into(), m["enabled"].clone());
        out.insert("roles".into(), m.get("roles").cloned().unwrap_or_else(|| json!([])));
        if let Some(t) = m.get("role_templates") {
            out.insert("role_templates".into(), t.clone());
        }
        out.insert("rules".into(), m["rules"].clone());
        out.insert("metadata".into(), m["metadata"].clone());
        let sec = &self.node.security;
        let created = sec.state.write().role_mappings.insert(name.to_string(), Value::Object(out)).is_none();
        sec.save()?;
        Ok(Resp::ok(json!({"role_mapping": {"created": created}})))
    }

    fn get_role_mappings(&self, names: Option<&str>) -> R {
        self.need(&format!("{P}/role_mapping/get"))?;
        let st = self.node.security.state.read();
        let mut out = Map::new();
        let wanted = names.map(names_of);
        for (n, d) in &st.role_mappings {
            if wanted.as_ref().map(|w| w.contains(n)).unwrap_or(true) {
                out.insert(n.clone(), d.clone());
            }
        }
        if out.is_empty() && names.is_some() {
            return Ok(Resp::json(404, json!({})));
        }
        Ok(Resp::ok(Value::Object(out)))
    }

    fn delete_role_mapping(&self, name: &str) -> R {
        self.need(&format!("{P}/role_mapping/delete"))?;
        let sec = &self.node.security;
        let found = sec.state.write().role_mappings.remove(name).is_some();
        if found {
            sec.save()?;
        }
        Ok(Resp::json(if found { 200 } else { 404 }, json!({"found": found})))
    }

    // ---------------------------------------------------------- api keys

    /// Without `manage_api_key` (only `manage_own_api_key`) callers see their own keys.
    fn api_key_admin(&self, op: &str) -> bool {
        let action = format!("{P}/api_key/{op}");
        let r = &self.perm.role;
        // Any grant of the action other than `manage_own_api_key` covers
        // every key: e.g. kibana_system's explicit
        // `cluster:admin/xpack/security/api_key/invalidate` (Kibana
        // invalidates keys it granted for other users).
        let has = |role: &roles::Role| {
            role.has_cluster_privilege("manage_api_key") || role.has_cluster_privilege("manage_security") || role.has_cluster_privilege("all")
                || (op == "get" && role.has_cluster_privilege("read_security")) || role.check_cluster_unconditional(&action)
        };
        has(r) && self.perm.limit.as_ref().map(|l| has(l)).unwrap_or(true)
    }

    fn owns(&self, k: &ApiKey) -> bool {
        let me = self.a.effective();
        if let Some(id) = self.a.api_key_id() {
            if k.id == id {
                return true;
            }
        }
        k.username == me.username && (k.realm == me.realm_name || self.a.is_api_key())
    }

    fn api_key_json(&self, k: &ApiKey, with_limited_by: bool) -> Value {
        let mut m = Map::new();
        m.insert("id".into(), json!(k.id));
        m.insert("name".into(), json!(k.name));
        m.insert("type".into(), json!(k.key_type));
        m.insert("creation".into(), json!(k.creation));
        if let Some(e) = k.expiration {
            m.insert("expiration".into(), json!(e));
        }
        m.insert("invalidated".into(), json!(k.invalidated));
        if let Some(i) = k.invalidation {
            m.insert("invalidation".into(), json!(i));
        }
        m.insert("username".into(), json!(k.username));
        m.insert("realm".into(), json!(k.realm));
        m.insert("realm_type".into(), json!(k.realm_type));
        m.insert("metadata".into(), if k.metadata.is_null() { json!({}) } else { k.metadata.clone() });
        m.insert("role_descriptors".into(), Value::Object(k.role_descriptors.clone()));
        if k.key_type == "cross_cluster" {
            m.insert("access".into(), k.access.clone());
        } else if with_limited_by {
            m.insert("limited_by".into(), json!([Value::Object(k.limited_by.clone())]));
        }
        Value::Object(m)
    }

    fn parse_key_request(&self, body: &Value, require_name: bool) -> Result<(Option<String>, Option<Option<i64>>, Option<Map<String, Value>>, Option<Value>), EsError> {
        let mut name = None;
        let mut expiration = None;
        let mut rds = None;
        let mut md = None;
        for (k, v) in body.as_object().into_iter().flatten() {
            match k.as_str() {
                "name" => name = v.as_str().map(str::to_string),
                "expiration" => {
                    expiration = Some(match v {
                        Value::Null => None,
                        Value::String(s) => Some(now_ms() + parse_time_ms(s, "expiration")?),
                        _ => return Err(parse_error("failed to parse expiration".into())),
                    })
                }
                "role_descriptors" => {
                    let mut out = Map::new();
                    for (rn, rd) in v.as_object().into_iter().flatten() {
                        let d = roles::parse_role(rn, rd, true)?;
                        out.insert(rn.clone(), d);
                    }
                    rds = Some(out);
                }
                "metadata" => {
                    check_metadata(v)?;
                    md = Some(v.clone());
                }
                "type" | "access" => {}
                other => return Err(parse_error(format!("[api_key_request] unknown field [{other}]"))),
            }
        }
        if require_name {
            match &name {
                None => return Err(validation("api key name is required")),
                Some(n) if n.is_empty() => return Err(validation("api key name is required")),
                Some(n) if n.starts_with('_') => return Err(validation("api key name may not begin with an underscore")),
                _ => {}
            }
        }
        Ok((name, expiration, rds, md))
    }

    fn owner_descriptors(&self, owner: &Authn) -> Map<String, Value> {
        if let Some(id) = owner.api_key_id() {
            let st = self.node.security.state.read();
            if let Some(k) = st.api_keys.get(id) {
                return if k.role_descriptors.is_empty() { k.limited_by.clone() } else { k.role_descriptors.clone() };
            }
        }
        if owner.is_service_account() {
            let mut m = Map::new();
            if let Some(d) = service_account(&owner.username) {
                m.insert(owner.username.clone(), d);
            }
            return m;
        }
        self.node.security.user_descriptors(owner)
    }

    fn do_create_key(&self, owner: &Authn, body: &Value) -> R {
        let (name, expiration, rds, md) = self.parse_key_request(body, true)?;
        if owner.is_api_key() {
            if let Some(r) = &rds {
                let any = r.values().any(|d| !d["cluster"].as_array().map(|a| a.is_empty()).unwrap_or(true) || !d["indices"].as_array().map(|a| a.is_empty()).unwrap_or(true) || !d["applications"].as_array().map(|a| a.is_empty()).unwrap_or(true));
                if r.is_empty() || any {
                    return Err(EsError::illegal_argument("creating derived api keys requires an explicit role descriptor that is empty (has no privileges)"));
                }
            } else {
                return Err(EsError::illegal_argument("creating derived api keys requires an explicit role descriptor that is empty (has no privileges)"));
            }
        }
        let limited = self.owner_descriptors(owner);
        let key = self.node.security.create_api_key(owner, limited, name.unwrap_or_default(), expiration.flatten(), rds.unwrap_or_default(), md.unwrap_or_else(|| json!({})))?;
        let secret = key.hash.clone();
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", key.id, secret));
        let mut m = Map::new();
        m.insert("id".into(), json!(key.id));
        m.insert("name".into(), json!(key.name));
        if let Some(e) = key.expiration {
            m.insert("expiration".into(), json!(e));
        }
        m.insert("api_key".into(), json!(secret));
        m.insert("encoded".into(), json!(encoded));
        Ok(Resp::ok(Value::Object(m)))
    }

    /// Build the synthetic `cross_cluster` role descriptor (and the normalized
    /// `access`) for a cross-cluster API key; mirrors ES's
    /// `CrossClusterApiKeyRoleDescriptorBuilder`.
    fn cross_cluster_descriptor(&self, access: &Value) -> Result<(Map<String, Value>, Value), EsError> {
        let section = |name: &str| -> Result<Vec<Value>, EsError> {
            let mut out = Vec::new();
            for e in access[name].as_array().into_iter().flatten() {
                let names = str_list(&e["names"]);
                if names.is_empty() {
                    return Err(parse_error(format!("failed to parse access [{name}]: Required [names]")));
                }
                if !e.is_object() {
                    return Err(parse_error(format!("failed to parse access [{name}]: expected an object")));
                }
                let mut item = json!({"names": names, "allow_restricted_indices": e["allow_restricted_indices"].as_bool().unwrap_or(false)});
                // Only `search` access supports document- and field-level security.
                if name == "search" {
                    if !e["field_security"].is_null() {
                        item["field_security"] = e["field_security"].clone();
                    }
                    if !e["query"].is_null() {
                        item["query"] = match &e["query"] {
                            Value::String(s) => Value::String(s.clone()),
                            other => Value::String(other.to_string()),
                        };
                    }
                }
                out.push(item);
            }
            Ok(out)
        };
        let (search, repl) = (section("search")?, section("replication")?);
        if search.is_empty() && repl.is_empty() {
            return Err(validation("must specify non-empty access for either [search] or [replication]"));
        }
        let mut cluster: Vec<&str> = vec![];
        let mut indices = vec![];
        if !search.is_empty() {
            cluster.extend(["cross_cluster_search", "monitor_enrich"]);
            for e in &search {
                let mut g = json!({"names": e["names"], "privileges": ["read", "read_cross_cluster", "view_index_metadata"], "allow_restricted_indices": e["allow_restricted_indices"]});
                for k in ["field_security", "query"] {
                    if !e[k].is_null() {
                        g[k] = e[k].clone();
                    }
                }
                indices.push(g);
            }
        }
        if !repl.is_empty() {
            cluster.push("cross_cluster_replication");
            for e in &repl {
                indices.push(json!({"names": e["names"], "privileges": ["cross_cluster_replication", "cross_cluster_replication_internal"], "allow_restricted_indices": e["allow_restricted_indices"]}));
            }
        }
        let mut rd = Map::new();
        rd.insert(
            "cross_cluster".into(),
            json!({"cluster": cluster, "indices": indices, "applications": [], "run_as": [], "metadata": {}, "transient_metadata": {"enabled": true}}),
        );
        let mut norm = Map::new();
        if !search.is_empty() {
            norm.insert("search".into(), Value::Array(search));
        }
        if !repl.is_empty() {
            norm.insert("replication".into(), Value::Array(repl));
        }
        Ok((rd, Value::Object(norm)))
    }

    fn create_cross_cluster_key(&self) -> R {
        self.need(&format!("{P}/cross_cluster/api_key/create"))?;
        if self.a.is_api_key() {
            return Err(EsError::illegal_argument("authentication via API key not supported: only the owner user of an API key can create a cross-cluster API key"));
        }
        let body = self.body()?;
        if !body["role_descriptors"].is_null() {
            return Err(parse_error("unknown field [role_descriptors]".into()));
        }
        let (name, expiration, _, md) = self.parse_key_request(&body, true)?;
        let (rds, access) = self.cross_cluster_descriptor(&body["access"])?;
        let key = self.node.security.create_api_key(self.a, Map::new(), name.unwrap_or_default(), expiration.flatten(), rds, md.unwrap_or_else(|| json!({})))?;
        {
            let mut st = self.node.security.state.write();
            if let Some(k) = st.api_keys.get_mut(&key.id) {
                k.key_type = "cross_cluster".into();
                k.access = access;
            }
        }
        self.node.security.save()?;
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", key.id, key.hash));
        let mut m = Map::new();
        m.insert("id".into(), json!(key.id));
        m.insert("name".into(), json!(key.name));
        if let Some(e) = key.expiration {
            m.insert("expiration".into(), json!(e));
        }
        m.insert("api_key".into(), json!(key.hash));
        m.insert("encoded".into(), json!(encoded));
        Ok(Resp::ok(Value::Object(m)))
    }

    fn update_cross_cluster_key(&self, id: &str) -> R {
        self.need(&format!("{P}/cross_cluster/api_key/update"))?;
        if self.a.is_api_key() {
            return Err(EsError::illegal_argument("authentication via API key not supported: An API key cannot be used to update API keys"));
        }
        let body = self.body()?;
        let (_, exp, _, md) = self.parse_key_request(&body, false)?;
        let new_access = if body["access"].is_null() { None } else { Some(self.cross_cluster_descriptor(&body["access"])?) };
        if new_access.is_none() && md.is_none() && exp.is_none() {
            return Err(validation("must update either [access] or [metadata] for cross-cluster API keys"));
        }
        let mut st = self.node.security.state.write();
        let k = match st.api_keys.get_mut(id).filter(|k| k.key_type == "cross_cluster") {
            Some(k) if self.owns(k) => k,
            _ => return Err(EsError::resource_not_found(format!("no API key owned by requesting user found for ID [{id}]"))),
        };
        if k.invalidated {
            return Err(validation(&format!("cannot update invalidated API key [{id}]")));
        }
        if k.expiration.map(|e| e <= now_ms()).unwrap_or(false) {
            return Err(validation(&format!("cannot update expired API key [{id}]")));
        }
        let before = (k.role_descriptors.clone(), k.access.clone(), k.metadata.clone(), k.expiration);
        if let Some((rds, access)) = new_access {
            k.role_descriptors = rds;
            k.access = access;
        }
        if let Some(m) = md {
            k.metadata = m;
        }
        if let Some(e) = exp {
            k.expiration = e;
        }
        let changed = before != (k.role_descriptors.clone(), k.access.clone(), k.metadata.clone(), k.expiration);
        drop(st);
        if changed {
            self.node.security.save()?;
        }
        Ok(Resp::ok(json!({"updated": changed})))
    }

    fn create_api_key(&self) -> R {
        self.need(&format!("{P}/api_key/create"))?;
        let body = self.body()?;
        self.do_create_key(self.a, &body)
    }

    /// `POST|PUT /_security/api_key/clone` (`RestCloneApiKeyAction`).
    fn clone_api_key(&self) -> R {
        self.need(&format!("{P}/api_key/clone"))?;
        let body = self.body()?;
        if let Some(o) = body.as_object() {
            for k in o.keys() {
                if !matches!(k.as_str(), "api_key" | "name" | "expiration" | "metadata") {
                    return Err(EsError::x_content_parse(format!("[1:2] [clone_api_key_request] unknown field [{k}]")));
                }
            }
        }
        let cred = body["api_key"].as_str().unwrap_or("");
        let name = body["name"].as_str().unwrap_or("").to_string();
        let mut errs: Vec<String> = Vec::new();
        if cred.is_empty() {
            errs.push("api key credential is required".into());
        }
        if name.trim().is_empty() {
            errs.push("api key name is required".into());
        }
        if !errs.is_empty() {
            let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
            return Err(EsError::action_validation(&refs));
        }
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD.decode(cred.trim()).ok().and_then(|b| String::from_utf8(b).ok());
        let Some((id, secret)) = decoded.as_deref().and_then(|d| d.split_once(':')) else {
            return Err(EsError::new(400, "status_exception", "invalid API key credential"));
        };
        let expiration = match body.get("expiration") {
            None => None,
            Some(Value::Null) => Some(None),
            Some(v) => {
                let t = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
                Some(Some(now_ms() + parse_time_ms(&t, "expiration")?))
            }
        };
        let key = match self.node.security.clone_api_key(id, secret, name, expiration, body.get("metadata").cloned()) {
            Ok(k) => k,
            Err(Ok(msg)) => {
                return Err(EsError::new(403, "security_exception", "Failed to clone API key").caused_by(EsError::new(401, "security_exception", msg)));
            }
            Err(Err(e)) => return Err(e),
        };
        let encoded = base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", key.id, key.hash));
        let mut m = Map::new();
        m.insert("id".into(), json!(key.id));
        m.insert("name".into(), json!(key.name));
        if let Some(e) = key.expiration {
            m.insert("expiration".into(), json!(e));
        }
        m.insert("api_key".into(), json!(key.hash));
        m.insert("encoded".into(), json!(encoded));
        Ok(Resp::ok(Value::Object(m)))
    }

    fn grant_api_key(&self) -> R {
        self.need(&format!("{P}/api_key/grant"))?;
        let body = self.body()?;
        if !body["client_authentication"].is_null() {
            return Err(EsError::new(400, "security_exception", "[client_authentication] not supported with the supplied access_token type"));
        }
        let grantee = match body["grant_type"].as_str() {
            Some("password") => {
                let u = body["username"].as_str().ok_or_else(|| validation("[username] is required for grant_type [password]"))?;
                let p = body["password"].as_str().ok_or_else(|| validation("[password] is required for grant_type [password]"))?;
                self.node.security.authenticate_password(u, p).map_err(|_| EsError::new(401, "security_exception", format!("unable to authenticate user [{u}] for action [{P}/api_key/grant]")))?
            }
            Some("access_token") => {
                let t = body["access_token"].as_str().ok_or_else(|| validation("[access_token] is required for grant_type [access_token]"))?;
                self.node.security.authenticate(sec::Credentials::Bearer(t.to_string())).map_err(|_| EsError::new(401, "security_exception", format!("unable to authenticate user for action [{P}/api_key/grant]")))?
            }
            Some(other) => return Err(validation(&format!("grant_type [{other}] is not supported"))),
            None => return Err(validation("[grant_type] is required")),
        };
        let grantee = match body["run_as"].as_str() {
            Some(t) => self.node.security.run_as(grantee.clone(), t).map_err(|_| EsError::new(403, "security_exception", format!("action [{P}/api_key/grant] is unauthorized for user [{}], because user [{}] is unauthorized to run as [{t}]", grantee.username, grantee.username)))?,
            None => grantee,
        };
        self.do_create_key(&grantee, &body["api_key"])
    }

    fn get_api_keys(&self) -> R {
        self.need(&format!("{P}/api_key/get"))?;
        let p = |k: &str| self.req.param(k).filter(|v| !v.is_empty()).map(str::to_string);
        let owner = self.req.flag("owner")?;
        let active_only = self.req.flag("active_only")?;
        let with_limited_by = self.req.flag("with_limited_by")?;
        let (id, name, username, realm) = (p("id"), p("name"), p("username"), p("realm_name"));
        if owner && (username.is_some() || realm.is_some()) {
            return Err(validation("username or realm name must not be specified when the current user should be used"));
        }
        if id.is_some() && name.is_some() {
            return Err(validation("only one of [api key id, api key name] can be specified"));
        }
        let admin = self.api_key_admin("get");
        if with_limited_by && !admin && self.a.is_api_key() {
            return Err(denied(self.node, self.a, self.perm, &format!("{P}/api_key/get"), None));
        }
        let own_only = owner || !admin;
        let st = self.node.security.state.read();
        let now = now_ms();
        let mut keys: Vec<&ApiKey> = st
            .api_keys
            .values()
            .filter(|k| id.as_ref().map(|i| &k.id == i).unwrap_or(true))
            .filter(|k| name.as_ref().map(|n| privilege::glob(n, &k.name)).unwrap_or(true))
            .filter(|k| username.as_ref().map(|u| &k.username == u).unwrap_or(true))
            .filter(|k| realm.as_ref().map(|r| &k.realm == r).unwrap_or(true))
            .filter(|k| !own_only || self.owns(k))
            .filter(|k| !active_only || (!k.invalidated && k.expiration.map(|e| e > now).unwrap_or(true)))
            .collect();
        if !admin && !owner && keys.is_empty() && (username.is_some() || realm.is_some()) {
            return Err(denied(self.node, self.a, self.perm, &format!("{P}/api_key/get"), None));
        }
        keys.sort_by_key(|k| (k.creation, k.id.clone()));
        let out: Vec<Value> = keys.iter().map(|k| self.api_key_json(k, with_limited_by)).collect();
        Ok(Resp::ok(json!({"api_keys": out})))
    }

    fn invalidate_api_keys(&self) -> R {
        let action = format!("{P}/api_key/invalidate");
        self.need(&action)?;
        let body = self.body()?;
        let mut ids: Vec<String> = str_list(&body["ids"]);
        if let Some(i) = body["id"].as_str() {
            ids.push(i.to_string());
        }
        let name = body["name"].as_str().map(str::to_string);
        let username = body["username"].as_str().map(str::to_string);
        let realm = body["realm_name"].as_str().map(str::to_string);
        let owner = body["owner"].as_bool().unwrap_or(false);
        if !body["id"].is_null() && !body["ids"].is_null() {
            return Err(validation("Must use either [id] or [ids], not both at the same time"));
        }
        if ids.is_empty() && name.is_none() && username.is_none() && realm.is_none() && !owner {
            return Err(validation("One of [api key id(s), api key name, username, realm name] must be specified if [owner] flag is false"));
        }
        let admin = self.api_key_admin("invalidate");
        if !admin && !owner && !(ids.len() == 1 && self.a.api_key_id() == Some(ids[0].as_str())) {
            // manage_own_api_key may only target its own keys
            let st = self.node.security.state.read();
            let all_own = !ids.is_empty() && ids.iter().all(|i| st.api_keys.get(i).map(|k| self.owns(k)).unwrap_or(false));
            if !all_own || username.is_some() || realm.is_some() {
                drop(st);
                return Err(denied(self.node, self.a, self.perm, &action, None));
            }
        }
        let sec = &self.node.security;
        let mut st = sec.state.write();
        let now = now_ms();
        let mut inv = Vec::new();
        let mut prev = Vec::new();
        let matched: Vec<String> = st
            .api_keys
            .values()
            .filter(|k| ids.is_empty() || ids.contains(&k.id))
            .filter(|k| name.as_ref().map(|n| privilege::glob(n, &k.name)).unwrap_or(true))
            .filter(|k| username.as_ref().map(|u| &k.username == u).unwrap_or(true))
            .filter(|k| realm.as_ref().map(|r| &k.realm == r).unwrap_or(true))
            .filter(|k| !owner || self.owns(k))
            .filter(|k| admin || owner || self.owns(k))
            .map(|k| k.id.clone())
            .collect();
        for id in matched {
            if let Some(k) = st.api_keys.get_mut(&id) {
                if k.invalidated {
                    prev.push(id);
                } else {
                    k.invalidated = true;
                    k.invalidation = Some(now);
                    inv.push(id);
                }
            }
        }
        drop(st);
        if !inv.is_empty() {
            sec.save()?;
        }
        Ok(Resp::ok(json!({"invalidated_api_keys": inv, "previously_invalidated_api_keys": prev, "error_count": 0})))
    }

    fn apply_update(&self, id: &str, rds: &Option<Map<String, Value>>, md: &Option<Value>, exp: &Option<Option<i64>>) -> Result<bool, EsError> {
        let limited = self.owner_descriptors(self.a);
        let sec = &self.node.security;
        let mut st = sec.state.write();
        let k = st.api_keys.get_mut(id).filter(|k| k.key_type == "rest");
        let k = match k {
            Some(k) if self.owns(k) && k.realm == self.a.effective().realm_name => k,
            _ => return Err(EsError::resource_not_found(format!("no API key owned by requesting user found for ID [{id}]"))),
        };
        if k.invalidated {
            return Err(validation(&format!("cannot update invalidated API key [{id}]")));
        }
        if k.expiration.map(|e| e <= now_ms()).unwrap_or(false) {
            return Err(validation(&format!("cannot update expired API key [{id}]")));
        }
        let before = (k.role_descriptors.clone(), k.metadata.clone(), k.limited_by.clone(), k.expiration);
        if let Some(r) = rds {
            k.role_descriptors = r.clone();
        }
        if let Some(m) = md {
            k.metadata = m.clone();
        }
        if let Some(e) = exp {
            k.expiration = *e;
        }
        k.limited_by = limited;
        k.username = self.a.effective().username.clone();
        let changed = before != (k.role_descriptors.clone(), k.metadata.clone(), k.limited_by.clone(), k.expiration);
        Ok(changed)
    }

    fn update_api_key(&self, id: &str) -> R {
        self.need(&format!("{P}/api_key/update"))?;
        if self.a.is_api_key() {
            return Err(EsError::illegal_argument("authentication via API key not supported: An API key cannot be used to update API keys"));
        }
        let body = self.body()?;
        let (_, exp, rds, md) = self.parse_key_request(&body, false)?;
        if !body["name"].is_null() {
            return Err(parse_error("[update_api_key_request] unknown field [name]".into()));
        }
        let changed = self.apply_update(id, &rds, &md, &exp)?;
        if changed {
            self.node.security.save()?;
        }
        Ok(Resp::ok(json!({"updated": changed})))
    }

    fn bulk_update_api_keys(&self) -> R {
        self.need(&format!("{P}/api_key/update"))?;
        if self.a.is_api_key() {
            return Err(EsError::illegal_argument("authentication via API key not supported: An API key cannot be used to update API keys"));
        }
        let body = self.body()?;
        let ids = str_list(&body["ids"]);
        let mut b = body.clone();
        if let Some(o) = b.as_object_mut() {
            o.remove("ids");
        }
        let (_, exp, rds, md) = self.parse_key_request(&b, false)?;
        let mut updated = Vec::new();
        let mut noops = Vec::new();
        let mut errors = Map::new();
        for id in ids {
            match self.apply_update(&id, &rds, &md, &exp) {
                Ok(true) => updated.push(id),
                Ok(false) => noops.push(id),
                Err(e) => {
                    errors.insert(id, json!({"type": e.kind, "reason": e.reason}));
                }
            }
        }
        self.node.security.save()?;
        let mut out = json!({"updated": updated, "noops": noops});
        if !errors.is_empty() {
            out["errors"] = json!({"count": errors.len(), "details": errors});
        }
        Ok(Resp::ok(out))
    }

    fn query_api_keys(&self) -> R {
        self.need(&format!("{P}/api_key/query"))?;
        let body = self.body()?;
        let with_limited_by = self.req.flag("with_limited_by")?;
        let admin = self.api_key_admin("get");
        if with_limited_by && !admin && self.a.is_api_key() {
            return Err(denied(self.node, self.a, self.perm, &format!("{P}/api_key/query"), None));
        }
        let docs: Vec<Value> = {
            let st = self.node.security.state.read();
            let mut ks: Vec<&ApiKey> = st.api_keys.values().filter(|k| admin || self.owns(k)).collect();
            ks.sort_by_key(|k| (k.creation, k.id.clone()));
            ks.iter().map(|k| self.api_key_json(k, with_limited_by)).collect()
        };
        let map = |f: &str| -> String {
            match f {
                "_id" => "id".into(),
                "realm_name" => "realm".into(),
                other => other.to_string(),
            }
        };
        let page = query::run(docs, &body, &map)?;
        let keys: Vec<Value> = page
            .hits
            .into_iter()
            .map(|(mut d, s)| {
                if let Some(s) = s {
                    d["_sort"] = json!(s);
                }
                d
            })
            .collect();
        let mut out = json!({"total": page.total, "count": keys.len(), "api_keys": keys});
        if let Some(a) = page.aggregations {
            out["aggregations"] = a;
        }
        Ok(Resp::ok(out))
    }

    // ------------------------------------------------------------ tokens

    fn tokens_available(&self) -> Result<(), EsError> {
        let c = self.node.security.cfg.read();
        if !c.trial {
            return Err(license_error("security tokens"));
        }
        if !c.token_enabled {
            return Err(EsError::new(400, "security_exception", "security tokens are not enabled").with("disabled.feature", "security_tokens"));
        }
        Ok(())
    }

    fn token_response(&self, doc: &sec::TokenDoc, authn: &Authn) -> Value {
        let mut m = Map::new();
        m.insert("access_token".into(), json!(doc.access_token));
        m.insert("type".into(), json!("Bearer"));
        m.insert("expires_in".into(), json!((doc.expires - doc.creation) / 1000));
        if let Some(r) = &doc.refresh_token {
            m.insert("refresh_token".into(), json!(r));
        }
        let mut auth = authn.to_json();
        if let Some(o) = auth.as_object_mut() {
            o.insert("authentication_type".into(), json!("realm"));
        }
        m.insert("authentication".into(), auth);
        Value::Object(m)
    }

    fn create_token(&self) -> R {
        self.tokens_available()?;
        let body = self.body()?;
        let gt = body["grant_type"].as_str().ok_or_else(|| validation("grant_type is missing"))?.to_string();
        let sec = &self.node.security;
        match gt.as_str() {
            "password" => {
                self.need(&format!("{P}/token/create"))?;
                let u = body["username"].as_str();
                let p = body["password"].as_str();
                let mut errs = Vec::new();
                if u.is_none() {
                    errs.push("username is missing");
                }
                if p.is_none() {
                    errs.push("password is missing");
                }
                if !errs.is_empty() {
                    return Err(EsError::action_validation(&errs));
                }
                let (u, p) = (u.unwrap_or(""), p.unwrap_or(""));
                let user = sec.authenticate_password(u, p).map_err(|_| EsError::new(401, "security_exception", format!("unable to authenticate user [{u}] for action [{P}/token/create]")))?;
                let doc = sec.create_token(&user, true)?;
                Ok(Resp::ok(self.token_response(&doc, &user)))
            }
            "client_credentials" => {
                self.need(&format!("{P}/token/create"))?;
                let doc = sec.create_token(self.a, false)?;
                Ok(Resp::ok(self.token_response(&doc, self.a)))
            }
            "refresh_token" => {
                self.need(&format!("{P}/token/refresh"))?;
                let rt = body["refresh_token"].as_str().ok_or_else(|| validation("refresh_token is missing"))?.to_string();
                let found = {
                    let st = sec.state.read();
                    st.tokens.values().find(|t| t.refresh_token.as_deref() == Some(rt.as_str())).cloned()
                };
                let invalid_grant = |why: &str| EsError::new(400, "security_exception", "invalid_grant").with("error_description", why);
                let Some(old) = found else { return Err(invalid_grant("could not refresh the requested token")) };
                if old.refresh_invalidated {
                    return Err(invalid_grant("token has been invalidated"));
                }
                if old.refreshed {
                    return Err(invalid_grant("token has already been refreshed"));
                }
                if old.creation + 24 * 3600 * 1000 < now_ms() {
                    return Err(invalid_grant("refresh token is expired"));
                }
                {
                    let mut st = sec.state.write();
                    if let Some(t) = st.tokens.get_mut(&old.access_token) {
                        t.refreshed = true;
                    }
                }
                let s = &old.authn;
                let user = sec.lookup_user(&s.username).unwrap_or_else(|| {
                    let mut a = self.a.clone();
                    a.username = s.username.clone();
                    a
                });
                let doc = sec.create_token(&user, true)?;
                Ok(Resp::ok(self.token_response(&doc, &user)))
            }
            "_kerberos" => Err(license_error("kerberos")),
            other => Err(validation(&format!("grant_type [{other}] is not supported"))),
        }
    }

    fn invalidate_token(&self) -> R {
        self.tokens_available()?;
        self.need(&format!("{P}/token/invalidate"))?;
        let body = self.body()?;
        let token = body["token"].as_str().map(str::to_string);
        let refresh = body["refresh_token"].as_str().map(str::to_string);
        let realm = body["realm_name"].as_str().map(str::to_string);
        let username = body["username"].as_str().map(str::to_string);
        if token.is_none() && refresh.is_none() && realm.is_none() && username.is_none() {
            return Err(validation("one of [token, refresh_token, realm_name, username] must be specified"));
        }
        let sec = &self.node.security;
        let mut st = sec.state.write();
        if st.tokens.is_empty() {
            return Err(EsError::new(400, "security_exception", "token malformed"));
        }
        let by_owner = token.is_none() && refresh.is_none();
        let (mut inv, mut prev) = (0, 0);
        let mut any = false;
        for t in st.tokens.values_mut() {
            let hit = token.as_deref().map(|x| x == t.access_token).unwrap_or(false)
                || refresh.as_deref().map(|x| Some(x) == t.refresh_token.as_deref()).unwrap_or(false)
                || (token.is_none() && refresh.is_none() && realm.as_deref().map(|r| r == t.authn.realm_name).unwrap_or(true) && username.as_deref().map(|u| u == t.authn.username).unwrap_or(true));
            if !hit {
                continue;
            }
            any = true;
            if by_owner {
                // access and refresh tokens count separately
                for (flag, present) in [(&mut t.invalidated, true), (&mut t.refresh_invalidated, t.refresh_token.is_some())] {
                    if !present {
                        continue;
                    }
                    if *flag {
                        prev += 1;
                    } else {
                        *flag = true;
                        inv += 1;
                    }
                }
                continue;
            }
            let already = if refresh.is_some() { t.refresh_invalidated } else { t.invalidated };
            if already {
                prev += 1;
            } else {
                inv += 1;
                if refresh.is_some() {
                    t.refresh_invalidated = true;
                } else {
                    t.invalidated = true;
                    t.refresh_invalidated = true;
                }
            }
        }
        drop(st);
        if (token.is_some() || refresh.is_some()) && !any {
            return Ok(Resp::json(404, json!({"invalidated_tokens": 0, "previously_invalidated_tokens": 0, "error_count": 0})));
        }
        sec.save()?;
        Ok(Resp::ok(json!({"invalidated_tokens": inv, "previously_invalidated_tokens": prev, "error_count": 0})))
    }

    // -------------------------------------------------- service accounts

    fn service(&self) -> R {
        let m = &self.req.method;
        let segs: Vec<&str> = self.req.segs.iter().map(String::as_str).collect();
        let n = segs.len();
        let get = *m == Method::GET || *m == Method::HEAD;
        if get && n <= 4 {
            self.need(&format!("{P}/service_account/get"))?;
            let ns = segs.get(2).copied();
            let svc = segs.get(3).copied();
            let mut out = Map::new();
            for a in SERVICE_ACCOUNTS {
                let (ans, asvc) = a.split_once('/').unwrap_or(("", ""));
                if ns.map(|x| x == ans).unwrap_or(true) && svc.map(|x| x == asvc).unwrap_or(true) {
                    out.insert(a.to_string(), json!({"role_descriptor": service_account(a)}));
                }
            }
            return Ok(Resp::ok(Value::Object(out)));
        }
        if n < 5 || segs[4] != "credential" {
            return crate::not_found_handler(self.req);
        }
        let account = format!("{}/{}", segs[2], segs[3]);
        if n == 5 && get {
            self.need(&format!("{P}/service_account/credential/get"))?;
            if service_account(&account).is_none() {
                return Err(EsError::illegal_argument(format!("service account [{account}] does not exist")));
            }
            let st = self.node.security.state.read();
            let idx: Map<String, Value> = st.service_tokens.get(&account).map(|m| m.keys().map(|k| (k.clone(), json!({}))).collect()).unwrap_or_default();
            drop(st);
            let prefix = format!("{account}/");
            let files: Map<String, Value> = self
                .node
                .security
                .cfg
                .read()
                .file_service_tokens
                .keys()
                .filter_map(|k| k.strip_prefix(&prefix))
                .map(|k| (k.to_string(), json!({"nodes": [self.node.name]})))
                .collect();
            return Ok(Resp::ok(json!({
                "service_account": account,
                "count": idx.len() + files.len(),
                "tokens": idx,
                "nodes_credentials": {"_nodes": {"total": 1, "successful": 1, "failed": 0}, "file_tokens": files}
            })));
        }
        if n < 6 || segs[5] != "token" {
            return crate::not_found_handler(self.req);
        }
        let name = segs.get(6).map(|s| s.to_string());
        if n == 8 && segs[7] == "_clear_cache" && *m == Method::POST {
            self.need(&format!("{P}/service_account/token/clear_cache"))?;
            return Ok(Resp::ok(self.nodes_response()));
        }
        let sec = &self.node.security;
        match *m {
            Method::POST | Method::PUT => {
                self.need(&format!("{P}/service_account/token/create"))?;
                if service_account(&account).is_none() {
                    return Err(EsError::illegal_argument(format!("service account [{account}] does not exist")));
                }
                let name = name.unwrap_or_else(|| format!("token_{}", hash::random_b64url(15)));
                if !valid_token_name(&name) {
                    return Err(validation("service account token name must have at least 1 character and at most 256 characters that are alphanumeric (A-Z, a-z, 0-9) or hyphen (-) or underscore (_). It must not begin with an underscore (_)."));
                }
                let secret = hash::random_b64url(16);
                let mut st = sec.state.write();
                let m = st.service_tokens.entry(account.clone()).or_default();
                if m.contains_key(&name) {
                    return Err(EsError::new(409, "version_conflict_engine_exception", format!("[service_account_token-{account}/{name}]: version conflict, document already exists (current version [1])")));
                }
                m.insert(name.clone(), ServiceToken { hash: Hasher::Ssha256.hash(&secret), creation: now_ms() });
                drop(st);
                sec.save()?;
                Ok(Resp::ok(json!({"created": true, "token": {"name": name, "value": sec::service_token_value(&account, &name, &secret)}})))
            }
            Method::DELETE => {
                self.need(&format!("{P}/service_account/token/delete"))?;
                let name = name.unwrap_or_default();
                let found = sec.state.write().service_tokens.get_mut(&account).map(|m| m.remove(&name).is_some()).unwrap_or(false);
                if found {
                    sec.save()?;
                }
                Ok(Resp::json(if found { 200 } else { 404 }, json!({"found": found})))
            }
            _ => crate::not_found_handler(self.req),
        }
    }

    // ---------------------------------------------------------- profiles

    fn profile(&self) -> R {
        let m = &self.req.method;
        let s2 = self.seg(2).to_string();
        let s3 = self.seg(3).to_string();
        let sec = &self.node.security;
        match (s2.as_str(), s3.as_str()) {
            ("_activate", _) if *m == Method::POST => {
                self.need(&format!("{P}/profile/activate"))?;
                let body = self.body()?;
                let user = match body["grant_type"].as_str() {
                    Some("password") => {
                        let u = body["username"].as_str().unwrap_or("");
                        sec.authenticate_password(u, body["password"].as_str().unwrap_or("")).map_err(|_| EsError::new(401, "security_exception", format!("unable to authenticate user [{u}] for action [{P}/profile/activate]")))?
                    }
                    Some("access_token") => sec.authenticate(sec::Credentials::Bearer(body["access_token"].as_str().unwrap_or("").to_string())).map_err(|_| EsError::new(401, "security_exception", "unable to authenticate with the provided access token"))?,
                    _ => return Err(validation("grant_type is missing")),
                };
                if user.is_api_key() || user.is_service_account() {
                    return Err(EsError::illegal_argument("profile is supported for user only, but subject is a [API_KEY]"));
                }
                let mut st = sec.state.write();
                let existing = st.profiles.iter().find(|(_, p)| p["user"]["username"] == json!(user.username) && p["user"]["realm_name"] == json!(user.realm_name)).map(|(k, _)| k.clone());
                // ES `ProfileService`: `u_` + base64url(sha256(username)) + `_<differentiator>`
                // (the differentiator skips uids already taken by another realm's user).
                let uid = existing.unwrap_or_else(|| {
                    use base64::Engine;
                    let base = format!("u_{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash::sha256(user.username.as_bytes())));
                    (0..).map(|i| format!("{base}_{i}")).find(|u| !st.profiles.contains_key(u)).unwrap_or_default()
                });
                let prev = st.profiles.get(&uid).cloned();
                let seq = prev.as_ref().and_then(|p| p["_doc"]["_seq_no"].as_i64()).map(|n| n + 1).unwrap_or(0);
                let mut u = Map::new();
                u.insert("username".into(), json!(user.username));
                u.insert("roles".into(), json!(user.roles));
                u.insert("realm_name".into(), json!(user.realm_name));
                if let Some(e) = &user.email {
                    u.insert("email".into(), json!(e));
                }
                if let Some(f) = &user.full_name {
                    u.insert("full_name".into(), json!(f));
                }
                let doc = json!({
                    "uid": uid, "enabled": prev.as_ref().map(|p| p["enabled"].clone()).unwrap_or(json!(true)), "last_synchronized": now_ms(),
                    "user": u,
                    "labels": prev.as_ref().map(|p| p["labels"].clone()).unwrap_or(json!({})),
                    "data": prev.as_ref().map(|p| p["data"].clone()).unwrap_or(json!({})),
                    "_doc": {"_primary_term": 1, "_seq_no": seq}
                });
                st.profiles.insert(uid, doc.clone());
                drop(st);
                sec.save()?;
                Ok(Resp::ok(doc))
            }
            ("_suggest", _) => {
                self.need(&format!("{P}/profile/suggest"))?;
                if !sec.cfg.read().trial {
                    return Err(license_error("user-profile-collaboration"));
                }
                self.suggest_profiles()
            }
            ("_has_privileges", _) => {
                self.need(&format!("{P}/profile/has_privileges"))?;
                if !sec.cfg.read().trial {
                    return Err(license_error("user-profile-collaboration"));
                }
                self.profile_has_privileges()
            }
            (uid, "_data") => {
                self.need(&format!("{P}/profile/put/data"))?;
                let body = self.body()?;
                let mut st = sec.state.write();
                let p = st.profiles.get_mut(uid).ok_or_else(|| EsError::new(404, "document_missing_exception", format!("[{uid}]: document missing")))?;
                for k in ["labels", "data"] {
                    if let Some(o) = body[k].as_object() {
                        for (kk, vv) in o {
                            p[k][kk.as_str()] = vv.clone();
                        }
                    }
                }
                bump_seq(p);
                drop(st);
                sec.save()?;
                Ok(Resp::ack())
            }
            (uid, "_enable") | (uid, "_disable") => {
                self.need(&format!("{P}/profile/put/enabled"))?;
                let mut st = sec.state.write();
                let p = st.profiles.get_mut(uid).ok_or_else(|| EsError::new(404, "document_missing_exception", format!("[{uid}]: document missing")))?;
                p["enabled"] = json!(s3 == "_enable");
                bump_seq(p);
                drop(st);
                sec.save()?;
                Ok(Resp::ack())
            }
            (uids, "") if *m == Method::GET => {
                self.need(&format!("{P}/profile/get"))?;
                let st = sec.state.read();
                let mut profiles = Vec::new();
                let mut missing = Vec::new();
                let data_keys: Vec<String> = self.req.param("data").map(names_of).unwrap_or_default();
                for u in names_of(uids) {
                    match st.profiles.get(&u) {
                        Some(p) => {
                            let mut p = p.clone();
                            p["data"] = filter_profile_data(&p["data"], &data_keys);
                            profiles.push(p);
                        }
                        None => missing.push(u),
                    }
                }
                let mut out = json!({"profiles": profiles});
                if !missing.is_empty() {
                    let details: Map<String, Value> = missing.iter().map(|u| (u.clone(), json!({"type": "resource_not_found_exception", "reason": "profile document not found"}))).collect();
                    out["errors"] = json!({"count": missing.len(), "details": details});
                }
                Ok(Resp::ok(out))
            }
            _ => crate::not_found_handler(self.req),
        }
    }

    /// `_security/profile/_suggest`: ES matches `name` (bool-prefix
    /// multi_match over username / full_name / email) on enabled profiles,
    /// `hint.uids` / `hint.labels` only boost.
    fn suggest_profiles(&self) -> R {
        let started = std::time::Instant::now();
        let body = self.body()?;
        let empty = Map::new();
        let o = match &body {
            Value::Object(o) => o,
            Value::Null => &empty,
            _ => return Err(parse_error("[suggest_profile_request_payload] expected an object".into())),
        };
        for k in o.keys() {
            if !matches!(k.as_str(), "name" | "size" | "data" | "hint") {
                return Err(EsError::x_content_parse(format!("[1:2] [suggest_profile_request_payload] unknown field [{k}]")));
            }
        }
        let name = o.get("name").and_then(Value::as_str).or_else(|| self.req.param("name")).unwrap_or("").to_string();
        let size = match o.get("size").and_then(Value::as_i64).or_else(|| self.req.param("size").and_then(|s| s.parse().ok())) {
            Some(n) if n < 0 => return Err(validation("[size] parameter cannot be negative but was [{n}]".replace("{n}", &n.to_string()).as_str())),
            Some(n) => n as usize,
            None => 10,
        };
        let mut data_keys: Vec<String> = self.req.param("data").map(names_of).unwrap_or_default();
        data_keys.extend(str_list(o.get("data").unwrap_or(&Value::Null)));
        let hint_uids = str_list(&o.get("hint").map(|h| h["uids"].clone()).unwrap_or(Value::Null));
        let hint_labels: Vec<(String, Vec<String>)> = o
            .get("hint")
            .and_then(|h| h["labels"].as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), str_list(v))).collect())
            .unwrap_or_default();
        let toks = |s: &str| -> Vec<String> { s.split(|c: char| !c.is_alphanumeric() && c != '.' && c != '_').filter(|t| !t.is_empty()).map(|t| t.to_lowercase()).collect() };
        let q = toks(&name);
        let st = self.node.security.state.read();
        let mut hits: Vec<(f64, Value)> = Vec::new();
        for (uid, p) in &st.profiles {
            if p["enabled"] != json!(true) {
                continue;
            }
            let u = &p["user"];
            let mut fields: Vec<String> = Vec::new();
            for f in ["username", "full_name", "email"] {
                if let Some(s) = u[f].as_str() {
                    fields.extend(toks(s));
                    fields.push(s.to_lowercase());
                }
            }
            let mut score = 0.0;
            if !q.is_empty() {
                for (i, t) in q.iter().enumerate() {
                    let last = i + 1 == q.len();
                    if fields.iter().any(|f| f == t || (last && f.starts_with(t.as_str()))) {
                        score += 1.0;
                    }
                }
                if score == 0.0 {
                    continue;
                }
            }
            if hint_uids.contains(uid) {
                score += 0.5;
            }
            for (k, vals) in &hint_labels {
                let have = str_list(&p["labels"][k.as_str()]);
                if vals.iter().any(|v| have.contains(v)) {
                    score += 0.5;
                }
            }
            let mut out = Map::new();
            out.insert("uid".into(), json!(uid));
            out.insert("user".into(), u.clone());
            out.insert("labels".into(), p["labels"].clone());
            out.insert("data".into(), filter_profile_data(&p["data"], &data_keys));
            hits.push((score, Value::Object(out)));
        }
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let total = hits.len();
        let profiles: Vec<Value> = hits.into_iter().take(size).map(|h| h.1).collect();
        Ok(Resp::ok(json!({"took": started.elapsed().as_millis() as u64, "total": {"value": total, "relation": "eq"}, "profiles": profiles})))
    }

    /// `_security/profile/_has_privileges`: the uids whose profile user
    /// (with the roles recorded at activation) holds every requested privilege.
    fn profile_has_privileges(&self) -> R {
        let body = self.body()?;
        let o = body.as_object().cloned().unwrap_or_default();
        for k in o.keys() {
            if !matches!(k.as_str(), "uids" | "privileges") {
                return Err(EsError::x_content_parse(format!("[1:2] [profile_has_privileges_request_payload] unknown field [{k}]")));
            }
        }
        let uids = o.get("uids").map(str_list).ok_or_else(|| EsError::illegal_argument("Required [uids]"))?;
        let privs = o.get("privileges").ok_or_else(|| EsError::illegal_argument("Required [privileges]"))?;
        let sec = &self.node.security;
        let profiles: Vec<(String, Option<Value>)> = {
            let st = sec.state.read();
            uids.iter().map(|u| (u.clone(), st.profiles.get(u).cloned())).collect()
        };
        let mut granted = Vec::new();
        let mut missing = Map::new();
        for (uid, p) in profiles {
            let Some(p) = p else {
                missing.insert(uid, json!({"type": "resource_not_found_exception", "reason": "profile document not found"}));
                continue;
            };
            let username = p["user"]["username"].as_str().unwrap_or("");
            let Some(mut a) = sec.lookup_user(username) else { continue };
            a.roles = str_list(&p["user"]["roles"]);
            let perm = sec.permission(&a);
            let mut ok = true;
            for c in str_list(&privs["cluster"]) {
                ok &= perm.has_cluster_privilege(&c);
            }
            for e in privs["index"].as_array().into_iter().flatten() {
                for n in str_list(&e["names"]) {
                    for pv in str_list(&e["privileges"]) {
                        ok &= perm.has_index_privilege(&n, &pv);
                    }
                }
            }
            for e in privs["application"].as_array().into_iter().flatten() {
                let app = e["application"].as_str().unwrap_or("");
                for r in str_list(&e["resources"]) {
                    for pv in str_list(&e["privileges"]) {
                        ok &= self.role_has_app(&perm.role, app, &r, &pv);
                    }
                }
            }
            if ok {
                granted.push(uid);
            }
        }
        let mut out = json!({"has_privilege_uids": granted});
        if !missing.is_empty() {
            out["errors"] = json!({"count": missing.len(), "details": missing});
        }
        Ok(Resp::ok(out))
    }
}

fn bump_seq(p: &mut Value) {
    let n = p["_doc"]["_seq_no"].as_i64().unwrap_or(0) + 1;
    p["_doc"]["_seq_no"] = json!(n);
}

/// Profile `data` limited to the requested top-level keys (`*` = all, none = empty).
fn filter_profile_data(data: &Value, keys: &[String]) -> Value {
    let mut filtered = Map::new();
    for (k, v) in data.as_object().into_iter().flatten() {
        if keys.iter().any(|d| d == "*" || k == d || privilege::glob(d, k) || k.starts_with(&format!("{d}."))) {
            filtered.insert(k.clone(), v.clone());
        }
    }
    Value::Object(filtered)
}

fn has_terms_lookup(v: &Value) -> bool {
    match v {
        Value::Object(o) => o.iter().any(|(k, x)| {
            (k == "terms" && x.as_object().map(|t| t.values().any(|f| f.get("index").is_some() || f.get("path").is_some())).unwrap_or(false)) || has_terms_lookup(x)
        }),
        Value::Array(a) => a.iter().any(has_terms_lookup),
        _ => false,
    }
}

fn token_name(v: &Value) -> &'static str {
    match v {
        Value::Object(_) => "START_OBJECT",
        Value::Array(_) => "START_ARRAY",
        Value::Number(_) => "VALUE_NUMBER",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Null => "VALUE_NULL",
        Value::String(_) => "VALUE_STRING",
    }
}

/// Glob `p` covers glob/literal `q`.
fn cover(p: &str, q: &str) -> bool {
    if p == "*" {
        return true;
    }
    match p.strip_suffix('*') {
        Some(prefix) if !prefix.contains('*') => q.starts_with(prefix),
        _ => p == q || (!q.contains('*') && privilege::glob(p, q)),
    }
}

fn validate_name(kind: &str, name: &str, max: usize) -> Result<(), EsError> {
    let ok = !name.is_empty() && name.len() <= max && name.trim() == name && name.chars().all(|c| (' '..='~').contains(&c));
    if !ok {
        return Err(validation(&format!(
            "{kind} names must be at least 1 and no more than {max} characters. They can contain alphanumeric characters (a-z, A-Z, 0-9), spaces, punctuation, and printable symbols in the Basic Latin (ASCII) block. Leading or trailing whitespace is not allowed."
        )));
    }
    Ok(())
}

fn validate_app_name(app: &str) -> Result<(), EsError> {
    let mut chars = app.chars();
    let first_ok = chars.next().map(|c| c.is_ascii_lowercase()).unwrap_or(false);
    let rest_ok = app.chars().all(|c| c.is_ascii_alphanumeric() || "_-.*".contains(c) || c == ':' || c == '/');
    if !first_ok || !rest_ok || app.len() < 3 {
        return Err(EsError::illegal_argument(format!(
            "An application name prefix must match the pattern ^[a-z][A-Za-z0-9]*$ (found '{app}')"
        )));
    }
    Ok(())
}

fn valid_token_name(n: &str) -> bool {
    let b = n.as_bytes();
    !b.is_empty() && b.len() <= 256 && b[0] != b'_' && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'-' || *c == b'_')
}
