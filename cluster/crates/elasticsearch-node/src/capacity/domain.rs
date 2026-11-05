//! Capacity-domain facts are separate from process RSS and from node attributes.
//! Unknown identity never establishes an independent target host or budget.
use crate::node::ClusterNode;
use crate::front::SharedText;
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use std::{io::Read, sync::Arc};

pub(super) const ACTION: &str = "internal:monitor/capacity_domain/v1";
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Domain {
    /// Linux boot identity: same kernel/host capacity is not independent even
    /// when the processes occupy different cgroup subtrees.
    #[serde(deserialize_with="crate::control_memory::decode")]
    pub boot_id: SharedText,
    #[serde(deserialize_with="crate::control_memory::decode")]
    pub cgroup_path: SharedText,
    pub cgroup_device: u64,
    pub cgroup_inode: u64,
}
impl Domain {
    pub(crate) fn independent_host(&self, other: &Self) -> bool { self.boot_id != other.boot_id }
    pub(crate) fn same_budget_domain(&self, other: &Self) -> bool {
        self.boot_id == other.boot_id && self.cgroup_device == other.cgroup_device && self.cgroup_inode == other.cgroup_inode
    }
}
#[derive(Serialize, Deserialize)]
pub(super) struct Request { #[serde(deserialize_with="crate::control_memory::decode")] pub expected_ephemeral_id: SharedText }
#[derive(Serialize, Deserialize)]
pub(super) struct Reply {
    pub version: u8,
    #[serde(deserialize_with="crate::control_memory::decode")]
    pub node_id: SharedText,
    #[serde(deserialize_with="crate::control_memory::decode")]
    pub ephemeral_id: SharedText,
    pub domain: Option<Domain>,
}
fn bounded_text(path: &str, limit: usize) -> Option<SharedText> {
    let capacity=limit.checked_add(1)?;
    let _memory=elasticsearch_engine::memory::breakers().reserve_native(capacity as u64,"capacity domain file read").ok()?;
    let mut bytes=Vec::new();bytes.try_reserve_exact(capacity).ok()?;bytes.resize(capacity,0);
    let mut file=std::fs::File::open(path).ok()?.take(capacity as u64);let mut used=0;
    while used<capacity{let n=file.read(&mut bytes[used..]).ok()?;if n==0{break}used+=n;}
    if used>limit{return None}
    crate::control_memory::text(std::str::from_utf8(&bytes[..used]).ok()?).ok()
}
fn valid_boot_id(id: &str) -> bool {
    id.len() == 36 && id.bytes().enumerate().all(|(i,b)| if [8,13,18,23].contains(&i) {b==b'-'} else {b.is_ascii_hexdigit()})
}
fn unified_path(text: &str) -> Option<&str> {
    let mut entries = text.lines().filter_map(|line| line.strip_prefix("0::"));
    let path = entries.next()?;
    if entries.next().is_some() || !path.starts_with('/') || path.split('/').any(|p| p==".." || p==".") { return None; }
    Some(path)
}
fn local() -> Option<Domain> {
    #[cfg(target_os="linux")]
    {
        use std::os::unix::fs::MetadataExt;
        let boot = bounded_text("/proc/sys/kernel/random/boot_id", 64)?;
        let boot = boot.trim();
        if !valid_boot_id(boot) {return None}
        let mut lower=[0u8;36];for (out,byte) in lower.iter_mut().zip(boot.bytes()){*out=byte.to_ascii_lowercase();}
        let boot_id=crate::control_memory::text(std::str::from_utf8(&lower).ok()?).ok()?;
        if !valid_boot_id(&boot_id) { return None; }
        let membership = bounded_text("/proc/self/cgroup", 16384)?;
        let path = unified_path(&membership)?;
        // The cgroup namespace may mount this process's subtree as root. Only
        // accept a path whose actual directory can be resolved; never replace
        // unknown membership with the root cgroup or a process ID.
        let filesystem_path=crate::control_memory::formatted(format_args!("/sys/fs/cgroup{path}")).ok()?;
        let metadata = std::fs::metadata(filesystem_path.as_str()).ok()?;
        if !metadata.is_dir() { return None; }
        return Some(Domain { boot_id, cgroup_path:crate::control_memory::text(path).ok()?, cgroup_device:metadata.dev(), cgroup_inode:metadata.ino() });
    }
    #[cfg(not(target_os="linux"))]
    { None }
}
pub(super) async fn fetch(node: &ClusterNode, peer: &elasticsearch_cluster::state::DiscoveryNode) -> Option<Domain> {
    if !crate::internal_capabilities::supports(peer, crate::internal_capabilities::Capability::CapacityDomainV1) { return None; }
    // Caller already verified the live capability probe in this bounded round.
    let reply: Reply = node.transport.send_request(peer.transport_address.parse().ok()?, ACTION,
        &Request { expected_ephemeral_id:crate::control_memory::text(&peer.ephemeral_id).ok()? }, std::time::Duration::from_secs(2)).await.ok()?;
    if reply.version != 1 || reply.node_id != peer.id || reply.ephemeral_id != peer.ephemeral_id { return None; }
    let domain = reply.domain?;
    if !valid_boot_id(&domain.boot_id) || !domain.cgroup_path.starts_with('/') { return None; }
    Some(domain)
}
pub(super) fn register(node: &Arc<ClusterNode>) {
    let node = node.clone();
    let transport = node.transport.clone();
    transport.register_handler(ACTION, move |request: Request| {
        let node = node.clone();
        async move {
            if request.expected_ephemeral_id != node.local.ephemeral_id { return Err(RemoteError::internal("capacity domain process changed")); }
            Ok::<_,RemoteError>(Reply { version:1, node_id:crate::control_memory::text(&node.local.id).map_err(|e|RemoteError::new(e.kind,e.reason,e.status))?, ephemeral_id:crate::control_memory::text(&node.local.ephemeral_id).map_err(|e|RemoteError::new(e.kind,e.reason,e.status))?, domain:local() })
        }
    });
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decoded_domain_clones_retain_backing_and_legacy_wire() {
        use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
        let b=Breakers::leak_for_tests(1<<20);b.apply_settings(&|k|(k=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=RequestScope::new("domain wire",b);
        let bytes=bincode::serialize(&("boot", "/scope".repeat(512),1u64,2u64)).unwrap();
        let value:Domain=scoped(Some(scope.clone()),||bincode::deserialize(&bytes)).unwrap();
        assert_eq!(bincode::serialize(&value).unwrap(),bytes);let copy=value.clone();drop(value);drop(bytes);drop(scope);
        assert!(b.get(Kind::Accounting).used()>0);assert!(copy.cgroup_path.starts_with("/scope"));drop(copy);assert_eq!(b.get(Kind::Accounting).used(),0);
    }
    #[test]
    fn cgroup_identity_is_not_invented_for_unknown_or_escaped_membership() {
        assert_eq!(unified_path("0::/tenant/worker\n"), Some("/tenant/worker"));
        for input in ["1:memory:/x", "0::/../x", "0::/a/./b", "0::relative", "0::/a\n0::/b"] { assert_eq!(unified_path(input),None); }
        assert!(valid_boot_id("12345678-1234-1234-1234-123456789abc"));
        assert!(!valid_boot_id("node-1"));
    }
    #[test]
    fn distinct_processes_or_cgroups_do_not_create_host_capacity() {
        let a=Domain{boot_id:crate::control_memory::text("same-host").unwrap(),cgroup_path:crate::control_memory::text("/a").unwrap(),cgroup_device:1,cgroup_inode:2};
        let mut b=a.clone(); b.cgroup_path=crate::control_memory::text("/b").unwrap(); b.cgroup_inode=3;
        assert!(!a.independent_host(&b)); assert!(!a.same_budget_domain(&b));
        b=a.clone(); assert!(a.same_budget_domain(&b));
        b.boot_id=crate::control_memory::text("different-host").unwrap(); assert!(a.independent_host(&b)); assert!(!a.same_budget_domain(&b));
    }
}
