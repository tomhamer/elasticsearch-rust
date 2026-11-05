//! CO2 and cost estimation (ES `CO2Calculator`, `CostCalculator`,
//! `CloudProviders`, `InstanceTypeService`, `HostMetadata`).

use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

const DEFAULT_SAMPLING_FREQUENCY: f64 = 20.0;
const DEFAULT_CO2_TONS_PER_KWH: f64 = 0.000379069;
const DEFAULT_KILOWATTS_PER_CORE_X86: f64 = 7.0 / 1000.0;
const DEFAULT_KILOWATTS_PER_CORE_ARM64: f64 = 2.8 / 1000.0;
const DEFAULT_KILOWATTS_PER_CORE: f64 = DEFAULT_KILOWATTS_PER_CORE_X86;
const DEFAULT_DATACENTER_PUE: f64 = 1.7;
const SECONDS_PER_HOUR: f64 = 3600.0;
const SECONDS_PER_YEAR: f64 = SECONDS_PER_HOUR * 24.0 * 365.0;
pub const DEFAULT_COST_USD_PER_CORE_HOUR: f64 = 0.0425;
pub const DEFAULT_PROFILING_NUM_CORES: i32 = 4;

/// (provider, region, instance type) -> USD per hour.
type CostTable = HashMap<(String, String, String), f64>;

static COSTS: OnceLock<CostTable> = OnceLock::new();

/// Install the instance cost tables (`profiling-costs-{aws,azure}.json`,
/// decompressed by the caller). Later calls are ignored.
pub fn install_costs(tables: &[(&str, &[u8])]) {
    COSTS.get_or_init(|| {
        let mut out = CostTable::new();
        for (provider, json) in tables {
            let Ok(Value::Array(rows)) = serde_json::from_slice::<Value>(json) else { continue };
            for r in rows {
                let region = r["region"].as_str().unwrap_or("").to_string();
                let ty = r["instance_type"].as_str().unwrap_or("").to_string();
                let usd = match &r["usd_per_hour"] {
                    Value::Number(n) => n.as_f64().unwrap_or(0.0),
                    _ => DEFAULT_COST_USD_PER_CORE_HOUR * DEFAULT_PROFILING_NUM_CORES as f64,
                };
                out.insert((provider.to_string(), region, ty), usd);
            }
        }
        out
    });
}

pub fn costs_installed() -> bool {
    COSTS.get().is_some()
}

fn cost_of(it: &InstanceType) -> Option<f64> {
    COSTS.get()?.get(&(it.provider.clone(), it.region.clone(), it.name.clone())).copied()
}

#[derive(Clone, Default, Debug)]
pub struct InstanceType {
    pub provider: String,
    pub region: String,
    pub name: String,
}

fn as_str(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str).map(str::to_string)
}

impl InstanceType {
    /// `InstanceType.fromHostSource` (reads dotted top-level keys).
    pub fn from_host_source(src: &serde_json::Map<String, Value>) -> InstanceType {
        let mk = |p: Option<String>, r: Option<String>, n: Option<String>| InstanceType { provider: p.unwrap_or_default(), region: r.unwrap_or_default(), name: n.unwrap_or_default() };
        if let Some(p) = as_str(src.get("cloud.provider")) {
            return mk(Some(p), as_str(src.get("cloud.region")), as_str(src.get("host.type")));
        }
        if let Some(r) = as_str(src.get("ec2.placement.region")) {
            return mk(Some("aws".into()), Some(r), as_str(src.get("ec2.instance_type")));
        }
        if let Some(zone) = as_str(src.get("gce.instance.zone")) {
            let mut region = zone.rsplit('/').next().unwrap_or("").to_string();
            let tokens: Vec<&str> = region.splitn(3, '-').collect();
            if tokens.len() > 2 {
                region = format!("{}-{}", tokens[0], tokens[1]);
            }
            return mk(Some("gcp".into()), Some(region), None);
        }
        if let Some(r) = as_str(src.get("azure.compute.location")) {
            return mk(Some("azure".into()), Some(r), as_str(src.get("azure.compute.vmsize")));
        }
        let (mut provider, mut region) = (None, None);
        let tags: Vec<String> = match src.get("profiling.host.tags") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
            Some(Value::String(s)) => vec![s.clone()],
            _ => vec![],
        };
        for t in tags {
            let lower = t.to_lowercase();
            let Some((k, v)) = lower.split_once(':') else { continue };
            if k == "cloud_provider" {
                provider = Some(v.to_string());
            }
            if k == "cloud_region" {
                region = Some(v.to_string());
            }
        }
        mk(provider, region, None)
    }
}

#[derive(Clone, Debug)]
pub struct HostMetadata {
    pub host_id: String,
    pub instance_type: InstanceType,
    pub arch: String,
    pub num_cores: i32,
}

impl HostMetadata {
    pub fn from_source(src: &serde_json::Map<String, Value>) -> HostMetadata {
        let arch = as_str(src.get("host.arch")).or_else(|| as_str(src.get("profiling.host.machine")));
        let cores = src.get("profiling.agent.config.present_cpu_cores").and_then(Value::as_i64).map(|v| v as i32);
        HostMetadata {
            host_id: as_str(src.get("host.id")).unwrap_or_default(),
            instance_type: InstanceType::from_host_source(src),
            arch: arch.unwrap_or_default(),
            num_cores: cores.unwrap_or(DEFAULT_PROFILING_NUM_CORES),
        }
    }
}

fn provider_pue(p: &str) -> Option<f64> {
    match p {
        "aws" => Some(1.135),
        "gcp" => Some(1.1),
        "azure" => Some(1.185),
        _ => None,
    }
}

fn provider_co2(p: &str, region: &str) -> Option<Option<f64>> {
    let table: &[(&str, f64)] = match p {
        "aws" => &[
            ("us-east-1", 0.000379069), ("us-east-2", 0.000410608), ("us-west-1", 0.000322167), ("us-west-2", 0.000322167),
            ("us-gov-east-1", 0.000379069), ("us-gov-west-1", 0.000322167), ("af-south-1", 0.0009006), ("ap-east-1", 0.00071),
            ("ap-south-1", 0.0007082), ("ap-northeast-3", 0.0004658), ("ap-northeast-2", 0.0004156), ("ap-southeast-1", 0.000408),
            ("ap-southeast-2", 0.00076), ("ap-southeast-3", 0.0007177), ("ap-northeast-1", 0.0004658), ("ca-central-1", 0.00012),
            ("cn-north-1", 0.0005374), ("cn-northwest-1", 0.0005374), ("eu-central-1", 0.000311), ("eu-west-1", 0.0002786),
            ("eu-west-2", 0.000225), ("eu-south-1", 0.0002134), ("eu-west-3", 0.0000511), ("eu-north-1", 0.0000088),
            ("me-south-1", 0.0005059), ("me-central-1", 0.0004041), ("sa-east-1", 0.0000617),
        ],
        "gcp" => &[
            ("us-central1", 0.0002152373529), ("us-central2", 0.0002152373529), ("us-east1", 0.0003255), ("us-east4", 0.00011124),
            ("us-east5", 0.00011124), ("us-west1", 0.0000072), ("us-west2", 0.0000893), ("us-west3", 0.00030912), ("us-west4", 0.00028835),
            ("us-south1", 0.0001776), ("asia-east1", 0.00037848), ("asia-east2", 0.0002592), ("asia-northeast1", 0.00038976),
            ("asia-northeast2", 0.00026496), ("asia-northeast3", 0.00029325), ("asia-south1", 0.000603), ("asia-south2", 0.00061732),
            ("asia-southeast1", 0.00035712), ("asia-southeast2", 0.0005046), ("australia-southeast1", 0.00047242),
            ("australia-southeast2", 0.00035949), ("europe-central2", 0.0004608), ("europe-north1", 0.00001143),
            ("europe-southwest1", 0.000121), ("europe-west1", 0.0000198), ("europe-west2", 0.00007396), ("europe-west3", 0.0001076),
            ("europe-west4", 0.00013301), ("europe-west6", 0.0000129), ("europe-west8", 0.000298), ("europe-west9", 0.000059),
            ("northamerica-northeast1", 0.0), ("northamerica-northeast2", 0.00000232), ("southamerica-east1", 0.00002838),
            ("southamerica-west1", 0.0000589),
        ],
        "azure" => &[
            ("centralus", 0.000426254), ("centraluseuap", 0.000426254), ("centralusestage", 0.000426254), ("eastus", 0.000379069),
            ("useast", 0.000379069), ("eastusstage", 0.000379069), ("eastus2", 0.000379069), ("useast2", 0.000379069),
            ("eastus2euap", 0.000379069), ("eastus2stage", 0.000379069), ("eastus3", 0.000379069), ("usnorth", 0.000410608),
            ("northcentralus", 0.000410608), ("northcentralusstage", 0.000410608), ("southcentralus", 0.000373231),
            ("southcentralusstage", 0.000373231), ("unitedstates", 0.000373231), ("unitedstateseuap", 0.000373231),
            ("westcentralus", 0.000322167), ("westcentralusstage", 0.000322167), ("westus", 0.000322167), ("westusstage", 0.000322167),
            ("westus2", 0.000322167), ("westus2stage", 0.000322167), ("westus3", 0.000322167), ("asia", 0.0005647),
            ("asiapacific", 0.0005647), ("eastasia", 0.00071), ("eastasiastage", 0.00071), ("southeastasia", 0.000408),
            ("asiasoutheast", 0.000408), ("southafricanorth", 0.0009006), ("southafricawest", 0.0009006), ("southafrica", 0.0009006),
            ("australia", 0.00079), ("australiacentral", 0.00079), ("australiacentral2", 0.00079), ("australiaeast", 0.00079),
            ("australiasoutheast", 0.00096), ("apeast", 0.00071), ("apsoutheast", 0.000408), ("japan", 0.0004658),
            ("japanwest", 0.0004658), ("japaneast", 0.0004658), ("korea", 0.0004156), ("koreaeast", 0.0004156), ("koreasouth", 0.0004156),
            ("india", 0.0007082), ("indiacentral", 0.0007082), ("centralindia", 0.0007082), ("jioindiacentral", 0.0007082),
            ("indiawest", 0.0007082), ("westindia", 0.0007082), ("jioindiawest", 0.0007082), ("indiasouth", 0.0007082),
            ("southindia", 0.0007082), ("northeurope", 0.0002786), ("europenorth", 0.0002786), ("westeurope", 0.0003284),
            ("europewest", 0.0003284), ("france", 0.00005128), ("francecentral", 0.00005128), ("francesouth", 0.00005128),
            ("swedencentral", 0.00000567), ("switzerland", 0.00001152), ("switzerlandnorth", 0.00001152), ("switzerlandwest", 0.00001152),
            ("uk", 0.000225), ("uksouth", 0.000225), ("ukwest", 0.000228), ("germany", 0.00033866), ("germanynorth", 0.00033866),
            ("germanywestcentral", 0.00033866), ("norway", 0.00000762), ("norwayeast", 0.00000762), ("norwaywest", 0.00000762),
            ("uae", 0.0004041), ("uaenorth", 0.0004041), ("uaecentral", 0.0004041), ("canada", 0.00012), ("canadacentral", 0.00012),
            ("canadaeast", 0.00012), ("brazil", 0.0000617), ("brazilsouth", 0.0000617), ("brazilsoutheast", 0.0000617),
        ],
        _ => return None,
    };
    Some(table.iter().find(|(r, _)| *r == region).map(|(_, v)| *v))
}

/// User overrides from the request.
#[derive(Clone, Default, Debug)]
pub struct CostParams {
    pub co2_per_kwh: Option<f64>,
    pub datacenter_pue: Option<f64>,
    pub per_core_watt_x86: Option<f64>,
    pub per_core_watt_arm64: Option<f64>,
    pub cost_per_core_hour: Option<f64>,
    pub aws_cost_factor: Option<f64>,
    pub azure_cost_factor: Option<f64>,
}

pub fn annual_core_hours(duration: f64, samples: f64, freq: f64) -> f64 {
    (SECONDS_PER_YEAR / duration * samples / freq) / SECONDS_PER_HOUR
}

pub struct Calculator<'a> {
    hosts: &'a HashMap<String, HostMetadata>,
    duration: f64,
    co2_per_kwh: f64,
    pue: f64,
    kw_x86: f64,
    kw_arm: f64,
    cost_per_core_hour: f64,
    aws_factor: f64,
    azure_factor: f64,
}

impl<'a> Calculator<'a> {
    pub fn new(hosts: &'a HashMap<String, HostMetadata>, duration: f64, p: &CostParams) -> Self {
        Calculator {
            hosts,
            duration: if duration > 0.0 { duration } else { 1.0 },
            co2_per_kwh: p.co2_per_kwh.unwrap_or(DEFAULT_CO2_TONS_PER_KWH),
            pue: p.datacenter_pue.unwrap_or(DEFAULT_DATACENTER_PUE),
            kw_x86: p.per_core_watt_x86.map(|w| w / 1000.0).unwrap_or(DEFAULT_KILOWATTS_PER_CORE_X86),
            kw_arm: p.per_core_watt_arm64.map(|w| w / 1000.0).unwrap_or(DEFAULT_KILOWATTS_PER_CORE_ARM64),
            cost_per_core_hour: p.cost_per_core_hour.unwrap_or(DEFAULT_COST_USD_PER_CORE_HOUR),
            aws_factor: p.aws_cost_factor.unwrap_or(1.0),
            azure_factor: p.azure_cost_factor.unwrap_or(1.0),
        }
    }

    pub fn annual_co2_tons(&self, host_id: &str, samples: i64) -> f64 {
        let hours = annual_core_hours(self.duration, samples as f64, DEFAULT_SAMPLING_FREQUENCY);
        let Some(h) = self.hosts.get(host_id) else {
            return DEFAULT_KILOWATTS_PER_CORE * self.co2_per_kwh * hours * self.pue;
        };
        let kw = match h.arch.as_str() {
            "arm64" | "aarch64" => self.kw_arm,
            "amd64" | "x86_64" => self.kw_x86,
            _ => DEFAULT_KILOWATTS_PER_CORE,
        };
        let co2 = match provider_co2(&h.instance_type.provider, &h.instance_type.region) {
            None | Some(None) => self.co2_per_kwh,
            Some(Some(v)) => v,
        };
        let pue = provider_pue(&h.instance_type.provider).unwrap_or(self.pue);
        kw * co2 * hours * pue
    }

    pub fn annual_costs_usd(&self, host_id: &str, samples: f64) -> f64 {
        let hours = annual_core_hours(self.duration, samples, DEFAULT_SAMPLING_FREQUENCY);
        let Some(h) = self.hosts.get(host_id) else {
            return hours * self.cost_per_core_hour;
        };
        let factor = match h.instance_type.provider.as_str() {
            "aws" => self.aws_factor,
            "azure" => self.azure_factor,
            _ => 1.0,
        };
        match cost_of(&h.instance_type) {
            None => hours * self.cost_per_core_hour * factor,
            Some(usd) => hours * (usd / h.num_cores as f64) * factor,
        }
    }
}
