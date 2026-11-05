//! The one source of truth for every version this node reports or gates on:
//! the Elasticsearch release we emulate, the strings ES derives from it
//! (`lucene_version`, build hash/date, minimum wire / index compatibility),
//! the REST API versions a request may ask for (`compatible-with=N`), and
//! the [`IndexVersion`] ids ES records in `index.version.created`.
//!
//! We do not use Lucene: `LUCENE_VERSION` is only the string ES 9.5.5
//! reports (`GET /`, `_segments`, `_cat/segments`).
//!
//! Index-version gating: a default that changed between releases is a
//! function of the index's *creation* version, never of the process
//! version, so indices created by an 8.x build keep 8.x behaviour:
//!
//! ```ignore
//! if svc.created_version().on_or_after(IndexVersion::EXCLUDE_SOURCE_VECTORS_DEFAULT) { ... }
//! ```

/// The Elasticsearch release this node is (`version.number`).
pub const ES_VERSION: &str = "9.5.5";
/// `(major, minor, patch)` of [`ES_VERSION`].
pub const ES_VERSION_TRIPLE: (u32, u32, u32) = (9, 5, 5);
/// `Version.id` of [`ES_VERSION`] (`9_05_05_99`).
pub const ES_VERSION_ID: u32 = 9_05_05_99;
/// `version.lucene_version` as ES 9.5.5 reports it (a string only).
pub const LUCENE_VERSION: &str = "10.5.1";
/// `version.build_hash` of the official 9.5.5 build.
pub const BUILD_HASH: &str = "48acba663b09e70b115dd879329141506c377890";
/// `version.build_date` of the official 9.5.5 build.
pub const BUILD_DATE: &str = "2026-09-30T22:12:03.984586187Z";
pub const BUILD_FLAVOR: &str = "default";
pub const BUILD_TYPE: &str = "tar";
/// `minimum_wire_compatibility_version`: the first release of the previous
/// major's last minor (`Version.computeMinCompatVersion`).
pub const MIN_WIRE_COMPAT: &str = "8.19.0";
/// `minimum_index_compatibility_version` (`IndexVersions.MINIMUM_COMPATIBLE`).
pub const MIN_INDEX_COMPAT: &str = "8.0.0";
/// `TransportVersion.current()` id (nodes info `transport_version`).
pub const TRANSPORT_VERSION: u32 = 9_458_013;
/// Nodes info `component_versions`.
pub const API_KEY_VERSION: u32 = 8_150_099;
pub const ML_CONFIG_VERSION: u32 = 12_000_099;
pub const TRANSFORM_CONFIG_VERSION: u32 = 10_000_099;
/// `IndexVersion.current().toReleaseVersion()`: the releases sharing the
/// current index version (`_cluster/stats` `indices.versions`,
/// `_migration/system_features` `minimum_index_version`).
pub const INDEX_RELEASE_VERSION: &str = "9.5.2-9.5.5";

/// ES `RestApiVersion`: the REST contract a request asked for. 9.x knows
/// only the current major and the previous one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RestApiVersion {
    V8,
    #[default]
    V9,
}

impl RestApiVersion {
    pub const CURRENT: RestApiVersion = RestApiVersion::V9;
    /// `RestApiVersion.minimumSupported()` (= previous major).
    pub const MINIMUM_SUPPORTED: RestApiVersion = RestApiVersion::V8;

    pub fn major(self) -> u8 {
        match self {
            RestApiVersion::V8 => 8,
            RestApiVersion::V9 => 9,
        }
    }

    pub fn for_major(major: u8) -> Option<RestApiVersion> {
        match major {
            8 => Some(RestApiVersion::V8),
            9 => Some(RestApiVersion::V9),
            _ => None,
        }
    }

    /// True for `compatible-with=8` requests (8.x behaviour where it differs).
    pub fn is_v8(self) -> bool {
        self == RestApiVersion::V8
    }
}

thread_local! {
    static CURRENT_REST: std::cell::Cell<RestApiVersion> = const { std::cell::Cell::new(RestApiVersion::V9) };
}

/// The REST API version of the request being served on this thread, for
/// engine code that renders or parses differently under `compatible-with=8`
/// without a request object at hand. The http layer sets it around
/// blocking handler work ([`with_rest_api_version`]); outside a request it
/// is the current version.
pub fn current_rest_api_version() -> RestApiVersion {
    CURRENT_REST.with(|c| c.get())
}

/// Run `f` with [`current_rest_api_version`] = `v` on this thread.
pub fn with_rest_api_version<T>(v: RestApiVersion, f: impl FnOnce() -> T) -> T {
    let prev = CURRENT_REST.with(|c| c.replace(v));
    struct Restore(RestApiVersion);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT_REST.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(prev);
    f()
}

/// ES `IndexVersion`: the id recorded in `index.version.created`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexVersion(pub u32);

impl IndexVersion {
    /// What a new index records (`IndexVersion.current()`).
    pub const CURRENT: IndexVersion = IndexVersion::UPGRADE_TO_LUCENE_10_5_1;
    /// `IndexVersions.MINIMUM_COMPATIBLE`.
    pub const MINIMUM_COMPATIBLE: IndexVersion = IndexVersion::V_8_0_0;
    /// What indices written by our 8.19-targeting builds recorded: an index
    /// whose metadata lacks `index.version.created` reads as this.
    pub const LEGACY_DEFAULT: IndexVersion = IndexVersion::V8_DEPRECATE_SOURCE_MODE_MAPPER;
    /// The last 8.x index version (8.19).
    pub const LATEST_8X: IndexVersion = IndexVersion::MATCH_ONLY_TEXT_STORED_AS_BYTES_BACKPORT_8_X;

    pub fn id(self) -> u32 {
        self.0
    }

    /// Parse an `index.version.created` value (a JSON string or number).
    pub fn parse(s: &str) -> Option<IndexVersion> {
        s.trim().parse::<u32>().ok().map(IndexVersion)
    }

    /// The creation version of an index from its `index.version.created`
    /// setting value; absent / unparseable means an index written by an
    /// older (8.x-targeting) build.
    pub fn from_setting(v: Option<&serde_json::Value>) -> IndexVersion {
        match v {
            Some(serde_json::Value::String(s)) => IndexVersion::parse(s).unwrap_or(IndexVersion::LEGACY_DEFAULT),
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|n| IndexVersion(n as u32)).unwrap_or(IndexVersion::LEGACY_DEFAULT),
            _ => IndexVersion::LEGACY_DEFAULT,
        }
    }

    pub fn major(self) -> u32 {
        self.0 / 1_000_000
    }

    pub fn before(self, other: IndexVersion) -> bool {
        self < other
    }

    pub fn on_or_after(self, other: IndexVersion) -> bool {
        self >= other
    }

    /// `between(lower, upper)`: `lower <= self < upper`.
    pub fn between(self, lower: IndexVersion, upper: IndexVersion) -> bool {
        lower <= self && self < upper
    }

    /// Created by a 9.x release (any 9.x default applies).
    pub fn is_9x(self) -> bool {
        self.major() >= 9
    }

    /// `IndexVersion.toReleaseVersion()` for the versions we create or read.
    pub fn release_version(self) -> String {
        if self == IndexVersion::CURRENT {
            INDEX_RELEASE_VERSION.to_string()
        } else if self == IndexVersion::LATEST_8X {
            "8.19.9-8.19.22".to_string()
        } else {
            self.0.to_string()
        }
    }
}

/// The current version: what a new index records.
impl Default for IndexVersion {
    fn default() -> Self {
        IndexVersion::CURRENT
    }
}

impl std::fmt::Display for IndexVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// `IndexVersions` constants (8.x and 9.x) as ES 9.5.5 defines them.
#[allow(dead_code)]
impl IndexVersion {
    pub const V_8_0_0: IndexVersion = IndexVersion(8_00_00_99);
    pub const V_8_1_0: IndexVersion = IndexVersion(8_01_00_99);
    pub const V_8_2_0: IndexVersion = IndexVersion(8_02_00_99);
    pub const V_8_3_0: IndexVersion = IndexVersion(8_03_00_99);
    pub const V_8_4_0: IndexVersion = IndexVersion(8_04_00_99);
    pub const V_8_5_0: IndexVersion = IndexVersion(8_05_00_99);
    pub const V_8_5_3: IndexVersion = IndexVersion(8_05_03_99);
    pub const V_8_6_0: IndexVersion = IndexVersion(8_06_00_99);
    pub const V_8_7_0: IndexVersion = IndexVersion(8_07_00_99);
    pub const V_8_8_0: IndexVersion = IndexVersion(8_08_00_99);
    pub const V_8_8_2: IndexVersion = IndexVersion(8_08_02_99);
    pub const V_8_9_0: IndexVersion = IndexVersion(8_09_00_99);
    pub const V_8_9_1: IndexVersion = IndexVersion(8_09_01_99);
    pub const V_8_10_0: IndexVersion = IndexVersion(8_10_00_99);
    pub const FIRST_DETACHED_INDEX_VERSION: IndexVersion = IndexVersion(8_500_000);
    pub const NEW_SPARSE_VECTOR: IndexVersion = IndexVersion(8_500_001);
    pub const SPARSE_VECTOR_IN_FIELD_NAMES_SUPPORT: IndexVersion = IndexVersion(8_500_002);
    pub const UPGRADE_LUCENE_9_8: IndexVersion = IndexVersion(8_500_003);
    pub const ES_VERSION_8_12: IndexVersion = IndexVersion(8_500_004);
    pub const NORMALIZED_VECTOR_COSINE: IndexVersion = IndexVersion(8_500_005);
    pub const UPGRADE_LUCENE_9_9: IndexVersion = IndexVersion(8_500_006);
    pub const NORI_DUPLICATES: IndexVersion = IndexVersion(8_500_007);
    pub const UPGRADE_LUCENE_9_9_1: IndexVersion = IndexVersion(8_500_008);
    pub const ES_VERSION_8_12_1: IndexVersion = IndexVersion(8_500_009);
    pub const UPGRADE_8_12_1_LUCENE_9_9_2: IndexVersion = IndexVersion(8_500_010);
    pub const NEW_INDEXVERSION_FORMAT: IndexVersion = IndexVersion(8_501_0_00);
    pub const UPGRADE_LUCENE_9_9_2: IndexVersion = IndexVersion(8_502_0_00);
    pub const TIME_SERIES_ID_HASHING: IndexVersion = IndexVersion(8_502_0_01);
    pub const UPGRADE_TO_LUCENE_9_10: IndexVersion = IndexVersion(8_503_0_00);
    pub const TIME_SERIES_ROUTING_HASH_IN_ID: IndexVersion = IndexVersion(8_504_0_00);
    pub const DEFAULT_DENSE_VECTOR_TO_INT8_HNSW: IndexVersion = IndexVersion(8_505_0_00);
    pub const DOC_VALUES_FOR_IGNORED_META_FIELD: IndexVersion = IndexVersion(8_505_0_01);
    pub const SOURCE_MAPPER_LOSSY_PARAMS_CHECK: IndexVersion = IndexVersion(8_506_0_00);
    pub const SEMANTIC_TEXT_FIELD_TYPE: IndexVersion = IndexVersion(8_507_0_00);
    pub const UPGRADE_TO_LUCENE_9_11: IndexVersion = IndexVersion(8_508_0_00);
    pub const UNIQUE_TOKEN_FILTER_POS_FIX: IndexVersion = IndexVersion(8_509_0_00);
    pub const ADD_SECURITY_MIGRATION: IndexVersion = IndexVersion(8_510_0_00);
    pub const UPGRADE_TO_LUCENE_9_11_1: IndexVersion = IndexVersion(8_511_0_00);
    pub const INDEX_SORTING_ON_NESTED: IndexVersion = IndexVersion(8_512_0_00);
    pub const LENIENT_UPDATEABLE_SYNONYMS: IndexVersion = IndexVersion(8_513_0_00);
    pub const ENABLE_IGNORE_MALFORMED_LOGSDB: IndexVersion = IndexVersion(8_514_0_00);
    pub const MERGE_ON_RECOVERY_VERSION: IndexVersion = IndexVersion(8_515_0_00);
    pub const UPGRADE_TO_LUCENE_9_12: IndexVersion = IndexVersion(8_516_0_00);
    pub const ENABLE_IGNORE_ABOVE_LOGSDB: IndexVersion = IndexVersion(8_517_0_00);
    pub const ADD_ROLE_MAPPING_CLEANUP_MIGRATION: IndexVersion = IndexVersion(8_518_0_00);
    pub const LOGSDB_DEFAULT_IGNORE_DYNAMIC_BEYOND_LIMIT_BACKPORT: IndexVersion = IndexVersion(8_519_0_00);
    pub const TIME_BASED_K_ORDERED_DOC_ID_BACKPORT: IndexVersion = IndexVersion(8_520_0_00);
    pub const V8_DEPRECATE_SOURCE_MODE_MAPPER: IndexVersion = IndexVersion(8_521_0_00);
    pub const USE_SYNTHETIC_SOURCE_FOR_RECOVERY_BACKPORT: IndexVersion = IndexVersion(8_522_0_00);
    pub const UPGRADE_TO_LUCENE_9_12_1: IndexVersion = IndexVersion(8_523_0_00);
    pub const INFERENCE_METADATA_FIELDS_BACKPORT: IndexVersion = IndexVersion(8_524_0_00);
    pub const LOGSB_OPTIONAL_SORTING_ON_HOST_NAME_BACKPORT: IndexVersion = IndexVersion(8_525_0_00);
    pub const USE_SYNTHETIC_SOURCE_FOR_RECOVERY_BY_DEFAULT_BACKPORT: IndexVersion = IndexVersion(8_526_0_00);
    pub const SYNTHETIC_SOURCE_STORE_ARRAYS_NATIVELY_BACKPORT_8_X: IndexVersion = IndexVersion(8_527_0_00);
    pub const ADD_RESCORE_PARAMS_TO_QUANTIZED_VECTORS_BACKPORT_8_X: IndexVersion = IndexVersion(8_528_0_00);
    pub const RESCORE_PARAMS_ALLOW_ZERO_TO_QUANTIZED_VECTORS_BACKPORT_8_X: IndexVersion = IndexVersion(8_529_0_00);
    pub const DEFAULT_OVERSAMPLE_VALUE_FOR_BBQ_BACKPORT_8_X: IndexVersion = IndexVersion(8_530_0_00);
    pub const SEMANTIC_TEXT_DEFAULTS_TO_BBQ_BACKPORT_8_X: IndexVersion = IndexVersion(8_531_0_00);
    pub const INDEX_INT_SORT_INT_TYPE_8_19: IndexVersion = IndexVersion(8_532_0_00);
    pub const MAPPER_TEXT_MATCH_ONLY_MULTI_FIELDS_DEFAULT_NOT_STORED_8_19: IndexVersion = IndexVersion(8_533_0_00);
    pub const UPGRADE_TO_LUCENE_9_12_2: IndexVersion = IndexVersion(8_534_0_00);
    pub const SPARSE_VECTOR_PRUNING_INDEX_OPTIONS_SUPPORT_BACKPORT_8_X: IndexVersion = IndexVersion(8_535_0_00);
    pub const MATCH_ONLY_TEXT_STORED_AS_BYTES_BACKPORT_8_X: IndexVersion = IndexVersion(8_536_0_00);
    pub const UPGRADE_TO_LUCENE_10_0_0: IndexVersion = IndexVersion(9_000_0_00);
    pub const LOGSDB_DEFAULT_IGNORE_DYNAMIC_BEYOND_LIMIT: IndexVersion = IndexVersion(9_001_0_00);
    pub const TIME_BASED_K_ORDERED_DOC_ID: IndexVersion = IndexVersion(9_002_0_00);
    pub const DEPRECATE_SOURCE_MODE_MAPPER: IndexVersion = IndexVersion(9_003_0_00);
    pub const USE_SYNTHETIC_SOURCE_FOR_RECOVERY: IndexVersion = IndexVersion(9_004_0_00);
    pub const INFERENCE_METADATA_FIELDS: IndexVersion = IndexVersion(9_005_0_00);
    pub const LOGSB_OPTIONAL_SORTING_ON_HOST_NAME: IndexVersion = IndexVersion(9_006_0_00);
    pub const SOURCE_MAPPER_MODE_ATTRIBUTE_NOOP: IndexVersion = IndexVersion(9_007_0_00);
    pub const HOSTNAME_DOC_VALUES_SPARSE_INDEX: IndexVersion = IndexVersion(9_008_0_00);
    pub const UPGRADE_TO_LUCENE_10_1_0: IndexVersion = IndexVersion(9_009_0_00);
    pub const USE_SYNTHETIC_SOURCE_FOR_RECOVERY_BY_DEFAULT: IndexVersion = IndexVersion(9_010_00_0);
    pub const TIMESTAMP_DOC_VALUES_SPARSE_INDEX: IndexVersion = IndexVersion(9_011_0_00);
    pub const TIME_SERIES_ID_DOC_VALUES_SPARSE_INDEX: IndexVersion = IndexVersion(9_012_0_00);
    pub const SYNTHETIC_SOURCE_STORE_ARRAYS_NATIVELY_KEYWORD: IndexVersion = IndexVersion(9_013_0_00);
    pub const SYNTHETIC_SOURCE_STORE_ARRAYS_NATIVELY_IP: IndexVersion = IndexVersion(9_014_0_00);
    pub const ADD_RESCORE_PARAMS_TO_QUANTIZED_VECTORS: IndexVersion = IndexVersion(9_015_0_00);
    pub const SYNTHETIC_SOURCE_STORE_ARRAYS_NATIVELY_NUMBER: IndexVersion = IndexVersion(9_016_0_00);
    pub const SYNTHETIC_SOURCE_STORE_ARRAYS_NATIVELY_BOOLEAN: IndexVersion = IndexVersion(9_017_0_00);
    pub const RESCORE_PARAMS_ALLOW_ZERO_TO_QUANTIZED_VECTORS: IndexVersion = IndexVersion(9_018_0_00);
    pub const SYNTHETIC_SOURCE_STORE_ARRAYS_NATIVELY_UNSIGNED_LONG: IndexVersion = IndexVersion(9_019_0_00);
    pub const SYNTHETIC_SOURCE_STORE_ARRAYS_NATIVELY_SCALED_FLOAT: IndexVersion = IndexVersion(9_020_0_00);
    pub const USE_LUCENE101_POSTINGS_FORMAT: IndexVersion = IndexVersion(9_021_0_00);
    pub const UPGRADE_TO_LUCENE_10_2_0: IndexVersion = IndexVersion(9_022_00_0);
    pub const UPGRADE_TO_LUCENE_10_2_1: IndexVersion = IndexVersion(9_023_00_0);
    pub const DEFAULT_OVERSAMPLE_VALUE_FOR_BBQ: IndexVersion = IndexVersion(9_024_0_00);
    pub const SEMANTIC_TEXT_DEFAULTS_TO_BBQ: IndexVersion = IndexVersion(9_025_0_00);
    pub const DEFAULT_TO_ACORN_HNSW_FILTER_HEURISTIC: IndexVersion = IndexVersion(9_026_0_00);
    pub const SEQ_NO_WITHOUT_POINTS: IndexVersion = IndexVersion(9_027_0_00);
    pub const INDEX_INT_SORT_INT_TYPE: IndexVersion = IndexVersion(9_028_0_00);
    pub const MAPPER_TEXT_MATCH_ONLY_MULTI_FIELDS_DEFAULT_NOT_STORED: IndexVersion = IndexVersion(9_029_0_00);
    pub const UPGRADE_TO_LUCENE_10_2_2: IndexVersion = IndexVersion(9_030_0_00);
    pub const SPARSE_VECTOR_PRUNING_INDEX_OPTIONS_SUPPORT: IndexVersion = IndexVersion(9_031_0_00);
    pub const DEFAULT_DENSE_VECTOR_TO_BBQ_HNSW: IndexVersion = IndexVersion(9_032_0_00);
    pub const MATCH_ONLY_TEXT_STORED_AS_BYTES: IndexVersion = IndexVersion(9_033_0_00);
    pub const BACKPORT_91_SECURITY_MIGRATIONS_METADATA_FLATTENED_UPDATE: IndexVersion = IndexVersion(9_033_0_01);
    pub const IGNORED_SOURCE_COALESCED_ENTRIES_WITH_FF: IndexVersion = IndexVersion(9_034_0_00);
    pub const EXCLUDE_SOURCE_VECTORS_DEFAULT: IndexVersion = IndexVersion(9_035_0_00);
    pub const DISABLE_NORMS_BY_DEFAULT_FOR_LOGSDB_AND_TSDB: IndexVersion = IndexVersion(9_036_0_00);
    pub const TSID_CREATED_DURING_ROUTING: IndexVersion = IndexVersion(9_037_0_00);
    pub const UPGRADE_TO_LUCENE_10_3_0: IndexVersion = IndexVersion(9_038_0_00);
    pub const IGNORED_SOURCE_COALESCED_ENTRIES: IndexVersion = IndexVersion(9_039_0_00);
    pub const BACKPORT_UPGRADE_TO_LUCENE_10_3_1: IndexVersion = IndexVersion(9_039_0_01);
    pub const BACKPORT_UPGRADE_TO_LUCENE_10_3_2: IndexVersion = IndexVersion(9_039_0_02);
    pub const BACKPORT_SECURITY_MIGRATIONS_METADATA_FLATTENED_UPDATE: IndexVersion = IndexVersion(9_039_0_03);
    pub const KEYWORD_MULTI_FIELDS_NOT_STORED_WHEN_IGNORED: IndexVersion = IndexVersion(9_040_0_00);
    pub const UPGRADE_TO_LUCENE_10_3_1: IndexVersion = IndexVersion(9_041_0_00);
    pub const REENABLED_TIMESTAMP_DOC_VALUES_SPARSE_INDEX: IndexVersion = IndexVersion(9_042_0_00);
    pub const SKIPPERS_ENABLED_BY_DEFAULT: IndexVersion = IndexVersion(9_043_0_00);
    pub const TIME_SERIES_USE_SYNTHETIC_ID: IndexVersion = IndexVersion(9_044_0_00);
    pub const TIME_SERIES_DIMENSIONS_USE_SKIPPERS: IndexVersion = IndexVersion(9_045_0_00);
    pub const TIME_SERIES_ALL_FIELDS_USE_SKIPPERS: IndexVersion = IndexVersion(9_046_0_00);
    pub const UPGRADE_TO_LUCENE_10_3_2: IndexVersion = IndexVersion(9_047_0_00);
    pub const SECURITY_MIGRATIONS_METADATA_FLATTENED_UPDATE: IndexVersion = IndexVersion(9_048_0_00);
    pub const STANDARD_INDEXES_USE_SKIPPERS: IndexVersion = IndexVersion(9_049_0_00);
    pub const NESTED_PATH_LIMIT: IndexVersion = IndexVersion(9_050_0_00);
    pub const GENERIC_DENSE_VECTOR_FORMAT: IndexVersion = IndexVersion(9_051_0_00);
    pub const SKIPPER_DEFAULTS_ONLY_ON_TSDB: IndexVersion = IndexVersion(9_052_0_00);
    pub const DISK_BBQ_LICENSE_ENFORCEMENT: IndexVersion = IndexVersion(9_053_0_00);
    pub const STORE_IGNORED_KEYWORDS_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_054_0_00);
    pub const TIME_SERIES_USE_STORED_FIELDS_BLOOM_FILTER_FOR_ID: IndexVersion = IndexVersion(9_055_0_00);
    pub const AGG_METRIC_DOUBLE_REMOVE_POINTS: IndexVersion = IndexVersion(9_056_0_00);
    pub const STATELESS_SKIPPERS_ENABLED_FOR_TSDB: IndexVersion = IndexVersion(9_057_0_00);
    pub const TEXT_FIELDS_STORED_IN_IGNORED_SOURCE_FIX: IndexVersion = IndexVersion(9_058_0_00);
    pub const DEFAULT_HNSW_EARLY_TERMINATION: IndexVersion = IndexVersion(9_059_0_00);
    pub const PATTERN_TEXT_ARGS_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_060_0_00);
    pub const DENSE_VECTOR_BFLOAT16_DEFAULT_INDEX_OPTIONS_BACKPORT: IndexVersion = IndexVersion(9_060_0_01);
    pub const KEYWORD_FIELDS_KEEP_DUPLICATES_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_061_0_00);
    pub const STORE_FALLBACK_TEXT_FIELDS_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_062_0_00);
    pub const STORE_FALLBACK_MOT_FIELDS_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_063_0_00);
    pub const MOD_ROUTING_FUNCTION: IndexVersion = IndexVersion(9_064_0_00);
    pub const FALLBACK_TEXT_FIELDS_BINARY_DOC_VALUES_FORMAT_CHECK: IndexVersion = IndexVersion(9_065_0_00);
    pub const READ_SI_FILES_FROM_MEMORY_FOR_HOLLOW_COMMITS: IndexVersion = IndexVersion(9_066_0_00);
    pub const FLATTENED_FIELD_TSDB_CODEC_USE_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_067_0_00);
    pub const STORE_PATTERN_TEXT_FIELDS_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_068_0_00);
    pub const DISK_BBQ_QUANTIZE_BITS: IndexVersion = IndexVersion(9_069_0_00);
    pub const ID_FIELD_USE_ES812_POSTINGS_FORMAT: IndexVersion = IndexVersion(9_070_0_00);
    pub const TIME_SERIES_USE_SYNTHETIC_ID_94: IndexVersion = IndexVersion(9_071_0_00);
    pub const TIME_SERIES_DOC_VALUES_FORMAT_VERSION_3: IndexVersion = IndexVersion(9_072_0_00);
    pub const STORE_IGNORED_MALFORMED_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_073_0_00);
    pub const DISABLE_SEQUENCE_NUMBERS: IndexVersion = IndexVersion(9_074_0_00);
    pub const UPGRADE_TO_LUCENE_10_4_0: IndexVersion = IndexVersion(9_075_00_0);
    pub const UPGRADE_DISKBBQ_ES940: IndexVersion = IndexVersion(9_076_00_0);
    pub const FLATTENED_FIELD_NO_ROOT_DOC_VALUES: IndexVersion = IndexVersion(9_077_0_00);
    pub const IGNORED_SOURCE_AS_DOC_VALUES: IndexVersion = IndexVersion(9_078_0_00);
    pub const TIME_SERIES_USE_SYNTHETIC_ID_DEFAULT: IndexVersion = IndexVersion(9_079_0_00);
    pub const TSID_SINGLE_PREFIX_BYTE_FEATURE_FLAG: IndexVersion = IndexVersion(9_080_00_0);
    pub const SKIPPERS_ENABLED_BY_DEFAULT_IN_LOGSDB: IndexVersion = IndexVersion(9_081_0_00);
    pub const STORE_IGNORED_WILDCARD_FIELDS_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_082_0_00);
    pub const WRITE_TSID_PREFIX_PARTITION: IndexVersion = IndexVersion(9_083_0_00);
    pub const SEMANTIC_TEXT_DEFAULTS_TO_JINA_V5: IndexVersion = IndexVersion(9_084_0_00);
    pub const TIME_SERIES_DISABLE_SEQUENCE_NUMBERS: IndexVersion = IndexVersion(9_085_0_00);
    pub const DENSE_VECTOR_BFLOAT16_DEFAULT_INDEX_OPTIONS: IndexVersion = IndexVersion(9_086_0_00);
    pub const DEFAULT_DENSE_VECTOR_TO_BBQ_DISK: IndexVersion = IndexVersion(9_087_0_00);
    pub const SEMANTIC_TEXT_DEFAULTS_TO_BFLOAT16: IndexVersion = IndexVersion(9_088_0_00);
    pub const STORE_IGNORED_FLATTENED_FIELDS_IN_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_089_0_00);
    pub const TIME_SERIES_USE_SYNTHETIC_ID_BEST_COMPRESSION: IndexVersion = IndexVersion(9_090_0_00);
    pub const TSID_SINGLE_PREFIX_BYTE: IndexVersion = IndexVersion(9_091_0_00);
    pub const SEMANTIC_TEXT_USES_DENSE_VECTOR_DEFAULT_INDEX_OPTIONS: IndexVersion = IndexVersion(9_092_0_00);
    pub const TIME_SERIES_USE_SYNTHETIC_ID_DEFAULT_PROD: IndexVersion = IndexVersion(9_093_0_00);
    pub const TIME_SERIES_DISABLE_SEQUENCE_NUMBERS_DEFAULT: IndexVersion = IndexVersion(9_094_0_00);
    pub const DEPRECATE_INTEGRATED_COUNTS_BINARY_DOC_VALUES: IndexVersion = IndexVersion(9_095_0_00);
    pub const SEMANTIC_TEXT_LEGACY_FORMAT_FORBIDDEN: IndexVersion = IndexVersion(9_096_00_0);
    pub const SEMANTIC_FIELD_TYPE: IndexVersion = IndexVersion(9_097_00_0);
    pub const ES95_TSDB_CODEC_FEATURE_FLAG: IndexVersion = IndexVersion(9_098_0_00);
    pub const SHARD_OBLIVIOUS_SLICING: IndexVersion = IndexVersion(9_099_0_00);
    pub const DOC_COUNT_FIELD_USES_NUMERIC_DOC_VALUES: IndexVersion = IndexVersion(9_100_0_00);
    pub const MAPPING_ID_MODE_DEFAULT: IndexVersion = IndexVersion(9_101_0_00);
    pub const TIME_SERIES_ES95_CODEC_DEFAULT_FEATURE_FLAG: IndexVersion = IndexVersion(9_102_0_00);
    pub const SEMANTIC_TEXT_ORIGINAL_VALUES_DOC_VALUES: IndexVersion = IndexVersion(9_103_0_00);
    pub const UPGRADE_TO_LUCENE_10_5_0: IndexVersion = IndexVersion(9_104_0_00);
    pub const DISK_BBQ_ES950_AUTO_CALIBRATE: IndexVersion = IndexVersion(9_105_0_00);
    pub const TIME_SERIES_ES95_CODEC_DEFAULT: IndexVersion = IndexVersion(9_106_0_00);
    pub const IGNORED_SOURCE_AS_DOC_VALUES_NO_FF: IndexVersion = IndexVersion(9_107_0_00);
    pub const UPGRADE_TO_LUCENE_10_5_1: IndexVersion = IndexVersion(9_111_0_00);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_versions() {
        assert_eq!(IndexVersion::CURRENT.id(), 9_111_000);
        assert_eq!(IndexVersion::MINIMUM_COMPATIBLE.id(), 8_000_099);
        assert!(IndexVersion::LEGACY_DEFAULT.before(IndexVersion::UPGRADE_TO_LUCENE_10_0_0));
        assert!(!IndexVersion::LEGACY_DEFAULT.is_9x());
        assert!(IndexVersion::CURRENT.on_or_after(IndexVersion::EXCLUDE_SOURCE_VECTORS_DEFAULT));
        assert_eq!(IndexVersion::from_setting(None), IndexVersion::LEGACY_DEFAULT);
        assert_eq!(IndexVersion::from_setting(Some(&serde_json::json!("9111000"))), IndexVersion::CURRENT);
        assert_eq!(IndexVersion::CURRENT.release_version(), "9.5.2-9.5.5");
    }

    #[test]
    fn rest_api_version_scope() {
        assert_eq!(current_rest_api_version(), RestApiVersion::V9);
        with_rest_api_version(RestApiVersion::V8, || assert!(current_rest_api_version().is_v8()));
        assert_eq!(current_rest_api_version(), RestApiVersion::V9);
    }
}
