use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestEntry {
    pub path: String,
    pub name: String,
    pub kind: String, // lib | bin | mixed
    pub role: String, // core | adapter | plugin | tool | test-harness | contracts
    pub description: String,
    pub owner: String,
    pub spec_refs: Vec<String>,
    pub openapi_refs: Vec<String>,
    pub schema_refs: Vec<String>,
    pub binaries: Vec<String>,
    pub features: Vec<String>,
    pub tests: Vec<String>,
    pub docs_paths: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RootWorkspaceToml {
    pub workspace: WorkspaceSection,
}

#[derive(Debug, Deserialize)]
pub struct WorkspaceSection {
    pub members: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct CrateToml {
    pub package: Package,
    #[serde(default)]
    pub lib: Option<toml::Value>,
    #[serde(default)]
    pub bin: Option<Vec<toml::Value>>,
    #[serde(default)]
    pub features: std::collections::BTreeMap<String, toml::Value>,
}

#[derive(Debug, Deserialize)]
pub struct Package {
    pub name: String,
    #[serde(default)]
    pub description: String,
}
