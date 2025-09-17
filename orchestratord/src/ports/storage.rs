pub type Artifact = serde_json::Value;
pub type ArtifactId = String;

pub trait ArtifactStore: Send + Sync {
    fn put(&self, doc: Artifact) -> anyhow::Result<ArtifactId>;
    fn get(&self, id: &ArtifactId) -> anyhow::Result<Option<Artifact>>;
}
