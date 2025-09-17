#[derive(Debug, Clone)]
pub struct AdapterProps {
    pub ctx_max: i32,
}

#[derive(Debug, Clone)]
pub struct StreamRequest { pub prompt: String }
#[derive(Debug)]
pub struct AdapterStream; // placeholder

pub trait AdapterRegistry: Send + Sync {
    fn engines(&self) -> Vec<String>;
    fn props(&self, engine: &str) -> anyhow::Result<AdapterProps>;
}
