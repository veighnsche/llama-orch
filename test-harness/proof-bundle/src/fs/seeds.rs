use anyhow::{Context, Result};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

/// Append seed entries to `seeds.txt`.
#[derive(Clone, Debug)]
pub struct SeedsRecorder {
    pub(crate) path: PathBuf,
}

impl SeedsRecorder {
    pub fn record<S: std::fmt::Display>(&self, seed: S) -> Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("open seeds file: {}", self.path.display()))?;
        writeln!(f, "seed={}", seed)?;
        Ok(())
    }
}
