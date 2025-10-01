#[derive(Debug, Default, cucumber::World)]
pub struct BddWorld {
    pub temp_root: Option<tempfile::TempDir>,
    pub bundle_root: Option<std::path::PathBuf>,
    pub selected_type: Option<proof_bundle::TestType>,
}

impl BddWorld {
    pub fn ensure_root(&mut self) -> &std::path::Path {
        if self.temp_root.is_none() {
            self.temp_root = Some(tempfile::TempDir::new().expect("tempdir"));
        }
        self.temp_root.as_ref().unwrap().path()
    }

    pub fn set_env_for_bundle(&mut self, run_id: &str) {
        let root = self.ensure_root().to_path_buf();
        std::env::set_var("LLORCH_PROOF_DIR", &root);
        std::env::set_var("LLORCH_RUN_ID", run_id);
    }

    pub fn open_bundle(
        &mut self,
        tt: proof_bundle::TestType,
    ) -> anyhow::Result<proof_bundle::ProofBundle> {
        self.selected_type = Some(tt);
        let pb = proof_bundle::ProofBundle::for_type(tt)?;
        self.bundle_root = Some(pb.root().to_path_buf());
        Ok(pb)
    }

    pub fn get_pb(&self) -> anyhow::Result<proof_bundle::ProofBundle> {
        let tt =
            self.selected_type.ok_or_else(|| anyhow::anyhow!("no selected test type in world"))?;
        let pb = proof_bundle::ProofBundle::for_type(tt)?;
        Ok(pb)
    }
}

impl Drop for BddWorld {
    fn drop(&mut self) {
        // Prevent cross-scenario leakage of environment configuration
        std::env::remove_var("LLORCH_PROOF_DIR");
        std::env::remove_var("LLORCH_RUN_ID");
    }
}
