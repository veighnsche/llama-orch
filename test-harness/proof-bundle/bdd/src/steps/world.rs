#[derive(Debug, Default, cucumber::World)]
pub struct BddWorld {
    pub temp_root: Option<tempfile::TempDir>,
    pub bundle_root: Option<std::path::PathBuf>,
    pub selected_type: Option<proof_bundle::LegacyTestType>,
    pub test_summary: Option<proof_bundle::TestSummary>,
    // V2 API fields
    pub last_error: Option<String>,
    pub cargo_output: Option<String>,
    pub expected_test_count: usize,
    pub executive_summary: Option<String>,
    pub test_report: Option<String>,
    pub failure_report: Option<String>,
    pub metadata_report: Option<String>,
    // Metadata testing fields
    pub doc_comment: Option<String>,
    pub parsed_metadata: Option<proof_bundle::TestMetadata>,
    pub is_critical: Option<bool>,
    pub is_high_priority: Option<bool>,
    pub is_flaky: Option<bool>,
    pub test_result: Option<proof_bundle::TestResult>,
    pub json_output: Option<String>,
    pub roundtrip_metadata: Option<proof_bundle::TestMetadata>,
    pub timeout_value: Option<String>,
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
        tt: proof_bundle::LegacyTestType,
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
    
    pub fn clear(&mut self) {
        self.last_error = None;
        self.cargo_output = None;
        self.expected_test_count = 0;
        self.executive_summary = None;
        self.test_report = None;
        self.failure_report = None;
        self.metadata_report = None;
        self.doc_comment = None;
        self.parsed_metadata = None;
        self.is_critical = None;
        self.is_high_priority = None;
        self.is_flaky = None;
        self.test_result = None;
        self.json_output = None;
        self.roundtrip_metadata = None;
        self.timeout_value = None;
    }
}

impl Drop for BddWorld {
    fn drop(&mut self) {
        // Prevent cross-scenario leakage of environment configuration
        std::env::remove_var("LLORCH_PROOF_DIR");
        std::env::remove_var("LLORCH_RUN_ID");
    }
}
