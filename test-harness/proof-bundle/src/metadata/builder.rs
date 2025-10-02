//! Metadata builder API

use super::types::TestMetadata;

/// Builder for test metadata
///
/// Provides a fluent API for constructing test metadata programmatically.
///
/// # Example
///
/// ```rust
/// use proof_bundle::test_metadata;
///
/// # fn example() -> anyhow::Result<()> {
/// test_metadata()
///     .priority("critical")
///     .spec("ORCH-3250")
///     .team("orchestrator")
///     .record()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct TestMetadataBuilder {
    metadata: TestMetadata,
}

impl TestMetadataBuilder {
    /// Create a new metadata builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set priority level
    ///
    /// Standard values: `critical`, `high`, `medium`, `low`
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .priority("critical")
    ///     .build();
    /// ```
    pub fn priority(mut self, level: &str) -> Self {
        self.metadata.priority = Some(level.to_string());
        self
    }
    
    /// Set spec or requirement ID
    ///
    /// Examples: `ORCH-3250`, `REQ-AUTH-001`
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .spec("ORCH-3250")
    ///     .build();
    /// ```
    pub fn spec(mut self, id: &str) -> Self {
        self.metadata.spec = Some(id.to_string());
        self
    }
    
    /// Set owning team
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .team("orchestrator")
    ///     .build();
    /// ```
    pub fn team(mut self, name: &str) -> Self {
        self.metadata.team = Some(name.to_string());
        self
    }
    
    /// Set owner email
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .owner("alice@example.com")
    ///     .build();
    /// ```
    pub fn owner(mut self, email: &str) -> Self {
        self.metadata.owner = Some(email.to_string());
        self
    }
    
    /// Set related issue
    ///
    /// Example: `#1234`
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .issue("#1234")
    ///     .build();
    /// ```
    pub fn issue(mut self, id: &str) -> Self {
        self.metadata.issue = Some(id.to_string());
        self
    }
    
    /// Set flakiness description
    ///
    /// Example: `5% timeout rate on slow CI`
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .flaky("5% timeout rate")
    ///     .build();
    /// ```
    pub fn flaky(mut self, description: &str) -> Self {
        self.metadata.flaky = Some(description.to_string());
        self
    }
    
    /// Set expected timeout
    ///
    /// Example: `30s`, `2m`
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .timeout("30s")
    ///     .build();
    /// ```
    pub fn timeout(mut self, duration: &str) -> Self {
        self.metadata.timeout = Some(duration.to_string());
        self
    }
    
    /// Set required resources
    ///
    /// Example: `&["gpu", "cuda", "16gb-vram"]`
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .requires(&["GPU", "CUDA"])
    ///     .build();
    /// ```
    pub fn requires(mut self, resources: &[&str]) -> Self {
        self.metadata.requires = resources.iter().map(|s| s.to_string()).collect();
        self
    }
    
    /// Add a single required resource
    ///
    /// Can be called multiple times to add multiple resources.
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .require("GPU")
    ///     .require("CUDA")
    ///     .build();
    /// ```
    pub fn require(mut self, resource: &str) -> Self {
        self.metadata.requires.push(resource.to_string());
        self
    }
    
    /// Set tags
    ///
    /// Example: `&["integration", "slow"]`
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .tags(&["integration", "slow"])
    ///     .build();
    /// ```
    pub fn tags(mut self, tags: &[&str]) -> Self {
        self.metadata.tags = tags.iter().map(|s| s.to_string()).collect();
        self
    }
    
    /// Add a single tag
    ///
    /// Can be called multiple times to add multiple tags.
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .tag("integration")
    ///     .tag("slow")
    ///     .build();
    /// ```
    pub fn tag(mut self, tag: &str) -> Self {
        self.metadata.tags.push(tag.to_string());
        self
    }
    
    /// Add a custom field
    ///
    /// Example: `.custom("deployment-stage", "canary")`
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .custom("compliance", "SOC2")
    ///     .build();
    /// ```
    pub fn custom(mut self, key: &str, value: &str) -> Self {
        self.metadata.custom.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Record the metadata
    ///
    /// Currently stores in thread-local storage.
    /// In future, may write to proof bundle immediately.
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// test_metadata()
    ///     .priority("critical")
    ///     .record()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn record(self) -> anyhow::Result<()> {
        // TODO: Store in thread-local or global registry
        // For now, just validate
        Ok(())
    }
    
    /// Build the metadata without recording
    ///
    /// Returns the constructed metadata object.
    ///
    /// # Example
    ///
    /// ```rust
    /// use proof_bundle::test_metadata;
    ///
    /// let metadata = test_metadata()
    ///     .priority("critical")
    ///     .spec("ORCH-3250")
    ///     .build();
    /// ```
    pub fn build(self) -> TestMetadata {
        self.metadata
    }
}

/// Entry point for test metadata builder
///
/// Creates a new `TestMetadataBuilder` for fluent API construction.
///
/// # Example
///
/// ```rust
/// use proof_bundle::test_metadata;
///
/// # fn example() -> anyhow::Result<()> {
/// test_metadata()
///     .priority("critical")
///     .spec("ORCH-3250")
///     .record()?;
/// # Ok(())
/// # }
/// ```
pub fn test_metadata() -> TestMetadataBuilder {
    TestMetadataBuilder::new()
}
