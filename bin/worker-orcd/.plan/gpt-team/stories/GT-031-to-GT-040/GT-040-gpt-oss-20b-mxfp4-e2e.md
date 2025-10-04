# GT-040: GPT-OSS-20B MXFP4 E2E

**Team**: GPT-Gamma  
**Sprint**: Sprint 7 (Adapter + E2E)  
**Size**: M (3 days) ← **+1 day for provenance verification**  
**Days**: 93-95  
**Spec Ref**: M0-W-1001  
**Security Review**: auth-min Team 🎭

---

## Story Description

Implement end-to-end test for GPT-OSS-20B using MXFP4 quantization. Validate full model loading, inference, and text generation pipeline works correctly.

**Security Enhancement**: Add model provenance verification for supply chain security. Verify GPT-OSS-20B is downloaded from official OpenAI source, validate file hash against known-good values, and log provenance metadata. Prevents loading of compromised or poisoned models.

---

## Acceptance Criteria

- [ ] GPT-OSS-20B loads with MXFP4 weights
- [ ] Model fits in 24GB VRAM
- [ ] Model generates coherent text
- [ ] Test validates generation quality
- [ ] Test validates reproducibility (temp=0)
- [ ] Performance benchmarks complete
- [ ] Documentation updated
- [ ] Ready for Gate 3

**Provenance Verification Criteria**:
- [ ] Verify model source is official OpenAI repository
- [ ] Validate file hash against known-good SHA256
- [ ] Log model provenance (source, hash, download timestamp)
- [ ] Reject models from untrusted sources
- [ ] Create provenance verification helper function
- [ ] Add provenance metadata to model config

---

## Technical Details

### Provenance Verification Implementation

```rust
#[derive(Debug, Clone)]
pub struct ModelProvenance {
    pub source: String,           // e.g., "https://huggingface.co/openai/gpt-oss-20b"
    pub file_hash: String,        // SHA256 hash
    pub download_timestamp: u64,  // Unix timestamp
    pub verified: bool,           // Hash verification result
}

fn verify_model_provenance(model_path: &Path) -> Result<ModelProvenance> {
    // Calculate file hash
    let file_hash = sha256_file(model_path)?;
    
    // Known-good hashes for trusted models
    let known_good_hashes = HashMap::from([
        ("gpt-oss-20b-mxfp4.gguf", "abc123..."),  // From OpenAI
    ]);
    
    // Verify hash
    let verified = known_good_hashes
        .get(model_path.file_name().unwrap().to_str().unwrap())
        .map(|&expected| expected == file_hash)
        .unwrap_or(false);
    
    if !verified {
        tracing::warn!("Model hash mismatch or unknown model: {}", file_hash);
    }
    
    Ok(ModelProvenance {
        source: "https://huggingface.co/openai/gpt-oss-20b".to_string(),
        file_hash,
        download_timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        verified,
    })
}
```

## Definition of Done

- [ ] E2E test passing
- [ ] Generation quality validated
- [ ] Documentation updated
- [ ] Ready for Gate 3

**Security Definition of Done**:
- [ ] Model provenance verified and logged
- [ ] File hash validated against known-good value
- [ ] Provenance verification integrated into model loading
- [ ] Documentation includes trusted sources list
- [ ] Error handling for untrusted models

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---

**Security Note**: This story implements supply chain security for model provenance. Verifies GPT-OSS-20B is from official OpenAI source and validates file hash to prevent loading of compromised/poisoned models. Critical for M0 as we use trusted sources only (no user uploads).

**Trusted Model Sources**:
- ✅ OpenAI: GPT-OSS-20B (official Hugging Face repo)
- ✅ Qwen: Qwen2.5-0.5B-Instruct (official Qwen repo)
- ✅ Microsoft: Phi-3-Mini (official Microsoft repo)
- ❌ User uploads: NOT ALLOWED in M0

---

Detailed by Project Management Team — ready to implement 📋  
Security verified by auth-min Team 🎭
