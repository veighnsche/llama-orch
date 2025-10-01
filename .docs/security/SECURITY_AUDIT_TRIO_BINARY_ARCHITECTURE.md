# Security Audit: Trio-Binary Architecture — Complete Security Assessment

**Date**: 2025-10-01  
**Auditor**: Security Team  
**Scope**: `.docs/ARCHITECTURE_CHANGE_PLAN.md` — Full security analysis  
**Status**: **20 SECURITY ISSUES IDENTIFIED**

---

## Executive Summary

This audit provides a **complete security assessment** of the worker-orcd architecture plan, including:

**PART A: Security mechanisms from your email** (8 issues) — Authentication, mTLS, CA/RA, job tokens
**PART B: Hidden vulnerabilities in the plan** (12 issues) — Design flaws enabling RCE, data theft, DoS

### All 20 Security Issues

**From Your Email** (correctly identified):
1. Worker-orcd endpoint authentication
2. mTLS for internal communication  
3. Certificate Authority infrastructure
4. Registration Authority delegation
5. Short-lived job tokens
6. Worker enrollment protocol
7. Credential rotation/revocation
8. pool-managerd authentication

**Hidden in Architecture Plan** (not addressed):
9. **Model poisoning via Commit endpoint** (RCE)
10. **VRAM pointer leakage** (memory exploits)
11. **Unsafe CUDA FFI** (arbitrary code execution)
12. **No input validation** (injection attacks)
13. **SSE streaming lacks auth** (token theft)
14. **NCCL plaintext** (model exfiltration)
15. **Digest TOCTOU** (integrity bypass)
16. **Forgeable ModelShardHandle** (seal spoofing)
17. **No resource limits** (DoS)
18. **Unchecked privileges** (escalation)
19. **GGUF parser trusts input** (buffer overflow)
20. **No process isolation** (cross-tenant attacks)

---

## PART A: Security Mechanisms from Your Email (Correctly Identified)

These 8 security concerns from your email are **valid and necessary**. The architecture plan should address them.

### Issue 1: Worker-orcd Endpoint Authentication

**Your Concern**: "How will worker-orcd authenticate callers?"

**Current State**: Architecture plan defines endpoints but doesn't specify authentication:
```
POST /worker/plan
POST /worker/commit
GET  /worker/ready
POST /worker/execute
```

**Why This Matters**: Without auth, anyone on network can:
- Call `/commit` to load arbitrary models
- Call `/execute` to run inference
- Probe `/plan` to learn GPU capacity

**Solution Needed**: Add Bearer token auth or mTLS to all endpoints (except maybe `/ready` for health checks).

**Reference**: `auth-min` crate provides Bearer token primitives (timing-safe comparison, fingerprinting).

---

### Issue 2: mTLS for Internal Communication

**Your Concern**: "All internal communication is mTLS"

**Current State**: Plan shows HTTP communication between services but no mention of TLS.

**Why This Matters**: 
- orchestratord → pool-managerd traffic is plaintext
- pool-managerd → worker-orcd traffic is plaintext
- Bearer tokens sent in cleartext can be captured

**Solution Needed**: 
- Option A: mTLS with certificate-based mutual auth
- Option B: TLS with server certs only
- Option C: Trust internal network (acceptable for M0 if documented)

**Reference**: Would require extending `auth-min` with mTLS support (reqwest + rustls).

---

### Issue 3: Certificate Authority Infrastructure

**Your Concern**: "orchestord and worker-orcd enrolling via our central CA"

**Current State**: No CA mentioned in architecture plan.

**Why This Matters**: If using mTLS, need a way to:
- Issue certificates to services
- Verify certificate signatures
- Manage trust root

**Solution Needed**:
- Option A: External CA (HashiCorp Vault, cert-manager)
- Option B: Embedded CA using `rcgen` crate
- Option C: Pre-shared certificates (manual deployment)

**Recommendation**: Start with Option C for M0 (simplest), evolve to Option B post-M0.

---

### Issue 4: Registration Authority Delegation

**Your Concern**: "Pool managers may act as delegated Registration Authorities"

**Current State**: Not addressed in plan.

**Why This Matters**: If pool-managerd can issue worker certificates:
- Reduces central CA bottleneck
- Enables per-host autonomy
- Requires intermediate certificate model

**Solution Needed**: RA delegation with intermediate certs:
```rust
pub struct RegistrationAuthority {
    intermediate_cert: Certificate,  // Signed by root CA
    intermediate_key: PrivateKey,
    max_ttl: Duration,  // Max worker cert lifetime
}

impl RegistrationAuthority {
    pub fn issue_worker_cert(worker_id: &str, ttl: Duration) -> Certificate;
}
```

**Recommendation**: Defer to post-M0 (adds significant complexity).

---

### Issue 5: Short-Lived Job Tokens

**Your Concern**: "Job execution gated by short-lived, signed job tokens"

**Current State**: Not addressed in plan.

**Why This Matters**: Long-lived Bearer tokens have large blast radius. Job-specific tokens provide:
- Scoped authorization (job_id, worker_id, model_ref)
- Limited time window (5 minutes)
- Easier revocation

**Solution Needed**: HMAC-signed or Ed25519-signed tokens:
```rust
pub struct JobToken {
    job_id: String,
    worker_id: String,
    model_ref: String,
    expires_at: SystemTime,
    signature: Vec<u8>,
}

pub fn issue_job_token(claims: &JobToken, key: &[u8]) -> String;
pub fn verify_job_token(token: &str, key: &[u8]) -> Result<JobToken>;
```

**Recommendation**: Implement for M0 if multi-tenancy is needed, otherwise defer.

---

### Issue 6: Worker Enrollment Protocol

**Your Concern**: How do workers prove identity when registering with pool-managerd?

**Current State**: Plan says pool-managerd "spawns" workers but doesn't detail enrollment.

**Why This Matters**: Need to distinguish legitimate workers from rogue processes.

**Solution Needed**: Enrollment flow:
```
1. pool-managerd generates worker certificate (or gets from CA)
2. Spawns worker-orcd with certificate
3. Worker registers with pool-managerd using certificate
4. pool-managerd verifies certificate signature
5. Worker added to registry
```

**Recommendation**: For M0, use shared secret per pool (simpler than per-worker certs).

---

### Issue 7: Credential Rotation/Revocation

**Your Concern**: "Short-lived worker certs, job tokens, and intermediate revocation"

**Current State**: Not addressed in plan.

**Why This Matters**: Credentials eventually need to be rotated or revoked:
- Worker compromised → revoke certificate
- Token leaked → revoke/blacklist
- Periodic rotation → reduce blast radius

**Solution Needed**:
- Short TTLs (< 1 hour) to limit exposure
- In-memory revocation list for emergency cases
- Graceful rotation with overlapping validity periods

**Recommendation**: For M0, rely on short TTLs and manual restart. Add graceful rotation post-M0.

---

### Issue 8: pool-managerd Authentication

**Your Concern**: pool-managerd needs authentication (per SEC-AUTH-3002)

**Current State**: ⚠️ Current code has NO authentication on any endpoint

**Why This Matters**: Anyone on network can:
- Preload engines
- Query pool status
- (Future) Dispatch tasks

**Solution Needed**: Add auth middleware:
```rust
// bin/pool-managerd/src/api/auth.rs
pub async fn bearer_auth_middleware(
    State(state): State<Arc<PoolState>>,
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let token = auth_min::parse_bearer(auth_header)?;
    if !auth_min::timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(next.run(req).await)
}
```

**Recommendation**: Implement when pool-managerd integration work begins (use orchestratord pattern).

---

## PART B: Hidden Vulnerabilities (Not in Email or Plan)

These 12 vulnerabilities are **design flaws** in the architecture plan that enable serious attacks.

### Vulnerability 9: Model Poisoning via Commit Endpoint

**The Problem**: `POST /worker/commit` accepts arbitrary `model_bytes` with NO validation.

**Attack Scenario**:
1. Attacker compromises pool-managerd or MitMs pool→worker connection
2. Sends malicious model_bytes to Commit endpoint
3. Worker loads bytes into VRAM blindly
4. Malicious payload exploits CUDA driver or causes buffer overflow

**Why Critical**: Remote code execution on GPU worker.

**Required Fix**: Validate BEFORE loading:
```rust
pub async fn commit_model(req: CommitRequest) -> Result<ModelShardHandle> {
    // 1. Verify cryptographic signature
    verify_model_signature(&req.model_bytes, &req.signature)?;
    
    // 2. Compute and compare hash
    let hash = sha256(&req.model_bytes);
    if hash != req.expected_hash {
        return Err(\"Hash mismatch\");
    }
    
    // 3. Validate GGUF format
    validate_gguf_format(&req.model_bytes)?;
    
    // 4. Check size limits
    if req.model_bytes.len() > MAX_MODEL_SIZE {
        return Err(\"Too large\");
    }
    
    // NOW safe to load
    load_to_vram(&req.model_bytes)
}
```

**Severity**: CRITICAL

---

### Vulnerability 10: VRAM Pointer Leakage

**The Problem**: `ModelShardHandle` exposes `pub vram_ptr: *mut c_void` in API responses.

**Attack Scenario**:
1. Attacker calls `/worker/ready`
2. Response contains `vram_ptr: 0x7f8a4c00000`
3. Attacker uses pointer to:
   - Bypass ASLR
   - Target buffer overflows at known addresses
   - Craft ROP chains

**Why Critical**: Defeats memory protection, enables exploitation.

**Required Fix**: Never expose pointers:
```rust
pub struct ModelShardHandle {
    pub shard_id: String,      // Opaque ID
    vram_ptr: *mut c_void,     // PRIVATE
    pub sealed: bool,
}

// API returns only shard_id, not pointer
```

**Severity**: HIGH

---

### Vulnerability 11: Unsafe CUDA FFI

**The Problem**: Plan uses `unsafe` CUDA calls without bounds checking:
```rust
unsafe {
    cuda_gemm(ptr, size);  // No validation
}
```

**Attack Scenario**:
1. Attacker sets `max_tokens: usize::MAX`
2. CUDA allocates unchecked → out-of-bounds write
3. GPU memory corruption → potential code execution

**Required Fix**: Validated wrapper:
```rust
pub struct SafeCudaPtr {
    ptr: *mut c_void,
    size: usize,
}

impl SafeCudaPtr {
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.size {
            return Err(\"Out of bounds\");
        }
        unsafe { cuda_memcpy(self.ptr.add(offset), data.as_ptr(), data.len()) };
        Ok(())
    }
}
```

**Severity**: CRITICAL

---

### Vulnerability 12: No Input Validation

**The Problem**: `/worker/execute` accepts arbitrary prompts and params:
```json
{
  \"prompt\": \"<ANY TEXT>\",
  \"max_tokens\": <ANY NUMBER>
}
```

**Attack Scenarios**:
- Null bytes in prompt → tokenizer exploit
- 10MB prompt → VRAM exhaustion
- `max_tokens: usize::MAX` → infinite loop
- Unicode exploits → bypass logging

**Required Fix**:
```rust
const MAX_PROMPT_LEN: usize = 100_000;
const MAX_TOKENS: usize = 4096;

pub fn validate_request(req: &ExecuteRequest) -> Result<()> {
    if req.prompt.len() > MAX_PROMPT_LEN {
        return Err(\"Prompt too long\");
    }
    if req.prompt.contains('\\0') {
        return Err(\"Null bytes\");
    }
    if req.params.max_tokens > MAX_TOKENS {
        return Err(\"max_tokens too large\");
    }
    Ok(())
}
```

**Severity**: HIGH

---

### Vulnerability 13: SSE Streaming Lacks Authentication

**The Problem**: SSE token streams have no auth mentioned in plan.

**Attack Scenario**:
1. User starts job, gets job_id
2. Attacker guesses job_id
3. Attacker connects to SSE stream
4. Receives all tokens (may contain PII, secrets)

**Required Fix**: Authenticate SSE connections:
```rust
pub async fn execute_stream(
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    Path(job_id): Path<String>,
) -> Sse<Stream> {
    verify_token(&auth.token())?;
    verify_job_ownership(&job_id, &auth.token())?;
    stream_tokens(job_id)
}
```

**Severity**: CRITICAL

---

### Vulnerability 14: NCCL Plaintext Communication

**The Problem**: NCCL traffic is unencrypted TCP/RDMA.

**Attack Scenario**:
1. Tensor-parallel job uses 4 workers
2. Workers exchange activations via NCCL
3. Attacker captures packets
4. Reconstructs model weights from activations

**Why Critical**: Model weights are intellectual property (Llama-70B cost millions to train).

**Required Fix**: Options:
- Option A: NCCL over TLS tunnel (stunnel)
- Option B: Isolate NCCL to dedicated VLAN
- Option C: Trust internal network for M0 (document limitation)

**Severity**: HIGH

---

### Vulnerability 15: Digest Verification TOCTOU

**The Problem**: Digest computed once at load, never re-checked:
```
Time 0: Compute digest → \"abc123\"
Time 1: Seal shard
Time 2: <VRAM modified by attacker>
Time 3: Execute with modified weights
```

**Attack Scenario**: GPU driver exploit modifies VRAM after seal.

**Required Fix**: Re-verify before execution:
```rust
pub fn execute(& mut self, handle_id: &str) -> Result<Stream> {
    let handle = self.handles.get(handle_id)?;
    let current_digest = compute_vram_digest(handle.vram_ptr, handle.vram_bytes)?;
    
    if current_digest != handle.digest {
        return Err(\"VRAM integrity violation\");
    }
    
    self.run_inference(handle)
}
```

**Severity**: MEDIUM

---

### Vulnerability 16: Forgeable ModelShardHandle

**The Problem**: Seal is just a boolean, no cryptographic proof:
```rust
ModelShardHandle {
    sealed: true,  // Anyone can set this
    digest: \"fake\"
}
```

**Attack Scenario**: Compromised pool-managerd forges fake seal.

**Required Fix**: Cryptographic seal:
```rust
pub struct ModelShardHandle {
    pub digest: String,
    pub sealed_at: SystemTime,
    pub signature: Vec<u8>,  // HMAC(digest || sealed_at)
}

impl ModelShardHandle {
    pub fn verify(&self, key: &[u8]) -> Result<()> {
        let msg = format!(\"{}|{:?}\", self.digest, self.sealed_at);
        let expected = hmac_sha256(key, msg.as_bytes());
        if expected != self.signature {
            return Err(\"Invalid seal\");
        }
        Ok(())
    }
}
```

**Severity**: HIGH

---

### Vulnerability 17: No Resource Limits

**The Problem**: Plan is silent on resource limits.

**Attack Scenario**: DoS via resource exhaustion:
```python
for i in range(1000):
    requests.post('/worker/execute', json={
        'prompt': 'A' * 100000,
        'max_tokens': 4096
    })
```

**Required Fix**: Enforce limits:
```rust
pub struct ResourceLimits {
    max_vram_per_job: usize,
    max_execution_time: Duration,
    max_concurrent_jobs: usize,
    max_requests_per_minute: usize,
}
```

**Severity**: HIGH

---

### Vulnerability 18: Unchecked Privileges

**The Problem**: If pool-managerd runs as root, workers inherit root privileges.

**Attack Scenario**:
1. Attacker exploits worker vulnerability
2. Worker running as root
3. Attacker has root on GPU node

**Required Fix**: Drop privileges:
```rust
// pool-managerd spawns with reduced privileges
sudo -u worker-orcd /usr/bin/worker-orcd --gpu=0

// Or use capabilities
caps::clear(None, CapSet::Effective)?;
```

**Severity**: HIGH

---

### Vulnerability 19: GGUF Parser Trusts Input

**The Problem**: Parsing untrusted binary format without validation.

**Attack Scenario**:
```
GGUF header:
  tensor_count: 4294967295  // usize::MAX
  
// Parser allocates based on count → OOM or overflow
```

**Required Fix**: Defensive parsing:
```rust
const MAX_TENSORS: usize = 10_000;
const MAX_FILE_SIZE: usize = 100_000_000_000;

pub fn parse_gguf_safe(bytes: &[u8]) -> Result<Model> {
    if bytes.len() > MAX_FILE_SIZE {
        return Err(\"Too large\");
    }
    
    let tensor_count = read_u64()?;
    if tensor_count > MAX_TENSORS {
        return Err(\"Too many tensors\");
    }
    
    // Bounds-checked parsing
    ...
}
```

**Severity**: CRITICAL

---

### Vulnerability 20: No Process Isolation

**The Problem**: No mention of containers, namespaces, or sandboxing.

**Attack Scenario**:
1. Attacker exploits worker-orcd
2. Worker has access to:
   - Other workers' VRAM (GPU shared memory)
   - Host filesystem (other models)
   - Network (exfiltration)

**Required Fix**: Isolate workers:
- Option A: Run in containers (Docker, Podman)
- Option B: Use Linux namespaces (CLONE_NEWPID, CLONE_NEWNET)
- Option C: SELinux/AppArmor policies

**Severity**: HIGH

---

## Current auth-min Capabilities

### What auth-min Provides ✅

| Capability | Status | Implementation |
|------------|--------|----------------|
| Timing-safe token comparison | ✅ Implemented | `timing_safe_eq()` with compiler fence |
| Token fingerprinting (fp6) | ✅ Implemented | SHA-256 → first 6 hex chars |
| Bearer token parsing | ✅ Implemented | RFC 6750 compliant |
| Bind policy enforcement | ✅ Implemented | Loopback detection + startup validation |

### What Needs to Be Added

For your email's requirements:
- mTLS support (certificates, rustls integration)
- Token rotation helpers (graceful overlap)
- Job token issuance/verification (HMAC or Ed25519)
- Revocation lists (in-memory blacklist)
- Startup refusal for non-loopback without token

---

## Summary: All 20 Security Issues

### Severity Breakdown

**CRITICAL (6 issues)**:
- #9: Model poisoning via Commit endpoint
- #11: Unsafe CUDA FFI
- #13: SSE streaming lacks auth
- #19: GGUF parser trusts input
- Plus: #1 Worker-orcd endpoint auth, #8 pool-managerd auth

**HIGH (11 issues)**:
- #2: mTLS for internal communication
- #3: Certificate Authority infrastructure
- #5: Short-lived job tokens
- #6: Worker enrollment protocol
- #10: VRAM pointer leakage
- #12: No input validation
- #14: NCCL plaintext
- #16: Forgeable ModelShardHandle
- #17: No resource limits
- #18: Unchecked privileges
- #20: No process isolation

**MEDIUM (3 issues)**:
- #4: Registration Authority delegation
- #7: Credential rotation/revocation
- #15: Digest TOCTOU

---

## Recommendations by Phase

### M0 Pilot (Must Fix)

**Authentication & Authorization**:
1. ✅ Add Bearer token auth to worker-orcd endpoints (Issue #1)
2. ✅ Add Bearer token auth to pool-managerd endpoints (Issue #8)
3. ✅ Authenticate SSE streams (Issue #13)

**Input Validation**:
4. ✅ Validate model_bytes before loading (Issue #9)
5. ✅ Validate prompts and params (Issue #12)
6. ✅ Validate GGUF format defensively (Issue #19)

**Memory Safety**:
7. ✅ Make VRAM pointers private (Issue #10)
8. ✅ Add bounds checking to CUDA FFI (Issue #11)

**Resource Protection**:
9. ✅ Enforce resource limits (Issue #17)
10. ✅ Drop worker privileges (Issue #18)

### Post-M0 Hardening

**Infrastructure Security**:
11. Consider mTLS for internal communication (Issue #2)
12. Consider CA infrastructure if using mTLS (Issue #3)
13. Consider job tokens for multi-tenancy (Issue #5)

**Advanced Features**:
14. Implement worker enrollment protocol (Issue #6)
15. Add graceful credential rotation (Issue #7)
16. Consider RA delegation for scale (Issue #4)

**Defense in Depth**:
17. Isolate NCCL traffic (Issue #14)
18. Re-verify digests periodically (Issue #15)
19. Add cryptographic seal signatures (Issue #16)
20. Containerize worker processes (Issue #20)

---

## Action Items for Development Team

### Before Implementation Starts

- [ ] Review all 20 security issues with team
- [ ] Decide on M0 security baseline (recommend items 1-10 above)
- [ ] Update architecture plan with security sections
- [ ] Add security requirements to task breakdown

### During worker-orcd Implementation

**Task Group 1: Authentication** (2 days)
- [ ] Add Bearer token middleware to all endpoints
- [ ] Implement token verification using `auth-min`
- [ ] Add identity breadcrumbs to logs (fp6)
- [ ] Write auth tests (valid/invalid tokens)

**Task Group 2: Input Validation** (1-2 days)
- [ ] Add prompt/param validation
- [ ] Add model signature verification to Commit
- [ ] Implement defensive GGUF parser
- [ ] Add fuzz tests for parser

**Task Group 3: Memory Safety** (1 day)
- [ ] Make `vram_ptr` private in ModelShardHandle
- [ ] Create SafeCudaPtr wrapper
- [ ] Add bounds checking to all FFI calls
- [ ] Add CUDA error handling tests

**Task Group 4: Resource Protection** (1 day)
- [ ] Define ResourceLimits struct
- [ ] Implement execution timeouts
- [ ] Add rate limiting
- [ ] Configure worker to run as non-root user

### Security Testing

- [ ] Add security test suite to BDD
- [ ] Test auth bypass attempts
- [ ] Test input validation edge cases
- [ ] Test resource exhaustion scenarios
- [ ] Perform code review focusing on unsafe blocks

---

## Conclusion

This audit identified **20 security issues** in the worker-orcd architecture:

**8 from your email** (correctly identified): Authentication, mTLS, CA/RA, job tokens, rotation
**12 hidden vulnerabilities** (found in plan): Model poisoning, pointer leakage, unsafe FFI, no validation, etc.

**Critical finding**: The architecture plan focuses on VRAM residency correctness but lacks security hardening. Issues #9-#20 are design flaws that enable RCE, data theft, and DoS attacks.

**Recommendation**: 
- Implement items 1-10 (authentication + validation + memory safety) for M0
- Items 11-20 can be deferred to post-M0 based on deployment environment
- Update architecture plan to include security sections before implementation

**This is now a COMPLETE security audit** covering both your concerns and hidden vulnerabilities.

**Current State**:
- No RA concept in codebase
- No intermediate certificate handling
- No delegation protocol
- This is an advanced PKI feature

**Design Approach for the Team**:
```rust
// REQUIRED: RA delegation model
pub struct RegistrationAuthority {
    intermediate_cert: Certificate,  // Signed by root CA
    intermediate_key: PrivateKey,    // For signing worker certs
    max_ttl: Duration,               // Max worker cert lifetime
}

impl RegistrationAuthority {
    pub fn issue_worker_cert(
        &self,
        worker_id: &str,
        gpu_device: u32,
        ttl: Duration,
    ) -> Result<Certificate>;
}
```

**Questions to Answer**:
- Is RA delegation needed for M0, or can pool-managerd use a shared secret?
- What's the trust model: central CA only, or delegated per-host?
- How do we handle intermediate cert rotation?

**Suggested Starting Point**: Defer RA delegation to post-M0; use central CA for M0 pilot

---

#### Decision 4: Job Token Security

**Architecture Goal**: "Job execution gated by short-lived, signed job tokens"

**Current State**:
- No job token concept in codebase
- `auth-min` provides Bearer tokens (long-lived, shared)
- No token signing/verification beyond timing-safe comparison
- No TTL enforcement or scope claims

**Design Approach for the Team**:
```rust
// REQUIRED: Job token with claims
pub struct JobToken {
    job_id: String,
    worker_id: String,
    model_ref: String,
    issued_at: SystemTime,
    expires_at: SystemTime,
    signature: Vec<u8>,  // HMAC-SHA256 or Ed25519
}

pub fn issue_job_token(
    job_id: &str,
    worker_id: &str,
    ttl: Duration,
    signing_key: &[u8],
) -> Result<String>;

pub fn verify_job_token(
    token: &str,
    expected_job_id: &str,
    signing_key: &[u8],
) -> Result<JobToken>;
```

**Questions to Answer**:
- Do we need job tokens for M0, or can we use Bearer tokens per worker?
- What's the token format: JWT (standard) vs custom binary?
- What signing algorithm: HMAC-SHA256 (simpler) vs Ed25519 (more secure)?
- What's the TTL: 5 minutes, 1 hour, or configurable?

**Suggested Starting Point**: HMAC-SHA256 signed tokens with 5-minute TTL for M0

---

#### Decision 5: Credential Lifecycle (Revocation & Rotation)

**Architecture Goal**: "Short-lived worker certs, job tokens, and intermediate revocation handled per host"

**Current State**:
- No revocation mechanism
- No CRL or OCSP support
- No token blacklist
- Rotation requires manual restart (acceptable for M0)

**Design Approach for the Team**:
1. **Short-lived credentials** (TTL < 1 hour) to limit blast radius
2. **In-memory revocation list** for emergency cases (optional for M0)
3. **Graceful rotation** via overlapping validity periods (post-M0)

```rust
pub struct RevocationList {
    revoked_certs: HashSet<String>,  // Certificate serial numbers
    revoked_tokens: HashSet<String>, // Token fingerprints
}

pub fn check_revocation(cert_serial: &str, revoked: &RevocationList) -> bool;
```

---

**Questions to Answer**:
- What's the minimum viable revocation for M0?
- Can we rely on short TTLs instead of explicit revocation?
- When do we need graceful rotation (post-M0 hardening)?

---

## 3. Worker-orcd Security Considerations

### 3.1 RPC Endpoint Authentication

**Architecture Plan Endpoints**:
```
POST /worker/plan      - Feasibility check
POST /worker/commit    - Load model into VRAM
GET  /worker/ready     - Attest worker status
POST /worker/execute   - Run inference
```

**Design Questions for the Team**:

1. **Authentication strategy**
   - Should all endpoints require authentication?
   - Bearer tokens (like orchestratord) or mTLS?
   - Can `/ready` be unauthenticated for health checks?

2. **Authorization model**
   - Who can commit models? (pool-managerd only?)
   - Who can execute jobs? (orchestratord only? clients?)
   - How do we prevent cross-tenant access?

3. **Rate limiting**
   - Do we need rate limiting for M0?
   - Which endpoints are most vulnerable to abuse?
   - Can we rely on admission queue policy?

**Example Implementation Pattern**:
```rust
// REQUIRED: Endpoint authentication
#[axum::debug_handler]
async fn commit_model(
    State(state): State<WorkerState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,  // ← ADD
    Json(req): Json<CommitRequest>,
) -> Result<Json<CommitResponse>, WorkerError> {
    // Verify token is from pool-managerd
    verify_pool_manager_token(&auth.token())?;
    
    // Verify model is authorized for this worker
    verify_model_authorization(&req.model_ref, &state.worker_id)?;
    
    // Proceed with commit...
}
```

### 3.2 ModelShardHandle Security Design

**Architecture Plan**:
```rust
pub struct ModelShardHandle {
    pub vram_ptr: *mut c_void,  // ← EXPOSED POINTER
    pub sealed: bool,           // ← ATTESTATION CLAIM
    pub digest: String,         // ← INTEGRITY CHECK
}
```

**Design Considerations for the Team**:

1. **Pointer exposure**
   - Should VRAM pointers be in API responses?
   - Risk: Pointer leakage in logs/errors
   - Consider: Make `vram_ptr` private, expose only opaque handle ID

2. **Attestation trust**
   - How do we verify the `sealed` claim?
   - Should we add cryptographic proof (signature)?
   - Who is trusted to attest seal status?

3. **Digest verification**
   - When is digest computed? (load time only?)
   - Should we re-verify before each execution?
   - How do we handle digest mismatches?

**Suggested Enhancement**:
```rust
pub struct ModelShardHandle {
    pub shard_id: String,
    pub gpu_device: u32,
    // REMOVE: pub vram_ptr: *mut c_void,  // ← NEVER EXPOSE
    vram_ptr: *mut c_void,  // ← PRIVATE
    pub vram_bytes: usize,
    pub sealed: bool,
    pub digest: String,
    pub sealed_at: SystemTime,  // ← ADD: Seal timestamp
    pub signature: Vec<u8>,     // ← ADD: Cryptographic seal proof
}

impl ModelShardHandle {
    pub fn verify_seal(&self, signing_key: &[u8]) -> Result<()> {
        // Verify signature covers (shard_id, digest, sealed_at)
        // Prevents seal forgery
    }
}
```

### 3.3 NCCL Security (Tensor-Parallel)

**Architecture Plan**: "Workers join NCCL communicator group"

**Design Considerations for the Team**:

1. **NCCL group membership**
   - How do we authenticate workers before they join?
   - NCCL protocol has no built-in auth—need external mechanism
   - Consider: Pre-authenticate via mTLS or signed group tokens

2. **Cross-GPU communication**
   - NCCL uses RDMA/InfiniBand/TCP (no encryption by default)
   - Is internal network trusted, or do we need encryption?
   - Trade-off: Security vs performance overhead

3. **Coordinator trust**
   - Who coordinates NCCL setup? (pool-managerd likely)
   - How do we prevent coordinator spoofing?
   - Consider: Signed NCCL group membership tokens

**Suggested Approach for M0**:
1. **Pre-authenticate workers** before NCCL group creation (via existing auth)
2. **Trust internal network** for M0 (document as limitation)
3. **Post-M0**: Add mTLS for NCCL TCP backend or isolate to dedicated VLAN

---

## 4. Comparison with Current Implementation

### 4.1 orchestratord Current State

**Authentication**: ✅ Implemented (Bearer token)
```rust
// bin/orchestratord/src/app/auth_min.rs
pub async fn bearer_auth_middleware(
    State(state): State<Arc<AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let token = auth_min::parse_bearer(auth_header)?;
    if !auth_min::timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    // ✅ Correct usage
}
```

**Areas for Enhancement**:
- Consider adding mTLS for internal communication (design decision)
- Consider adding job token issuance (if needed for worker-orcd)
- Consider adding worker certificate verification (if using mTLS)

### 4.2 pool-managerd Current State

**Authentication**: ⚠️ **TO BE IMPLEMENTED**

**Current Status** (per SEC-AUTH-3002):
```
⚠️ No authentication on any endpoint yet
⚠️ No Bearer token validation middleware
⚠️ No LLORCH_API_TOKEN configuration
```

**When Implementing pool-managerd Integration**:
- Add auth middleware (pattern exists in orchestratord)
- Wire `LLORCH_API_TOKEN` configuration
- Apply middleware to all endpoints except `/health`
- Add integration tests with valid/invalid tokens

**Reference Implementation**: `bin/orchestratord/src/app/auth_min.rs`

### 4.3 worker-orcd Proposed State

**Authentication**: 📝 **TO BE DESIGNED**

**Questions for the Team**:
- How should worker-orcd authenticate callers?
- How should worker-orcd prove identity to pool-managerd?
- Should worker-orcd validate job tokens (if we implement them)?
- Bearer tokens (simple) or mTLS (more secure)?

---

## 5. Threat Scenarios to Consider

These scenarios help inform security design decisions. They're not immediate vulnerabilities—they're things to think about when designing the authentication model.

### 5.1 Rogue Worker Registration

**Scenario**: Attacker deploys fake worker-orcd process

**Why This Matters**:
- Without per-worker identity, any process can claim to be a worker
- Shared Bearer tokens don't distinguish between legitimate and rogue workers
- Attacker could receive job assignments and exfiltrate data

**Design Options to Prevent This**:
- Option A: Per-worker certificates (strongest, most complex)
- Option B: Per-pool shared secrets (simpler, less granular)
- Option C: Worker registration requires manual approval (simplest, operational overhead)

**Suggested for M0**: Option B (per-pool secrets), evolve to Option A post-M0

### 5.2 Job Token Replay

**Scenario**: Attacker captures job token, replays to different worker

**Why This Matters**:
- Without job-specific tokens, any captured credential works everywhere
- Long-lived tokens increase window of opportunity
- No way to scope access to specific jobs/workers

**Design Options to Prevent This**:
- Option A: Short-lived job tokens (TTL < 5 min) with (job_id, worker_id) binding
- Option B: One-time use tokens (more complex, requires state)
- Option C: No job tokens for M0, rely on service-level auth

**Suggested for M0**: Option C (defer job tokens), implement Option A post-M0

### 5.3 Model Shard Tampering

**Scenario**: Attacker modifies model weights in VRAM

**Why This Matters**:
- Architecture plan includes SHA-256 digest (good!)
- Question: When is digest verified? Load time only, or before each execution?
- Tampering could inject backdoors or bias outputs

**Design Options to Prevent This**:
- Option A: Re-verify digest before each execution (safest, performance cost)
- Option B: Verify digest periodically (balanced)
- Option C: Verify once at load, trust VRAM isolation (simplest)

**Suggested for M0**: Option C (verify at load), add Option B if needed

### 5.4 NCCL Group Poisoning (Post-M0)

**Scenario**: Attacker joins NCCL group, corrupts multi-GPU inference

**Why This Matters**:
- NCCL has no built-in authentication
- Relevant for tensor-parallel (post-M0)
- Need external mechanism to verify group membership

**Design Options to Prevent This**:
- Pre-authenticate workers before NCCL group creation
- Use signed group membership tokens
- Isolate NCCL traffic to dedicated network

**Suggested for M0**: Not applicable (single-GPU only), design for post-M0

### 5.5 Internal MitM

**Scenario**: Attacker on internal network intercepts service traffic

**Why This Matters**:
- Current Bearer tokens sent in plaintext HTTP
- Captured tokens can be replayed
- Depends on trust model: Is internal network trusted?

**Design Options to Prevent This**:
- Option A: mTLS for all internal communication (strongest)
- Option B: TLS with server certs only (simpler)
- Option C: Trust internal network for M0 (simplest, document as limitation)

**Suggested for M0**: Option C (trust internal network), add mTLS post-M0 if needed

---

## 6. Planning Recommendations

### 6.1 Security Design Tasks for Architecture Phase

#### Task 1: Document Security Architecture

Add a "Security Architecture" section to the architecture change plan covering:

1. **Trust model diagram**
   - Who authenticates whom?
   - What credentials are used where?
   - How are credentials issued/rotated/revoked?

2. **Choose mTLS strategy** (if needed)
   - Option A: External CA (Vault, cert-manager)
   - Option B: Embedded CA (rcgen)
   - Option C: Pre-shared certs (manual)
   - Option D: Defer mTLS to post-M0, use Bearer tokens

3. **Design job token format** (if needed)
   - Claims: job_id, worker_id, model_ref, exp
   - Signature: HMAC-SHA256 or Ed25519
   - Encoding: JWT or custom binary
   - Or: Defer to post-M0

4. **Define credential lifecycle**
   - Issuance: How are credentials created?
   - Rotation: Manual restart or graceful overlap?
   - Revocation: Short TTL or explicit blacklist?

#### Task 2: Implement pool-managerd Authentication

When the team works on pool-managerd integration:

```rust
// REQUIRED: Add to bin/pool-managerd/src/api/auth.rs
pub async fn bearer_auth_middleware(
    State(state): State<Arc<PoolState>>,
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth = req.headers().get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    
    let token = auth_min::parse_bearer(auth)
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    let expected = state.config.api_token
        .as_ref()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    
    if !auth_min::timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
        let fp6 = auth_min::token_fp6(&token);
        tracing::warn!(
            identity = %format!("token:{}", fp6),
            event = "auth_failed"
        );
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    let fp6 = auth_min::token_fp6(&token);
    req.extensions_mut().insert(Identity {
        breadcrumb: format!("token:{}", fp6),
    });
    
    Ok(next.run(req).await)
}
```

**Reference**: `bin/orchestratord/src/app/auth_min.rs` (working example)

---

### 6.2 Potential auth-min Extensions

These are features the team might want to add to `auth-min` depending on design decisions:

#### Extension 1: mTLS Support (If Chosen)

```rust
// NEW: libs/auth-min/src/mtls.rs
pub struct MtlsConfig {
    pub ca_cert_path: PathBuf,
    pub client_cert_path: PathBuf,
    pub client_key_path: PathBuf,
    pub verify_peer: bool,
}

pub fn build_mtls_client(config: &MtlsConfig) -> Result<reqwest::Client> {
    let ca_cert = std::fs::read(&config.ca_cert_path)?;
    let client_cert = std::fs::read(&config.client_cert_path)?;
    let client_key = std::fs::read(&config.client_key_path)?;
    
    let identity = reqwest::Identity::from_pem(&[&client_cert[..], &client_key[..]].concat())?;
    let ca = reqwest::Certificate::from_pem(&ca_cert)?;
    
    reqwest::Client::builder()
        .identity(identity)
        .add_root_certificate(ca)
        .build()
}

pub fn build_mtls_server_config(config: &MtlsConfig) -> Result<axum_server::tls_rustls::RustlsConfig> {
    // Implementation for Axum server with mTLS
}
```

**When to implement**: If team chooses mTLS for internal communication

---

#### Extension 2: Token Rotation Helpers (If Needed)

```rust
// NEW: libs/auth-min/src/rotation.rs
pub struct TokenRotation {
    current_token: String,
    next_token: Option<String>,
    rotation_deadline: SystemTime,
}

impl TokenRotation {
    pub fn verify(&self, token: &str) -> bool {
        // Accept both current and next token during overlap period
        auth_min::timing_safe_eq(token.as_bytes(), self.current_token.as_bytes())
            || self.next_token.as_ref()
                .map(|next| auth_min::timing_safe_eq(token.as_bytes(), next.as_bytes()))
                .unwrap_or(false)
    }
    
    pub fn rotate(&mut self, new_token: String, overlap: Duration) {
        self.next_token = Some(new_token);
        self.rotation_deadline = SystemTime::now() + overlap;
    }
    
    pub fn finalize_rotation(&mut self) {
        if let Some(next) = self.next_token.take() {
            self.current_token = next;
        }
    }
}
```

**When to implement**: If team wants graceful token rotation without downtime

---

#### Extension 3: Job Token Support (If Needed)

```rust
// NEW: libs/auth-min/src/job_token.rs
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

pub struct JobToken {
    pub job_id: String,
    pub worker_id: String,
    pub model_ref: String,
    pub issued_at: SystemTime,
    pub expires_at: SystemTime,
}

pub fn issue_job_token(
    claims: &JobToken,
    signing_key: &[u8],
) -> Result<String> {
    let payload = serde_json::to_vec(claims)?;
    let mut mac = HmacSha256::new_from_slice(signing_key)?;
    mac.update(&payload);
    let signature = mac.finalize().into_bytes();
    
    let token = base64::encode(&[&payload[..], &signature[..]].concat());
    Ok(token)
}

pub fn verify_job_token(
    token: &str,
    signing_key: &[u8],
) -> Result<JobToken> {
    let decoded = base64::decode(token)?;
    let (payload, signature) = decoded.split_at(decoded.len() - 32);
    
    let mut mac = HmacSha256::new_from_slice(signing_key)?;
    mac.update(payload);
    mac.verify_slice(signature)?;
    
    let claims: JobToken = serde_json::from_slice(payload)?;
    
    // Verify expiry
    if SystemTime::now() > claims.expires_at {
        return Err(AuthError::TokenExpired);
    }
    
    Ok(claims)
}
```

**When to implement**: If team chooses short-lived job tokens for authorization

---

### 6.3 Suggested Architecture Plan Updates

**Consider adding**: "§X. Security Architecture" section

Could include:

1. **Trust Model Diagram**
   ```
   Root CA
     ├─ orchestratord cert (CN=orchestratord)
     ├─ pool-managerd intermediate cert (CN=pool-manager-*)
     │   └─ worker-orcd certs (CN=worker-*, TTL=1h)
     └─ (future) client certs
   ```

2. **Authentication Matrix**
   | Caller | Callee | Mechanism | Token Type |
   |--------|--------|-----------|------------|
   | Client → orchestratord | Bearer | API token |
   | orchestratord → pool-managerd | mTLS | Service cert |
   | pool-managerd → worker-orcd | mTLS | Worker cert |
   | orchestratord → worker-orcd | Job token | Signed JWT |

3. **Credential Lifecycle**
   - Root CA: Manual rotation (yearly)
   - Service certs: 90-day TTL, auto-renewal
   - Worker certs: 1-hour TTL, issued by pool-managerd RA
   - Job tokens: 5-minute TTL, issued per job
   - API tokens: Manual rotation (monthly)

4. **Revocation Strategy**
   - Worker certs: Short TTL (1h) + in-memory blacklist
   - Job tokens: Short TTL (5m) + no revocation needed
   - Service certs: CRL checked at startup
   - API tokens: Restart required

5. **Threat Model**
   - In-scope: Rogue workers, token replay, MitM, model tampering
   - Out-of-scope: Physical access, side-channel, supply chain

**Note**: This is optional—the team can decide how much security detail to include in the architecture plan.

---

### 6.4 Testing Considerations

**When implementing worker-orcd**, consider adding security tests:

#### Suggested Security Test Scenarios

```gherkin
@security @worker-orcd
Scenario: Rogue worker rejected at enrollment
  Given a worker-orcd without valid certificate
  When it attempts to register with pool-managerd
  Then registration is rejected with TLS handshake failure
  And no worker entry is created in registry

@security @worker-orcd
Scenario: Job token with wrong worker_id rejected
  Given a valid job token for worker-A
  When worker-B attempts to execute with that token
  Then execution is rejected with 403 Forbidden
  And security event is logged

@security @worker-orcd
Scenario: Expired job token rejected
  Given a job token issued 10 minutes ago (TTL=5m)
  When worker attempts to execute
  Then execution is rejected with 401 Unauthorized
  And token expiry is logged

@security @worker-orcd
Scenario: Model digest mismatch detected
  Given a model committed with digest=abc123
  When VRAM contents are modified (digest=xyz789)
  And worker attempts to execute
  Then execution is rejected with 500 Internal Error
  And security alert is raised
```

---

## 7. Audit Checklist

### 7.1 Current auth-min Crate

- ✅ Timing-safe comparison implemented correctly
- ✅ Token fingerprinting is non-reversible (SHA-256)
- ✅ Bearer token parsing is RFC 6750 compliant
- ✅ Bind policy enforcement works
- ✅ No token leakage in logs (fp6 only)
- ✅ Test coverage for timing attacks
- ❌ No mTLS support
- ❌ No token rotation helpers
- ❌ No job token support
- ❌ No revocation mechanism

### 7.2 Architecture Change Plan

- ✅ VRAM residency enforcement is well-designed
- ✅ ModelShardHandle provides integrity checks (digest)
- ✅ Plan/Commit/Ready/Execute protocol is clear
- ❌ **No authentication design for worker-orcd**
- ❌ **No mTLS infrastructure specified**
- ❌ **No CA/RA delegation model**
- ❌ **No job token security**
- ❌ **No revocation/rotation strategy**
- ❌ **No threat model documented**
- ❌ **No security testing plan**

### 7.3 Implementation Readiness

- ❌ pool-managerd has no authentication (CRITICAL)
- ❌ orchestratord → pool-managerd uses plaintext HTTP
- ❌ No mTLS client/server implementations
- ❌ No certificate generation tooling
- ❌ No job token issuance/verification
- ❌ No security test suite

---

## 8. Summary and Next Steps

### 8.1 Audit Summary

**Architecture Plan Status**: ✅ **Solid foundation for VRAM/inference control**

The VRAM residency enforcement, ModelShardHandle design, and Plan/Commit/Ready/Execute protocol are well thought out. The security aspects need design decisions before implementation.

### 8.2 Security Design Decisions Needed

Before or during worker-orcd implementation, the team should decide:

1. **Authentication strategy for worker-orcd**
   - Bearer tokens (simpler) or mTLS (more secure)?
   - Shared secrets or per-worker identity?

2. **Internal service communication**
   - Trust internal network for M0, or implement mTLS?
   - What's the operational complexity budget?

3. **Job authorization model**
   - Service-level auth sufficient, or need job tokens?
   - If job tokens: JWT or custom format?

4. **pool-managerd authentication**
   - Add auth middleware when implementing integration
   - Pattern exists in orchestratord—straightforward to replicate

5. **Credential lifecycle**
   - Manual rotation (restart) or graceful overlap?
   - Short TTLs or explicit revocation?

### 8.3 Suggested Approach for M0 Pilot

**Simplified Security Model** (gets you started quickly):

1. **Bearer tokens for all services**
   - orchestratord → pool-managerd: `LLORCH_API_TOKEN`
   - pool-managerd → worker-orcd: `POOL_SECRET` (shared per pool)
   - Leverage existing `auth-min` primitives

2. **No job tokens for M0**
   - Rely on service-level authentication
   - Add job tokens post-M0 if needed for multi-tenancy

3. **No mTLS for M0**
   - Trust internal network (document as limitation)
   - Add mTLS post-M0 if deploying on untrusted networks

4. **Manual credential rotation**
   - Restart services with new tokens
   - Add graceful rotation post-M0 if needed

**Benefits**:
- ✅ Minimal implementation complexity
- ✅ No certificate management
- ✅ Sufficient for home-lab/single-tenant
- ✅ Can evolve to mTLS/job tokens later

**Limitations** (document these):
- ⚠️ No defense against internal MitM
- ⚠️ No per-worker identity
- ⚠️ Shared secrets per pool
- ⚠️ Manual rotation requires restart

### 8.4 Post-M0 Evolution Path

**When to add more security**:

1. **mTLS** — When deploying on untrusted internal networks
2. **Job tokens** — When adding multi-tenancy or per-job isolation
3. **Per-worker certs** — When need to distinguish individual workers
4. **Graceful rotation** — When uptime requirements increase
5. **Revocation** — When need emergency credential invalidation

---

## 9. Conclusion

### Key Takeaways

**For the Development Team**:

1. **auth-min provides solid Bearer token primitives** — Use these for M0
2. **Security design decisions are needed** — Not blockers, just planning tasks
3. **Simplified approach works for M0** — mTLS/job tokens can come later
4. **pool-managerd needs auth** — Add when implementing integration (pattern exists)
5. **Threat scenarios inform design** — Consider them when making decisions

### What This Audit Provides

- ✅ Inventory of current `auth-min` capabilities
- ✅ Design options for authentication strategies
- ✅ Threat scenarios to consider
- ✅ Suggested M0 approach (simplified)
- ✅ Post-M0 evolution path
- ✅ Code examples and patterns

### What the Team Should Do

**During Architecture Phase**:
1. Decide on authentication strategy (Bearer tokens recommended for M0)
2. Consider adding "Security Architecture" section to plan (optional)
3. Document known limitations (e.g., trust internal network)

**During Implementation**:
1. Add auth middleware to pool-managerd (use orchestratord pattern)
2. Add auth middleware to worker-orcd (same pattern)
3. Consider adding security tests (examples provided)

**Post-M0**:
1. Evaluate need for mTLS based on deployment environment
2. Evaluate need for job tokens based on multi-tenancy requirements
3. Extend `auth-min` as needed

---

**Audit completed**: 2025-10-01  
**Purpose**: Planning guidance for worker-orcd security design  
**Status**: Architecture plan is solid; security decisions can be made during implementation

---

## Appendix A: auth-min API Reference

### Current API (v0.0.0)

```rust
// Timing-safe comparison
pub fn timing_safe_eq(a: &[u8], b: &[u8]) -> bool;

// Token fingerprinting
pub fn token_fp6(token: &str) -> String;

// Bearer token parsing
pub fn parse_bearer(header_val: Option<&str>) -> Option<String>;

// Bind policy
pub fn is_loopback_addr(addr: &str) -> bool;
pub fn enforce_startup_bind_policy(bind_addr: &str) -> Result<()>;

// Proxy trust
pub fn trust_proxy_auth() -> bool;
```

### Required Extensions (v0.3.0)

```rust
// mTLS support
pub mod mtls {
    pub struct MtlsConfig { /* ... */ }
    pub fn build_mtls_client(config: &MtlsConfig) -> Result<reqwest::Client>;
    pub fn build_mtls_server_config(config: &MtlsConfig) -> Result<RustlsConfig>;
}

// Token rotation
pub mod rotation {
    pub struct TokenRotation { /* ... */ }
    impl TokenRotation {
        pub fn verify(&self, token: &str) -> bool;
        pub fn rotate(&mut self, new_token: String, overlap: Duration);
        pub fn finalize_rotation(&mut self);
    }
}

// Job tokens
pub mod job_token {
    pub struct JobToken { /* ... */ }
    pub fn issue_job_token(claims: &JobToken, key: &[u8]) -> Result<String>;
    pub fn verify_job_token(token: &str, key: &[u8]) -> Result<JobToken>;
}

// Revocation
pub mod revocation {
    pub struct RevocationList { /* ... */ }
    pub fn check_revocation(cert_serial: &str, list: &RevocationList) -> bool;
}
```

---

## Appendix B: Security Event Log Schema

```json
{
  "timestamp": "2025-10-01T13:44:04Z",
  "level": "WARN",
  "event": "auth_failed",
  "identity": "token:a3f2c1",
  "remote_addr": "192.168.1.100:54321",
  "endpoint": "/worker/execute",
  "reason": "invalid_token",
  "job_id": "job-12345",
  "worker_id": "worker-gpu0"
}
```

Required security events:
- `auth_success` (INFO)
- `auth_failed` (WARN)
- `token_expired` (WARN)
- `cert_invalid` (ERROR)
- `digest_mismatch` (CRITICAL)
- `rogue_worker_detected` (CRITICAL)
- `token_rotation` (INFO)
