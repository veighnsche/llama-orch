# Feature File Design
# Created by: TEAM-077
# Date: 2025-10-11

## Naming Convention
- Format: `{number}-{feature-name}.feature`
- Numbers: 01, 02, 03... (for ordering)
- Names: kebab-case, descriptive

## Proposed Feature Files

### 01-ssh-registry-management.feature
**Purpose:** SSH connection setup and node registry management  
**Scenarios:** 10  
**Estimated Lines:** ~350

**Scenarios:**
1. Add remote rbee-hive node to registry (@setup @critical)
2. EH-001a - SSH connection timeout (@setup @error-handling)
3. EH-001b - SSH authentication failure (@setup @error-handling)
4. EH-001c - SSH command execution failure (@setup @error-handling)
5. Install rbee-hive on remote node (@setup)
6. List registered rbee-hive nodes (@setup)
7. Remove node from rbee-hive registry (@setup)
8. EH-011a - Invalid SSH key path (@setup @error-handling)
9. EH-011b - Duplicate node name (@setup @error-handling)
10. Inference fails when node not in registry (@setup @critical)

### 02-model-provisioning.feature
**Purpose:** Model download, catalog, and GGUF support  
**Scenarios:** 12  
**Estimated Lines:** ~450

**Scenarios:**
1. Model found in SQLite catalog
2. Model not found - download with progress
3. EH-007a - Model not found on Hugging Face (@error-handling)
4. EH-007b - Model repository is private (@error-handling)
5. EH-008a - Model download timeout (@error-handling)
6. EH-008b - Model download fails with retry (@error-handling)
7. EH-008c - Downloaded model checksum mismatch (@error-handling)
8. Model catalog registration after download
9. GGUF model detection by file extension (@gguf)
10. GGUF metadata extraction (@gguf)
11. GGUF quantization formats supported (@gguf)
12. GGUF model size calculation (@gguf)

### 03-worker-preflight-checks.feature
**Purpose:** Resource validation before worker startup  
**Scenarios:** 9  
**Estimated Lines:** ~300

**Scenarios:**
1. Worker preflight RAM check passes
2. EH-004a - Worker preflight RAM check fails (@error-handling)
3. EH-004b - RAM exhausted during model loading (@error-handling)
4. Worker preflight backend check passes
5. EH-005a - VRAM exhausted on CUDA device (@error-handling)
6. EH-009a - Backend not available (@error-handling)
7. EH-009b - CUDA not installed (@error-handling)
8. EH-006a - Insufficient disk space for model download (@error-handling)
9. EH-006b - Disk fills up during download (@error-handling)

### 04-worker-lifecycle.feature
**Purpose:** Worker startup, registration, and callbacks  
**Scenarios:** 10  
**Estimated Lines:** ~400

**Scenarios:**
1. Worker startup sequence
2. Worker ready callback
3. EH-012a - Worker binary not found (@error-handling)
4. EH-012b - Worker port already in use (@error-handling)
5. EH-012c - Worker crashes during startup (@error-handling)
6. Worker registration in in-memory registry
7. Worker health check while loading
8. Worker loading progress stream
9. Worker health check when ready
10. EH-016a - Worker loading timeout (@error-handling)

### 05-inference-execution.feature
**Purpose:** Inference request handling and token streaming  
**Scenarios:** 6  
**Estimated Lines:** ~250

**Scenarios:**
1. Inference request with SSE streaming
2. EH-018a - Worker busy with all slots occupied (@error-handling)
3. EH-013a - Worker crashes during inference (@error-handling)
4. EH-013b - Worker hangs during inference (@error-handling)
5. EH-003a - Worker HTTP connection lost mid-inference (@error-handling)
6. EC1 - Connection timeout with retry and backoff (@edge-case)

### 06-error-handling-network.feature
**Purpose:** HTTP, timeout, retry, and cancellation scenarios  
**Scenarios:** 10  
**Estimated Lines:** ~400

**Scenarios:**
1. EH-002a - rbee-hive HTTP connection timeout (@error-handling)
2. EH-002b - rbee-hive returns malformed JSON (@error-handling)
3. Pool preflight connection timeout with retry
4. EC2 - Model download failure with retry (@edge-case)
5. EC4 - Worker crash during inference (@edge-case)
6. Gap-G12a - Client cancellation with Ctrl+C (@edge-case)
7. Gap-G12b - Client disconnects during inference (@edge-case)
8. Gap-G12c - Explicit cancellation endpoint (@edge-case)
9. EC6 - Queue full with retry (@edge-case)
10. EC7 - Model loading timeout (@edge-case)

### 07-error-handling-resources.feature
**Purpose:** RAM, disk, VRAM error scenarios + validation errors  
**Scenarios:** 8  
**Estimated Lines:** ~350

**Scenarios:**
1. EC3 - Insufficient VRAM (@edge-case)
2. EC8 - Version mismatch (@edge-case)
3. EH-015a - Invalid model reference format (@error-handling)
4. EH-015b - Invalid backend name (@error-handling)
5. EH-015c - Device number out of range (@error-handling)
6. EH-017a - Missing API key (@error-handling)
7. EH-017b - Invalid API key (@error-handling)
8. EC10 - Idle timeout and worker auto-shutdown (@edge-case)

### 08-daemon-lifecycle.feature
**Purpose:** Daemon management, shutdown, and deployment modes  
**Scenarios:** 9  
**Estimated Lines:** ~350

**Scenarios:**
1. Rbee-hive remains running as persistent HTTP daemon
2. Rbee-hive monitors worker health
3. Rbee-hive enforces idle timeout (worker dies, pool lives)
4. Cascading shutdown when rbee-hive receives SIGTERM
5. EH-014a - Worker ignores shutdown signal (@error-handling)
6. EH-014b - Graceful shutdown with active request (@error-handling)
7. rbee-keeper exits after inference (CLI dies, daemons live)
8. Ephemeral mode - rbee-keeper spawns rbee-hive
9. Persistent mode - rbee-hive pre-started

### 09-happy-path-flows.feature
**Purpose:** End-to-end success scenarios including registry, preflight, CLI commands  
**Scenarios:** 17  
**Estimated Lines:** ~600

**Scenarios:**
1. Happy path - cold start inference on remote node (@critical)
2. Warm start - reuse existing idle worker
3. Worker registry returns empty list
4. Worker registry returns matching idle worker
5. Worker registry returns matching busy worker
6. Pool preflight health check succeeds
7. Pool preflight detects version mismatch
8. CLI command - install to user paths
9. CLI command - install to system paths
10. Config file loading with XDG priority
11. Remote binary path configuration
12. CLI command - basic inference
13. CLI command - list workers
14. CLI command - check worker health
15. CLI command - manually shutdown worker
16. CLI command - view logs
17. Error response structure validation (moved here for completeness)

## Verification Checklist
- [x] All 91 scenarios accounted for (10+12+9+10+6+10+8+9+17=91) ✅
- [x] No duplicates across files ✅
- [x] Logical grouping by feature ✅
- [x] File sizes reasonable (< 600 lines each) ✅
- [x] Tags preserved (@setup, @error-handling, @critical, @gguf, @edge-case) ✅

## Design Rationale

### Why 9 Files?
- **Separation of concerns:** Each file represents a distinct feature area
- **Maintainability:** Smaller files are easier to navigate and modify
- **Test execution:** Can run specific feature areas independently
- **Logical grouping:** Related scenarios stay together

### File Size Distribution
- Smallest: 05-inference-execution.feature (~250 lines, 6 scenarios)
- Largest: 09-happy-path-flows.feature (~600 lines, 17 scenarios)
- Average: ~370 lines per file

### Tag Strategy
- `@setup` - SSH and registry setup scenarios
- `@error-handling` - Error scenarios with specific error codes
- `@critical` - Critical path scenarios
- `@gguf` - GGUF-specific scenarios
- `@edge-case` - Edge cases and retry logic

## Next Steps
1. Create migration tracking document
2. Create feature files one by one
3. Copy scenarios from test-001.feature
4. Verify compilation after each file
5. Run full verification after all files created
