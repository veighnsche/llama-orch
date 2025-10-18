# Feature File Redesign - CORRECTED
# Created by: TEAM-077
# Date: 2025-10-11
# Status: ARCHITECTURAL FIX

## Problem Identified

**WRONG APPROACH (Original):**
- Created separate "error-handling" feature files
- Created "happy-path-flows" feature file
- Treated error handling as a feature (it's not - it's a cross-cutting concern)
- Treated happy path as a feature (it's not - it's a composition of features)

**CORRECT APPROACH:**
- Each feature file contains BOTH happy path AND error scenarios
- Error scenarios are distributed into their respective features
- No separate "error-handling" or "happy-path" files

## Correct Feature File Structure

### 01-ssh-registry-management.feature
**Feature:** SSH connection setup and node registry management
**Scenarios:**
- Happy: Add remote rbee-hive node to registry
- Happy: Install rbee-hive on remote node
- Happy: List registered rbee-hive nodes
- Happy: Remove node from rbee-hive registry
- Error: EH-001a - SSH connection timeout
- Error: EH-001b - SSH authentication failure
- Error: EH-001c - SSH command execution failure
- Error: EH-011a - Invalid SSH key path
- Error: EH-011b - Duplicate node name
- Error: Inference fails when node not in registry

### 02-model-provisioning.feature
**Feature:** Model download, catalog, and GGUF support
**Scenarios:**
- Happy: Model found in SQLite catalog
- Happy: Model not found - download with progress
- Happy: Model catalog registration after download
- Happy: GGUF model detection by file extension
- Happy: GGUF metadata extraction
- Happy: GGUF quantization formats supported
- Happy: GGUF model size calculation
- Error: EH-007a - Model not found on Hugging Face
- Error: EH-007b - Model repository is private
- Error: EH-008a - Model download timeout
- Error: EH-008b - Model download fails with retry
- Error: EH-008c - Downloaded model checksum mismatch
- Error: EC2 - Model download failure with retry

### 03-worker-preflight-checks.feature
**Feature:** Resource validation before worker startup
**Scenarios:**
- Happy: Worker preflight RAM check passes
- Happy: Worker preflight backend check passes
- Error: EH-004a - Worker preflight RAM check fails
- Error: EH-004b - RAM exhausted during model loading
- Error: EH-005a - VRAM exhausted on CUDA device
- Error: EH-009a - Backend not available
- Error: EH-009b - CUDA not installed
- Error: EH-006a - Insufficient disk space for model download
- Error: EH-006b - Disk fills up during download
- Error: EC3 - Insufficient VRAM

### 04-worker-lifecycle.feature
**Feature:** Worker startup, registration, and callbacks
**Scenarios:**
- Happy: Worker startup sequence
- Happy: Worker ready callback
- Happy: Worker registration in in-memory registry
- Happy: Worker health check while loading
- Happy: Worker loading progress stream
- Happy: Worker health check when ready
- Error: EH-012a - Worker binary not found
- Error: EH-012b - Worker port already in use
- Error: EH-012c - Worker crashes during startup
- Error: EH-016a - Worker loading timeout
- Error: EC7 - Model loading timeout

### 05-inference-execution.feature
**Feature:** Inference request handling and token streaming
**Scenarios:**
- Happy: Inference request with SSE streaming
- Error: EH-018a - Worker busy with all slots occupied
- Error: EH-013a - Worker crashes during inference
- Error: EH-013b - Worker hangs during inference
- Error: EH-003a - Worker HTTP connection lost mid-inference
- Error: EC1 - Connection timeout with retry and backoff
- Error: EC4 - Worker crash during inference
- Error: EC6 - Queue full with retry
- Cancellation: Gap-G12a - Client cancellation with Ctrl+C
- Cancellation: Gap-G12b - Client disconnects during inference
- Cancellation: Gap-G12c - Explicit cancellation endpoint

### 06-pool-management.feature (NEW - was scattered)
**Feature:** Pool-level health checks and version management
**Scenarios:**
- Happy: Pool preflight health check succeeds
- Happy: Worker registry returns empty list
- Happy: Worker registry returns matching idle worker
- Happy: Worker registry returns matching busy worker
- Error: EH-002a - rbee-hive HTTP connection timeout
- Error: EH-002b - rbee-hive returns malformed JSON
- Error: Pool preflight connection timeout with retry
- Error: Pool preflight detects version mismatch
- Error: EC8 - Version mismatch

### 07-daemon-lifecycle.feature
**Feature:** Daemon management, shutdown, and deployment modes
**Scenarios:**
- Happy: Rbee-hive remains running as persistent HTTP daemon
- Happy: Rbee-hive monitors worker health
- Happy: Rbee-hive enforces idle timeout (worker dies, pool lives)
- Happy: Cascading shutdown when rbee-hive receives SIGTERM
- Happy: rbee-keeper exits after inference (CLI dies, daemons live)
- Happy: Ephemeral mode - rbee-keeper spawns rbee-hive
- Happy: Persistent mode - rbee-hive pre-started
- Error: EH-014a - Worker ignores shutdown signal
- Error: EH-014b - Graceful shutdown with active request
- Error: EC10 - Idle timeout and worker auto-shutdown

### 08-input-validation.feature (NEW - was scattered)
**Feature:** CLI input validation and authentication
**Scenarios:**
- Error: EH-015a - Invalid model reference format
- Error: EH-015b - Invalid backend name
- Error: EH-015c - Device number out of range
- Error: EH-017a - Missing API key
- Error: EH-017b - Invalid API key
- Happy: Error response structure validation

### 09-cli-commands.feature (NEW - was "happy path")
**Feature:** CLI command interface
**Scenarios:**
- Happy: CLI command - install to user paths
- Happy: CLI command - install to system paths
- Happy: Config file loading with XDG priority
- Happy: Remote binary path configuration
- Happy: CLI command - basic inference
- Happy: CLI command - list workers
- Happy: CLI command - check worker health
- Happy: CLI command - manually shutdown worker
- Happy: CLI command - view logs

### 10-end-to-end-flows.feature (NEW - integration scenarios)
**Feature:** Complete end-to-end workflows
**Scenarios:**
- Integration: Happy path - cold start inference on remote node
- Integration: Warm start - reuse existing idle worker

## Summary of Changes

**Files to Merge/Reorganize:**
- `06-error-handling-network.feature` → Distribute scenarios to respective features
- `07-error-handling-resources.feature` → Distribute scenarios to respective features
- `09-happy-path-flows.feature` → Split into CLI commands and E2E flows

**New Files:**
- `06-pool-management.feature` - Pool-level concerns
- `08-input-validation.feature` - Validation and auth
- `09-cli-commands.feature` - CLI interface
- `10-end-to-end-flows.feature` - Integration tests

**Total:** 10 feature files (was 9)
**Scenarios:** Still 91 total

## Architectural Principle

**Each feature file should:**
1. Represent a single capability/behavior
2. Include BOTH happy path AND error scenarios for that capability
3. Be independently testable
4. Not be a cross-cutting concern (like "error handling")
5. Not be a composition (like "happy path flows")

**Error scenarios belong WITH the feature they test, not isolated.**
