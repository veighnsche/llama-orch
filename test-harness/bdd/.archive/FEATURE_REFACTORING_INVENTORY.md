# Feature Refactoring Inventory
# Created by: TEAM-077
# Date: 2025-10-11

## Current State
- File: test-001.feature
- Total lines: 1675
- Total scenarios: 91
- Status: NEEDS SPLITTING

## Scenario Inventory by Category

### SSH & Registry Setup (10 scenarios)
1. Add remote rbee-hive node to registry
2. EH-001a - SSH connection timeout
3. EH-001b - SSH authentication failure
4. EH-001c - SSH command execution failure
5. Install rbee-hive on remote node
6. List registered rbee-hive nodes
7. Remove node from rbee-hive registry
8. EH-011a - Invalid SSH key path
9. EH-011b - Duplicate node name
10. Inference fails when node not in registry

### Happy Path Flows (2 scenarios)
11. Happy path - cold start inference on remote node
12. Warm start - reuse existing idle worker

### Worker Registry Operations (3 scenarios)
13. Worker registry returns empty list
14. Worker registry returns matching idle worker
15. Worker registry returns matching busy worker

### Pool Preflight Checks (4 scenarios)
16. Pool preflight health check succeeds
17. EH-002a - rbee-hive HTTP connection timeout
18. EH-002b - rbee-hive returns malformed JSON
19. Pool preflight detects version mismatch
20. Pool preflight connection timeout with retry

### Model Provisioning (12 scenarios)
21. Model found in SQLite catalog
22. Model not found - download with progress
23. EH-007a - Model not found on Hugging Face
24. EH-007b - Model repository is private
25. EH-008a - Model download timeout
26. EH-008b - Model download fails with retry
27. EH-008c - Downloaded model checksum mismatch
28. Model catalog registration after download
29. GGUF model detection by file extension
30. GGUF metadata extraction
31. GGUF quantization formats supported
32. GGUF model size calculation

### Worker Preflight Checks (8 scenarios)
33. Worker preflight RAM check passes
34. EH-004a - Worker preflight RAM check fails
35. EH-004b - RAM exhausted during model loading
36. Worker preflight backend check passes
37. EH-005a - VRAM exhausted on CUDA device
38. EH-009a - Backend not available
39. EH-009b - CUDA not installed
40. EH-006a - Insufficient disk space for model download
41. EH-006b - Disk fills up during download

### Worker Lifecycle (10 scenarios)
42. Worker startup sequence
43. Worker ready callback
44. EH-012a - Worker binary not found
45. EH-012b - Worker port already in use
46. EH-012c - Worker crashes during startup
47. Worker registration in in-memory registry
48. Worker health check while loading
49. Worker loading progress stream
50. Worker health check when ready
51. EH-016a - Worker loading timeout

### Inference Execution (6 scenarios)
52. Inference request with SSE streaming
53. EH-018a - Worker busy with all slots occupied
54. EH-013a - Worker crashes during inference
55. EH-013b - Worker hangs during inference
56. EH-003a - Worker HTTP connection lost mid-inference
57. EC1 - Connection timeout with retry and backoff

### Error Handling - Network & Retry (2 scenarios)
58. EC2 - Model download failure with retry
59. EC3 - Insufficient VRAM

### Error Handling - Worker Failures (1 scenario)
60. EC4 - Worker crash during inference

### Cancellation & Client Disconnect (3 scenarios)
61. Gap-G12a - Client cancellation with Ctrl+C
62. Gap-G12b - Client disconnects during inference
63. Gap-G12c - Explicit cancellation endpoint

### Error Handling - Queue & Timeouts (3 scenarios)
64. EC6 - Queue full with retry
65. EC7 - Model loading timeout
66. EC8 - Version mismatch

### Error Handling - Validation (6 scenarios)
67. EH-015a - Invalid model reference format
68. EH-015b - Invalid backend name
69. EH-015c - Device number out of range
70. EH-017a - Missing API key
71. EH-017b - Invalid API key
72. EC10 - Idle timeout and worker auto-shutdown

### Daemon Lifecycle (7 scenarios)
73. Rbee-hive remains running as persistent HTTP daemon
74. Rbee-hive monitors worker health
75. Rbee-hive enforces idle timeout (worker dies, pool lives)
76. Cascading shutdown when rbee-hive receives SIGTERM
77. EH-014a - Worker ignores shutdown signal
78. EH-014b - Graceful shutdown with active request
79. rbee-keeper exits after inference (CLI dies, daemons live)

### Deployment Modes (2 scenarios)
80. Ephemeral mode - rbee-keeper spawns rbee-hive
81. Persistent mode - rbee-hive pre-started

### Error Response Structure (1 scenario)
82. Error response structure validation

### CLI Commands (10 scenarios)
83. CLI command - install to user paths
84. CLI command - install to system paths
85. Config file loading with XDG priority
86. Remote binary path configuration
87. CLI command - basic inference
88. CLI command - list workers
89. CLI command - check worker health
90. CLI command - manually shutdown worker
91. CLI command - view logs

## Verification
- [x] All 91 scenarios listed
- [x] All scenarios categorized by feature
- [x] No duplicates
- [x] No gaps in numbering

## Proposed Feature Files (9 files)

### 01-ssh-registry-management.feature (10 scenarios)
SSH connection setup and node registry management

### 02-model-provisioning.feature (12 scenarios)
Model download, catalog, and GGUF support

### 03-worker-preflight-checks.feature (8 scenarios)
Resource validation before worker startup

### 04-worker-lifecycle.feature (10 scenarios)
Worker startup, registration, and callbacks

### 05-inference-execution.feature (6 scenarios)
Inference request handling and token streaming

### 06-error-handling-network.feature (10 scenarios)
HTTP, timeout, retry, and cancellation scenarios

### 07-error-handling-resources.feature (9 scenarios)
RAM, disk, VRAM error scenarios + validation errors

### 08-daemon-lifecycle.feature (9 scenarios)
Daemon management, shutdown, and deployment modes

### 09-happy-path-flows.feature (17 scenarios)
End-to-end success scenarios including registry, preflight, CLI commands

**Total: 91 scenarios across 9 feature files**

## Status
- [x] Phase 1: Investigation & Inventory - COMPLETE
- [ ] Phase 2: Feature File Design
- [ ] Phase 3: Create New Feature Files
- [ ] Phase 4: Verification
- [ ] Phase 5: Cleanup
