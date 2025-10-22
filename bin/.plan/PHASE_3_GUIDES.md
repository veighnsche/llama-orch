# PHASE 3: Hive Crate Behavior Discovery

**Teams:** TEAM-223 to TEAM-229  
**Duration:** 1 day (all teams work concurrently)  
**Output:** 7 behavior inventory documents

---

## TEAM-223: device-detection

**Component:** `bin/25_rbee_hive_crates/device-detection`  
**Complexity:** High  
**Output:** `.plan/TEAM_223_DEVICE_DETECTION_BEHAVIORS.md`

### Investigation Areas

#### 1. GPU Detection
- Document CUDA detection logic
- Document GPU enumeration
- Document device property queries
- Document fallback to CPU

#### 2. VRAM Detection
- Document VRAM total detection
- Document VRAM available detection
- Document multi-GPU VRAM tracking

#### 3. Device Capabilities
- Document capability discovery
- Document compute capability detection
- Document driver version detection

#### 4. Error Handling
- Document no GPU scenarios
- Document CUDA unavailable scenarios
- Document driver errors
- Document device query failures

#### 5. Platform Support
- Document Linux support
- Document Windows support (if any)
- Document macOS support (if any)

---

## TEAM-224: download-tracker

**Component:** `bin/25_rbee_hive_crates/download-tracker`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_224_DOWNLOAD_TRACKER_BEHAVIORS.md`

### Investigation Areas

#### 1. Download Tracking
- Document active download tracking
- Document download progress calculation
- Document download state management

#### 2. Progress Reporting
- Document progress metrics
- Document ETA calculation
- Document speed calculation

#### 3. Download Lifecycle
- Document download start
- Document download pause/resume
- Document download completion
- Document download failure

#### 4. Concurrency
- Document concurrent downloads
- Document download limits
- Document resource management

#### 5. Cleanup
- Document partial download cleanup
- Document failed download cleanup

---

## TEAM-225: model-catalog

**Component:** `bin/25_rbee_hive_crates/model-catalog`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_225_MODEL_CATALOG_BEHAVIORS.md`

### Investigation Areas

#### 1. Catalog Management
- Document model registration
- Document model lookup
- Document model metadata

#### 2. Model Discovery
- Document local model discovery
- Document model validation
- Document model indexing

#### 3. Model Metadata
- Document metadata schema
- Document metadata validation
- Document metadata updates

#### 4. Persistence
- Document catalog storage
- Document catalog loading
- Document catalog updates

---

## TEAM-226: model-provisioner

**Component:** `bin/25_rbee_hive_crates/model-provisioner`  
**Complexity:** High  
**Output:** `.plan/TEAM_226_MODEL_PROVISIONER_BEHAVIORS.md`

### Investigation Areas

#### 1. Model Discovery
- Document `find_local_model()` behavior
- Document search paths
- Document model resolution

#### 2. Model Download
- Document `download_model()` behavior
- Document download sources
- Document download verification
- Document progress tracking

#### 3. Model Validation
- Document model file validation
- Document checksum verification
- Document format validation

#### 4. Cache Management
- Document cache directory structure
- Document cache cleanup
- Document cache limits

#### 5. Error Handling
- Document download failures
- Document validation failures
- Document retry logic

---

## TEAM-227: monitor

**Component:** `bin/25_rbee_hive_crates/monitor`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_227_MONITOR_BEHAVIORS.md`

### Investigation Areas

#### 1. System Monitoring
- Document what is monitored
- Document monitoring frequency
- Document data collection

#### 2. Worker Monitoring
- Document worker health checks
- Document worker status tracking
- Document worker failure detection

#### 3. Resource Monitoring
- Document CPU monitoring
- Document memory monitoring
- Document GPU monitoring
- Document disk monitoring

#### 4. Alert System
- Document alert conditions
- Document alert thresholds
- Document alert delivery

---

## TEAM-228: vram-checker

**Component:** `bin/25_rbee_hive_crates/vram-checker`  
**Complexity:** Low  
**Output:** `.plan/TEAM_228_VRAM_CHECKER_BEHAVIORS.md`

### Investigation Areas

#### 1. VRAM Checking
- Document VRAM availability checks
- Document VRAM requirement calculation
- Document multi-GPU VRAM selection

#### 2. Allocation Logic
- Document VRAM allocation strategy
- Document VRAM reservation
- Document VRAM release

#### 3. Error Handling
- Document insufficient VRAM scenarios
- Document VRAM exhaustion handling
- Document allocation failures

---

## TEAM-229: worker-management

**Component:** `bin/25_rbee_hive_crates/` (worker-catalog, worker-lifecycle, worker-registry)  
**Complexity:** High  
**Output:** `.plan/TEAM_229_WORKER_MANAGEMENT_BEHAVIORS.md`

### Investigation Areas

#### 1. Worker Catalog
- Document worker registration
- Document worker lookup
- Document worker metadata

#### 2. Worker Lifecycle
- Document worker spawn
- Document worker stop
- Document worker restart
- Document worker crash detection

#### 3. Worker Registry
- Document registry interface
- Document state tracking
- Document heartbeat handling

#### 4. State Management
- Document worker states
- Document state transitions
- Document invariants

#### 5. Integration
- Document how these 3 crates work together
- Document data flow between them
- Document shared contracts

---

## Investigation Methodology

### Step 1: Identify Files
```bash
find bin/25_rbee_hive_crates/[crate-name] -name "*.rs"
```

### Step 2: Read Cargo.toml
```bash
cat bin/25_rbee_hive_crates/[crate-name]/Cargo.toml
```

### Step 3: Read Source Files
```bash
cat bin/25_rbee_hive_crates/[crate-name]/src/lib.rs
```

### Step 4: Check Tests
```bash
find bin/25_rbee_hive_crates/[crate-name] -name "*test*.rs"
find bin/25_rbee_hive_crates/[crate-name]/bdd -name "*.feature"
```

---

## Deliverables Checklist

Each team must deliver:
- [ ] Behavior inventory document
- [ ] Follows template structure
- [ ] Max 3 pages
- [ ] All public APIs documented
- [ ] All state machines documented
- [ ] All error paths documented
- [ ] All integration points documented
- [ ] Test coverage gaps identified
- [ ] Code signatures added (`// TEAM-XXX: Investigated`)

---

## Success Criteria

### Per-Team
- ✅ Complete behavior inventory
- ✅ All behaviors documented
- ✅ All edge cases identified
- ✅ Test gaps identified

### Phase 3
- ✅ All 7 teams completed
- ✅ All inventories delivered
- ✅ Ready for Phase 4

---

## Coordination

### Concurrent Work
- All 7 teams work independently
- No dependencies between teams
- Can start as soon as Phase 2 completes

### Special Note: TEAM-229
- Investigates 3 related crates
- Documents how they integrate
- Single document covering all 3

---

**Status:** READY (after Phase 2)  
**Next:** Phase 4 (Shared crates)
