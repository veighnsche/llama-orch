awesome — here’s a tighter, branched “happy flow” that still reads like what you’ll see on the shell, and each step is annotated with the crate/binary that’s responsible.

I split it into two main entry cases (cold start vs warm start), then branch at each decision (hive present? model present? worker present? VRAM OK?). Crate names and roles are taken from your technical summary so you can wire READMEs directly.  

# Happy Flow (with branches) — `infer "hello" HF:author/minillama`

## 0) User entry (always)

* **Command (user → shell):**
  `rbee-keeper infer --model HF:author/minillama "hello"`
* **Crates/Binaries touched:**
  `rbee-keeper` → `rbee-keeper-cli` (parse) → `rbee-keeper-commands::infer` (dispatch). 
* **Narration:**
  `bee-keeper: starting inference request for HF:author/minillama`

---

## 1) Ensure Queen (cold vs warm)

### 1A. Cold start (queen not running)

* **Keeper → Queen health (HTTP):** `rbee-http-client`
  If `GET /health` fails:
  `bee-keeper: queen is asleep, waking queen…`
* **Keeper starts queen:** `rbee-keeper-queen-lifecycle` uses `daemon-lifecycle` to spawn `queen-rbee` (dev: hardcoded target path)
  Ports: `:8500` (dev default)
  `bee-keeper: launching queen on :8500 and waiting for health…`
* **Keeper polls queen:** retry via `rbee-http-client` until healthy
  `bee-keeper: queen is awake and healthy` 

### 1B. Warm start (queen already running)

* **Keeper → Queen health (HTTP):** `200 OK`
  `bee-keeper: queen is awake and healthy` 

---

## 2) Stream setup Keeper ⇄ Queen

* **Keeper opens SSE:** `GET /events` to queen (`rbee-http-client`)
  `bee-keeper: connected to queen via SSE` 
* **Keeper posts job:** `POST /jobs` (payload: model_ref=`HF:author/minillama`, prompt=`"hello"`)
  Queen replies with `job_id` and a `GET /jobs/:id/events` link. 

---

## 3) Queen selects/ensures Hive (catalog → startup → heartbeat)

### 3.0 Read catalog / registry

* **Queen checks hive catalog (SQLite):** `queen-rbee-hive-registry` (RAM) + `queen-rbee`’s catalog read (source-of-truth note)
  If none: `queen → SSE:`
  `queen: no hives found; adding local machine` 

### Branch A — No hive yet (cold machine)

* **Queen starts local hive:** `queen-rbee-hive-lifecycle` → `daemon-lifecycle` to spawn `rbee-hive` on `:8600` (dev target path)
  `queen: waking the beehive at localhost:8600`
* **Wait for hive heartbeat:** queen’s HTTP server idles; expects heartbeats from hive (no polling) via `heartbeat` crate
  On first heartbeat:
  `queen: first heartbeat from hive localhost; checking capabilities…`
* **Queen asks device detection:** `GET queen→hive /devices`
  Hive runs `rbee-hive-device-detection` and replies (CPU + GPU list), plus model/worker counts (registries).
  `queen: hive localhost has CPU, GPU0 RTX 3060, GPU1 RTX 3090; models: 0, workers: 0` 

### Branch B — Hive present (warm machine)

* **Queen lists hives (RAM):** `queen-rbee-hive-registry`
* **If capabilities unknown/stale:** queen asks hive `GET /devices` (same as above)
* **If capabilities cached & fresh:** skip detection.
  `queen: using registered capabilities of hive ‘localhost’` 

> Notes:
> • Only queen uses SSH for remote hives; after startup all ops are HTTP. If target is remote, `queen-rbee-ssh-client` is used by `hive-lifecycle` to spawn the hive daemon remotely, then switch to HTTP + heartbeats. 

---

## 4) Scheduling (pick device & hive)

* **Queen scheduler (basic):** choose strongest device (e.g., GPU1) on a hive with capacity.
  `queen: basic scheduler picked GPU1 on hive localhost`
* **Crates:** `queen-rbee-worker-registry` (cluster view), `queen-rbee-hive-registry` (hive view). 

### Branch: Multiple hives / remote

* If multiple hives exist: choose by policy (VRAM, load, locality).
* If remote chosen: ensure hive running there (SSH start if needed), then treat like local via HTTP/heartbeat. 

---

## 5) Capacity check (VRAM room?)

* **Queen → Hive:** `GET /capacity?device=gpu1&model=HF:author/minillama`
  Hive consults **VRAM checker** (part of `rbee-hive-worker-lifecycle` + device facts from `rbee-hive-device-detection`).
  `queen: asking hive if there’s room on GPU1…`
  Hive replies `204 No Content` (OK) or `409 Conflict` (not enough room). 

### Branch: Not enough VRAM

* **If 409:**
  `queen: insufficient VRAM on GPU1 for HF:author/minillama; trying next candidate…`
  Retry other GPUs → CPU fallback (if allowed by policy) → else fail the job with guidance.
  **Crates:** same as above. 

---

## 6) Ensure model present (concurrent with worker)

### Branch M0 — Model missing on hive

* **Queen → Hive:** `POST /models/provision { model_ref: HF:author/minillama }`
  Hive replies with SSE link; queen connects immediately.
  `queen: asked hive to download model HF:author/minillama`
* **Hive model-provision:** `rbee-hive-model-provisioner` + `rbee-hive-download-tracker` writes to `rbee-hive-model-catalog` (SQLite).
  Streamed SSE (relayed up to keeper):
  `hive: model size = X MB; starting download`
  `hive: … XX% @ YY MB/s …`
  `hive: model downloaded in Z seconds`
  On completion: queen updates cluster registry view. 

### Branch M1 — Model already present

* **Queen:** skip provision; confirm path via `GET hive /models/:ref` (catalog).
  `queen: model HF:author/minillama found in hive catalog` 

---

## 7) Ensure worker binary present (concurrent with model)

### Branch W0 — Worker missing (dev mode: hardcoded path)

* **Queen → Hive:** `POST /workers/provision { kind: cuda-llm-worker-rbee }`
  Dev path: use local build artifact (`target/.../llm-worker-rbee` w/ CUDA) per your development rule.
  `queen: asked hive to prepare worker cuda-llm-worker-rbee`
* **Hive worker catalog:** registers binary location.
  `queen: worker prepared and ready to deploy` 

### Branch W1 — Worker present

* `queen: worker cuda-llm-worker-rbee already available` 

---

## 8) Start worker process (attach model & port)

* **Queen → Hive:** `POST /workers/start { device: gpu1, model_ref, port_hint: 8601 }`
* **Hive does:**

  * Lookup model path in `rbee-hive-model-catalog` (SQLite)
  * Lookup worker binary path in worker-catalog (RAM)
  * Spawn process: `daemon-lifecycle` (env & args)
  * Register in `rbee-hive-worker-registry` (RAM) with PID, port, state=Loading
  * Start sending worker heartbeats (nested in hive heartbeat) via `heartbeat`
    **Narration:**
    `hive: waking worker on :8601 with model=<path>`
* **Hive → Queen:** returns `port=8601`
* **Queen:** adds “wait-for-worker” task into its SSE queue; when worker heartbeat arrives:
  `queen: CUDA worker is awake` 

### Branch: Worker fails to boot

* Hive restarts as per policy (`rbee-hive-worker-lifecycle` restart count).
* If exceeds retries: queen marks node/device degraded and reschedules elsewhere or fails job. 

---

## 9) Run inference (SSE streaming)

* **Queen → Worker:** `POST /infer { prompt:"hello", stream:true }`
  Worker replies with SSE link; queen connects and relays tokens up to keeper SSE.
  **Narration chain:**
  `queen: connection to worker established; starting inference`
  Keeper shows tokens live:
  `bee-keeper(stdout): he… l… lo…`
  On completion: worker sends `[DONE]` → queen forwards → keeper prints `[DONE]`.
  `queen: worker finished inference` 

### Branch: Non-streaming mode

* If `stream:false`, queen blocks on worker HTTP and pushes a single completion event to keeper. 

---

## 10) Teardown policy (cascading shutdown vs keep-alive)

### 10A. Cold-session policy (your specified “all services off” ending)

* **Keeper → Queen:** `POST /shutdown?cascade=true`
  Queen:

  * `POST /hives/:id/shutdown` (graceful)
  * `POST /workers/:id/shutdown` (graceful)
  * Fallback via SSH kill if remote and unresponsive.
    **Narration:**
    `bee-keeper: asking queen to cascade shutdown`
    `queen: shutting down workers…`
    `queen: shutting down hives…`
    `queen: goodbye` 

### 10B. Warm-session policy (services stay up)

* **No shutdown:** heartbeats continue; models and workers remain cached/idle per policy.
  `queen: keeping hive and worker warm` 

---

# Quick Branch Matrix (what happens when…)

| Decision point      | If yes / present         | If no / missing                                               | Crates / Notes                                                                          |
| ------------------- | ------------------------ | ------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Queen healthy?      | Go to SSE/job            | Keeper spawns queen and polls                                 | `rbee-keeper-queen-lifecycle`, `daemon-lifecycle`, `rbee-http-client`                   |
| Hive exists?        | Use registry device info | Queen starts hive, waits for heartbeat, runs device detection | `queen-rbee-hive-lifecycle`, `heartbeat`, `rbee-hive-device-detection`                  |
| Device capacity ok? | Proceed                  | Try next device → CPU → fail job                              | `rbee-hive-worker-lifecycle` (capacity probe)                                           |
| Model present?      | Skip download            | Provision model via SSE                                       | `rbee-hive-model-provisioner`, `rbee-hive-download-tracker`, `rbee-hive-model-catalog`  |
| Worker present?     | Skip provision           | Provision worker binary (dev: target path)                    | worker catalog (RAM) under hive lifecycle                                               |
| Worker boots?       | Heartbeats → ready       | Restart per policy → reschedule/fail                          | `rbee-hive-worker-registry`, `heartbeat`, restart in lifecycle                          |
| Streaming?          | SSE tokens relay         | Single response event                                         | Worker/Queen HTTP servers; `rbee-http-client`                                           |
| Shutdown?           | Cascade stop             | Keep-alive                                                    | Queen’s hive/worker lifecycle; SSH fallback only in queen                               |

---

# Crate-to-Step Index (for wiring READMEs)

* **Keeper layer:** `rbee-keeper-cli` (args) → `rbee-keeper-commands::infer` (dispatch) → `rbee-keeper-queen-lifecycle` (boot) → `rbee-http-client` (HTTP/SSE). 
* **Queen layer:** `queen-rbee-http-server` (API) • `queen-rbee-hive-lifecycle` (start/stop) • `queen-rbee-hive-registry` & `queen-rbee-worker-registry` (cluster view) • `queen-rbee-ssh-client` (only here, remote boot) • `queen-rbee-preflight` (optional pre-checks). 
* **Hive layer:** `rbee-hive-http-server` (API) • `rbee-hive-device-detection` (capabilities) • `rbee-hive-model-provisioner` + `rbee-hive-download-tracker` + `rbee-hive-model-catalog` (models) • `rbee-hive-worker-lifecycle` + `rbee-hive-worker-registry` (workers) • `heartbeat` (hive & nested worker beats). 
* **Worker layer:** `worker-rbee-http-server` (infer endpoints) • backend in `30_llm_worker_rbee/src/backend/*` (tokenizer, sampling, model impl) • `heartbeat` (to hive). 

---

## Copy-ready narration (you can sprinkle these directly in SSE streams)

* keeper: `queen is asleep, waking queen…`
* keeper: `queen is awake and healthy`
* keeper: `connected to queen via SSE`
* queen: `no hives found; adding local machine`
* queen: `waking the beehive at localhost:8600`
* queen: `first heartbeat from hive localhost; checking capabilities…`
* queen: `basic scheduler picked GPU1 for this job`
* queen: `asking hive if there’s room on GPU1`
* queen: `model HF:author/minillama: provisioning started`
* hive: `model size X MB; starting download`
* hive: `… 35% @ 42 MB/s …`
* hive: `model downloaded in Z seconds`
* hive: `waking worker on :8601 with model=<path>`
* queen: `CUDA worker is awake`
* queen: `connection to worker established; starting inference`
* keeper(stdout): `he… l… lo…`
* queen: `worker finished inference`
* queen (cascade): `shutting down workers… shutting down hives…`

All of this matches your architecture and crate boundaries so you can now drop these into READMEs and BDD steps one part at a time without adding ceremony. If you want, I can turn this into per-crate README skeletons next. 
