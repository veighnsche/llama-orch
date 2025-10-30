# Solution 2: Parent-Child Registration

**Approach:** Spawner tells Queen about new workers  
**Complexity:** MEDIUM  
**Recommended:** ⭐⭐⭐⭐ **GOOD ALTERNATIVE**

---

## 🎯 Core Idea

Leverage the fact that **the component that spawns a worker knows its port**.

**Key Insight:** Don't try to discover workers. Instead, let the spawner register them.

---

## 🏗️ Architecture

```
┌────────────┐
│    GUI     │  Spawns worker locally
└─────┬──────┘
      │
      │ 1. Spawn worker on port 8080
      │ 2. POST /v1/queen/register-worker
      │    {worker_id, hostname, port}
      ▼
┌────────────┐
│   Queen    │  Knows about worker now!
└─────┬──────┘
      │
      │ 3. Queen fetches capabilities
      │    GET http://localhost:8080/capabilities
      ▼
┌────────────┐
│  Worker    │  Now Queen can communicate
└────────────┘
      │
      │ 4. Worker starts heartbeats
      └─────► Queen
```

---

## 📋 Component Responsibilities

### **Queen (NEW API)**

**New Endpoint:**
```
POST /v1/queen/register-worker

Body:
{
  "worker_id": "worker-llm-8080",
  "hostname": "localhost",
  "port": 8080,
  "worker_type": "llm",
  "model": "llama-3.2-1b",
  "spawned_by": "gui"  // or "hive-localhost"
}

Response: 200 OK
```

**Behavior:**
1. Receive registration
2. Store worker in WorkerRegistry (mark as "Registered", not "Online" yet)
3. Fetch capabilities: `GET http://{hostname}:{port}/capabilities`
4. Update WorkerRegistry with capabilities
5. Mark worker as "Online" when first heartbeat received

---

### **GUI (Spawner)**

**When spawning worker:**

```rust
// 1. Spawn worker process
let port = find_available_port(8080..8200)?;
let worker_process = spawn_worker_process(worker_type, model, port).await?;

// 2. Wait for worker to be ready
wait_for_worker_health(port).await?;

// 3. Register worker with Queen (if Queen exists)
if let Some(queen_url) = get_queen_url() {
    register_worker_with_queen(&queen_url, WorkerRegistration {
        worker_id: format!("worker-{}-{}", worker_type, port),
        hostname: "localhost",
        port,
        worker_type,
        model: Some(model),
        spawned_by: "gui",
    }).await?;
}

// 4. Store worker in GUI's local registry (for Queenless operation)
gui_state.workers.insert(worker_id, worker_info);
```

---

### **Hive (Spawner)**

**When spawning worker (via job from Queen):**

```rust
// 1. Spawn worker process
let port = find_available_port(8080..8200)?;
let worker_process = spawn_worker_process(worker_type, model, port).await?;

// 2. Wait for worker to be ready
wait_for_worker_health(port).await?;

// 3. Register worker with Queen
register_worker_with_queen(&queen_url, WorkerRegistration {
    worker_id: format!("worker-{}-{}", worker_type, port),
    hostname: get_hive_hostname(),
    port,
    worker_type,
    model: Some(model),
    spawned_by: format!("hive-{}", hive_id),
}).await?;

// 4. Tell worker the Queen URL
// (Pass as CLI arg: --queen-url)
```

---

### **Worker**

**On startup:**

```rust
// Worker receives Queen URL via CLI arg
let queen_url = args.queen_url;

if let Some(queen_url) = queen_url {
    // Start heartbeat task
    // Queen already knows about us (spawner registered us)
    start_heartbeat_task(worker_info, queen_url).await;
} else {
    // No Queen URL - Queenless mode
    // Just listen for requests, no heartbeats
}
```

---

## 🔄 Discovery Flows

### **Flow 1: GUI Spawns Worker Before Queen**

```
1. GUI spawns Worker on port 8080
2. GUI tries to register worker with Queen → 404 (Queen doesn't exist)
3. GUI stores worker in local registry
4. User does inference via GUI → Worker
5. Queen starts later
6. Queen queries... nobody? ❌

Problem: Queen doesn't know about pre-existing workers!
```

**❌ This breaks the Queenless requirement!**

---

### **Flow 2: GUI Spawns Worker After Queen**

```
1. Queen starts
2. GUI spawns Worker on port 8080
3. GUI registers worker with Queen → 200 OK
4. Queen fetches worker capabilities
5. Queen marks worker as "Registered"
6. Worker starts heartbeats to Queen
7. Queen marks worker as "Online"
8. ✅ Works!
```

---

### **Flow 3: Hive Spawns Worker (Normal)**

```
1. Queen sends job to Hive: "Spawn llm-worker with llama-3.2-1b"
2. Hive spawns worker on port 8080
3. Hive registers worker with Queen → 200 OK
4. Queen fetches worker capabilities
5. Queen marks worker as "Registered"
6. Worker starts heartbeats to Queen
7. Queen marks worker as "Online"
8. ✅ Works!
```

---

## ✅ Advantages

### **1. Simple**

- No new daemon
- No network discovery
- Leverages existing knowledge (spawner knows port)

### **2. No Hive Worker Registry**

- Hive just forwards registration to Queen
- Hive doesn't track workers
- ✅ Meets user requirement

### **3. No Hive Worker Heartbeats**

- Workers heartbeat directly to Queen
- Hive never receives worker heartbeats
- ✅ Meets user requirement

### **4. Queen-Controlled**

- Queen is the source of truth
- Queen decides what to do with registration
- Clean ownership

---

## ❌ Disadvantages

### **1. Requires Queen to Exist** ⚠️

- If Queen doesn't exist, registration fails
- Breaks Queenless requirement
- **CRITICAL FLAW**

### **2. No Discovery of Pre-existing Workers**

- If workers started before Queen, Queen never learns about them
- Queen can't discover workers retroactively
- **CRITICAL FLAW**

### **3. Coordination Overhead**

- Spawner must know Queen URL
- Spawner must wait for registration success
- More moving parts

---

## 🛠️ Mitigation: Hybrid Approach

**Problem:** Workers started before Queen are lost.

**Solution:** Combine with Worker Self-Registration:

```rust
// Worker on startup
if let Some(queen_url) = args.queen_url {
    // Self-register with Queen (if Queen exists)
    self_register_with_queen(&queen_url, worker_info).await?;
    
    // Start heartbeats
    start_heartbeat_task(worker_info, queen_url).await;
} else {
    // Queenless mode - just serve requests
}
```

**But this requires Worker to know Queen URL...**

---

## 🎯 Verdict

**⭐⭐⭐⭐ GOOD, BUT INSUFFICIENT ALONE**

**Why NOT Recommended as Sole Solution:**
- ❌ Breaks Queenless requirement
- ❌ Can't discover pre-existing workers
- ❌ Requires Queen to exist at spawn time

**Why Still Useful:**
- ✅ Good as PART of a solution
- ✅ Optimizes the common case (Queen exists)
- ✅ Clean and simple

**Recommendation:** **Use as part of Solution 3 (Hybrid)**

---

## 🔄 Integration with Solution 1 (Registry Service)

**Best of Both Worlds:**

```rust
// Spawner (GUI or Hive)
async fn spawn_worker(...) {
    let port = find_available_port()?;
    let worker_process = spawn_worker_process(..., port).await?;
    
    // 1. Register with registry (if exists)
    if registry_reachable().await {
        register_with_registry(worker_info).await?;
    }
    
    // 2. Register with Queen (if exists) - OPTIMIZATION
    if queen_reachable().await {
        register_with_queen(worker_info).await?;
    }
    
    // Result: Worker is discoverable via registry
    //         AND Queen is notified immediately (fast path)
}
```

---

**This solution is good, but needs to be combined with Solution 1 to handle Queenless scenarios.**
