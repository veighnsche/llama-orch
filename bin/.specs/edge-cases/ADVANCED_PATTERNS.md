# Advanced Patterns from Production LLM Engines

Analysis of candle-vllm, mistral.rs, Ollama, and llama.cpp for rbees architecture.

---

## 1. Advanced Scheduling Patterns

### **Preemption Strategies (candle-vllm)**

**Pattern:** When GPU memory is exhausted, preempt running sequences to make room.

**Two preemption methods:**

1. **Preempt by Recompute** (for single sequences)
   - Free KV cache blocks
   - Move sequence back to waiting queue
   - Recompute from scratch when resources available
   - **Advantage:** No CPU memory needed
   - **Disadvantage:** Wastes computation

2. **Preempt by Swap** (for multi-sequence batches)
   - Swap KV cache blocks from GPU → CPU
   - Move sequence to "swapped_out" queue
   - Swap back when resources available
   - **Advantage:** Preserves computation
   - **Disadvantage:** Requires CPU memory + swap overhead

**Implementation:**
```rust
enum PreemptionMethod {
    Recompute,  // Free and recompute
    Swap,       // Swap to CPU
}

fn preempt(&mut self, seq: Sequence, method: PreemptionMethod) {
    match method {
        PreemptionMethod::Recompute => {
            self.block_engine.free_sequence(&seq);
            seq.set_state(SequenceState::Waiting);
            self.waiting.push_front(seq);  // High priority
        }
        PreemptionMethod::Swap => {
            if !self.block_engine.can_swap_out(&seq) {
                // Abort if cannot swap
                self.abort_sequence(&seq);
                return;
            }
            let blocks_to_swap = self.block_engine.swap_out(&seq);
            self.cache_engine.swap_out(blocks_to_swap)?;
            seq.set_state(SequenceState::Swapped);
            self.swapped_out.push_back(seq);
        }
    }
}
```

**For rbees:**
- **MVP:** Preempt by recompute only (simpler)
- **Post-MVP:** Add swap support for long-running requests

---

### **Priority-Based Scheduling (mistral.rs)**

**Pattern:** Sequences have urgency/priority scores that increase over time.

**Bucketing strategy:**
```rust
// Group sequences by length
let mut buckets: HashMap<usize, Vec<Sequence>> = HashMap::new();
let mut priorities: HashMap<usize, f64> = HashMap::new();

for seq in running {
    let len = seq.len();
    buckets.entry(len).or_default().push(seq);
    *priorities.entry(len).or_default() += seq.compute_priority();
}

// Run highest priority bucket
let highest_priority_bucket = priorities
    .iter()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(len, _)| len);

// Move other buckets back to waiting (with urgency++)
for (len, seqs) in buckets {
    if len != *highest_priority_bucket {
        for seq in seqs {
            waiting.push_back(seq.add_urgency());
        }
    }
}
```

**Priority calculation:**
```rust
impl Sequence {
    fn compute_priority(&self) -> f64 {
        let age = (now() - self.arrival_time).as_secs_f64();
        let urgency = self.urgency as f64;
        age * urgency  // Older + more urgent = higher priority
    }
    
    fn add_urgency(mut self) -> Self {
        self.urgency += 1;
        self
    }
}
```

**For rbees:**
- Track sequence arrival time
- Increment urgency when waitlisted
- Prevent starvation of long-waiting requests

---

### **Chunked Prefill (candle-vllm)**

**Pattern:** Split large prompts into chunks to avoid blocking other requests.

**Implementation:**
```rust
const PREFILL_CHUNK_SIZE: usize = 8192;

fn filter_prefill_finished(
    &mut self,
    scheduled: &[Sequence],
) -> (Vec<u32>, Vec<Sequence>) {
    let mut finished = Vec::new();
    let mut unfinished = Vec::new();
    
    for (i, seq) in scheduled.iter().enumerate() {
        let prompt_len = seq.get_prompt_len();
        let cached_tokens = seq.get_num_cached_tokens();
        
        if prompt_len < PREFILL_CHUNK_SIZE 
            || cached_tokens + PREFILL_CHUNK_SIZE >= prompt_len {
            // Prefill complete
            finished.push(i as u32);
        } else {
            // Partial prefill, push back to waiting
            seq.set_num_cached_tokens(cached_tokens + PREFILL_CHUNK_SIZE);
            seq.set_state(SequenceState::Pending);
            unfinished.push(seq.clone());
        }
    }
    
    (finished, unfinished)
}
```

**For rbees:**
- Chunk prompts > 8K tokens
- Stream progress during chunked prefill
- Prevents head-of-line blocking

---

## 2. Advanced Memory Management

### **Prefix Caching with LRU Eviction (mistral.rs)**

**Pattern:** Cache prompt prefixes on device, evict to CPU when limit reached.

**Implementation:**
```rust
struct PrefixCache {
    caches: HashMap<String, CachedSequence>,
    n_on_device: usize,  // Max allowed on device
}

impl PrefixCache {
    fn evict_caches(&mut self) -> Result<usize> {
        let mut n_on_device = self.count_on_device();
        let mut n_evicted = 0;
        
        // Evict oldest first (LRU)
        for cache in self.caches.values_mut() {
            if n_on_device - n_evicted <= self.n_on_device {
                break;
            }
            
            if cache.is_on_device() {
                cache.move_to_cpu()?;
                n_evicted += 1;
            }
        }
        
        Ok(n_evicted)
    }
}
```

**For rbees:**
- Cache common prompt prefixes (system prompts, few-shot examples)
- Evict to CPU when GPU memory pressure
- Restore from CPU when needed

---

### **Block-Level KV Cache Management (candle-vllm)**

**Pattern:** Manage KV cache in fixed-size blocks for efficient allocation.

**Block engine:**
```rust
struct BlockEngine {
    block_size: usize,  // e.g., 16 tokens per block
    gpu_blocks: Vec<PhysicalBlock>,
    cpu_blocks: Vec<PhysicalBlock>,
    block_tables: HashMap<SequenceId, Vec<BlockId>>,
}

impl BlockEngine {
    fn allocate(&mut self, seq: &Sequence) -> Result<()> {
        let num_blocks_needed = (seq.len() + self.block_size - 1) / self.block_size;
        let free_blocks = self.get_free_gpu_blocks();
        
        if free_blocks.len() < num_blocks_needed {
            return Err(AllocStatus::Later);
        }
        
        let allocated = free_blocks.drain(..num_blocks_needed).collect();
        self.block_tables.insert(seq.id(), allocated);
        Ok(())
    }
    
    fn can_swap_out(&self, seq: &Sequence) -> bool {
        let blocks_needed = self.block_tables[&seq.id()].len();
        self.get_free_cpu_blocks().len() >= blocks_needed
    }
    
    fn swap_out(&mut self, seq: &Sequence) -> HashMap<GPUBlock, CPUBlock> {
        let gpu_blocks = self.block_tables.remove(&seq.id()).unwrap();
        let cpu_blocks = self.allocate_cpu_blocks(gpu_blocks.len());
        
        gpu_blocks.into_iter()
            .zip(cpu_blocks.iter())
            .map(|(gpu, cpu)| (gpu, *cpu))
            .collect()
    }
}
```

**For rbees (post-MVP):**
- Implement PagedAttention with block-level KV cache
- Enables efficient memory sharing and swapping
- Reduces fragmentation

---

## 3. Advanced Request Handling

### **Waiting Timeout with Eviction (mistral.rs)**

**Pattern:** If a sequence waits too long, evict a running sequence to make room.

**Implementation:**
```rust
const WAITING_TIMEOUT: usize = 10;  // scheduling cycles

fn schedule(&mut self) -> SchedulerOutput {
    for seq in &self.waiting {
        let can_allocate = self.block_engine.can_allocate(seq);
        
        match can_allocate {
            AllocStatus::Later { waitlisted_count } => {
                if waitlisted_count > WAITING_TIMEOUT {
                    // Evict least-recently-created running sequence
                    if let Some(victim) = self.running.pop_back() {
                        self.preempt_by_recompute(victim);
                        
                        // Retry allocation
                        if self.block_engine.can_allocate(seq).is_ok() {
                            self.allocate(seq);
                            seq.set_state(SequenceState::Running);
                            self.running.push_back(seq);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
```

**For rbees:**
- Prevent indefinite waiting
- Evict idle workers to make room for new requests
- Configurable timeout threshold

---

### **Multi-Process Distributed Inference (candle-vllm)**

**Pattern:** Split model across multiple GPUs/processes with NCCL communication.

**Architecture:**
```rust
struct LLMEngine {
    pipelines: HashMap<usize, Pipeline>,  // One per GPU
    num_shards: usize,
    multi_process: bool,
    daemon_manager: Option<DaemonManager>,
}

async fn generate_parallel(
    engine: &Arc<RwLock<LLMEngine>>,
    ranks: Vec<usize>,
) -> Vec<HashMap<String, CompletionResult>> {
    // Parallel execution across GPUs
    let tasks: Vec<_> = ranks.par_iter()
        .map(|rank| {
            Self::generate_once(engine.clone(), *rank)
        })
        .collect();
    tasks
}
```

**For rbees (post-MVP):**
- Distribute large models across multiple GPUs
- Use NCCL for tensor parallelism
- Coordinate via daemon manager

---

## 4. Performance Optimizations

### **Continuous Batching (candle-vllm)**

**Pattern:** Add new requests to batch dynamically without waiting for current batch to finish.

**Implementation:**
```rust
fn schedule(&mut self) -> SchedulerOutput {
    // Add waiting sequences to running batch if space available
    while !self.waiting.is_empty() {
        let seq = self.waiting.front().unwrap();
        
        if self.running.len() >= self.config.max_num_seqs {
            break;
        }
        
        if !self.block_engine.can_allocate(seq) {
            break;
        }
        
        let seq = self.waiting.pop_front().unwrap();
        self.allocate(&seq);
        seq.set_state(SequenceState::Running);
        self.running.push_back(seq);
    }
    
    SchedulerOutput {
        scheduled: self.running.clone(),
        // ...
    }
}
```

**For rbees:**
- Don't wait for batch to finish before adding new requests
- Maximize GPU utilization
- Reduces average latency

---

### **Speculative Decoding (llama.cpp)**

**Pattern:** Use small draft model to predict multiple tokens, verify with main model.

**Configuration:**
```cpp
struct SpeculativeParams {
    std::string model_path;  // Draft model
    int n_draft;             // Number of tokens to predict
    int n_gpu_layers;        // GPU layers for draft model
};
```

**For rbees (post-MVP):**
- Implement speculative decoding for faster generation
- Use TinyLlama as draft model for Llama-7B
- 2-3x speedup for compatible models

---

## 5. Observability Enhancements

### **Detailed Throughput Metrics (candle-vllm)**

**Pattern:** Track prefill and decode throughput separately.

**Metrics:**
```rust
struct CompletionMetrics {
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_time_ms: u64,
    completion_time_ms: u64,
}

impl CompletionMetrics {
    fn prefill_tps(&self) -> f32 {
        (self.prompt_tokens as f32 * 1000.0) 
            / f32::max(self.prompt_time_ms as f32, 1.0)
    }
    
    fn decode_tps(&self) -> f32 {
        (self.completion_tokens as f32 * 1000.0) 
            / f32::max(self.completion_time_ms as f32, 1.0)
    }
    
    fn throughput(&self, batch_size: usize) -> f32 {
        self.prefill_tps() * batch_size as f32
    }
}
```

**Output:**
```
[5 requests] Prefilling: 1024 prompt tokens processed 
  (avg tps 2048.5 tokens/s, throughput 10242.5 tokens/s)
[5 requests] Decoding: 512 completion tokens generated 
  (avg tps 45.2 tokens/s, throughput 226.0 tokens/s)
```

**For rbees:**
- Expose prefill vs decode metrics separately
- Track per-request and aggregate throughput
- Helps identify bottlenecks

---

## 6. Recommended Additions to rbees

### **High Priority**

1. **Preemption by Recompute**
   - Simple to implement
   - Prevents OOM errors
   - Graceful degradation under load

2. **Priority Scheduling**
   - Prevent starvation
   - Fair resource allocation
   - Better UX for waiting users

3. **Chunked Prefill**
   - Handle large prompts (>8K tokens)
   - Prevent head-of-line blocking
   - Better responsiveness

### **Medium Priority**

4. **Prefix Caching with LRU**
   - Optimize repeated prompts
   - Reduce latency for common patterns
   - Evict to CPU under pressure

5. **Waiting Timeout with Eviction**
   - Prevent indefinite waiting
   - Automatic resource rebalancing
   - Configurable fairness policy

6. **Continuous Batching**
   - Maximize GPU utilization
   - Lower average latency
   - Better throughput

### **Low Priority (Post-MVP)**

7. **Block-Level KV Cache (PagedAttention)**
   - Efficient memory management
   - Enables swapping
   - Reduces fragmentation

8. **Multi-GPU Distribution**
   - Scale to larger models
   - Tensor parallelism
   - NCCL communication

9. **Speculative Decoding**
   - 2-3x speedup
   - Requires draft model
   - Complex implementation

---

## Implementation Roadmap

### **Phase 1: Core Scheduling (Week 2)**
- [ ] Implement preemption by recompute
- [ ] Add priority/urgency tracking
- [ ] Implement chunked prefill for large prompts

### **Phase 2: Memory Optimization (Week 3)**
- [ ] Add prefix caching
- [ ] Implement LRU eviction to CPU
- [ ] Add waiting timeout with eviction

### **Phase 3: Batching (Week 3-4)**
- [ ] Implement continuous batching
- [ ] Add detailed throughput metrics
- [ ] Optimize batch size selection

### **Phase 4: Advanced Features (Post-MVP)**
- [ ] PagedAttention with block-level KV cache
- [ ] Multi-GPU tensor parallelism
- [ ] Speculative decoding

---

## References

- **candle-vllm:** Preemption, swapping, chunked prefill, continuous batching
- **mistral.rs:** Priority scheduling, prefix caching, LRU eviction, bucketing
- **Ollama:** Keep-alive, VRAM recovery, scheduler architecture
- **llama.cpp:** Slot-based concurrency, speculative decoding, metrics

All patterns tested in production and proven to scale.
