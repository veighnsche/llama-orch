# vram-residency

**Enforces VRAM-only inference and prevents RAM fallback**

Ensures model weights never leave GPU memory during inference. Provides sealed handles to VRAM-resident model shards and validates all operations happen in GPU memory.

**Key responsibilities:**
- Pin model weights in VRAM at load time
- Prevent accidental RAM paging/swapping
- Provide `ModelShardHandle` abstraction (sealed VRAM shard)
- CUDA memory monitoring and attestation