# model-eviction

**Model eviction policy for VRAM management**

Decides which models to unload when VRAM is constrained, based on usage patterns (LRU, LFU, cost-based).

**Key responsibilities:**
- Track model access patterns (last used, frequency)
- LRU eviction (least recently used)
- LFU eviction (least frequently used)
- Cost-based eviction (unload large models first)
- Pin critical models (prevent eviction)