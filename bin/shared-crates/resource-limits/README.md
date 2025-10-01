# resource-limits

**Enforces bounded data structures and resource limits**

Provides bounded collections (HashMap, Vec, Queue) that automatically evict old entries to prevent memory exhaustion.

**Key responsibilities:**
- BoundedHashMap with max size + LRU eviction
- BoundedVec with ring buffer behavior
- BoundedQueue with capacity limits
- Automatic cleanup of expired entries
- Memory usage monitoring