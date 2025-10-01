# backpressure

**Backpressure and admission control policies**

Manages queue overflow scenarios, rejects tasks when overloaded, and returns proper HTTP 429 with Retry-After headers.

**Key responsibilities:**
- Detect queue full conditions
- Calculate Retry-After time based on queue depth
- Circuit breaking (stop accepting if downstream failing)
- Load shedding (drop low-priority tasks first)
- Admission policies (reject-new, drop-LRU, fail-fast)