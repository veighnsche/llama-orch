# task-cancellation

**Task cancellation and cleanup**

Handles client-initiated cancellations, deadline timeouts, and cascading cleanup across orchestrator → pool-manager → worker.

**Key responsibilities:**
- Client cancellation (DELETE /v2/jobs/{id})
- Deadline timeout enforcement
- Propagate cancellation to worker
- Clean up partial results
- Return 499 (Client Closed Request)