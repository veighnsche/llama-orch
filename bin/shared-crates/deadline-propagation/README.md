# deadline-propagation

**Request deadline propagation and enforcement**

Propagates client-specified deadlines through orchestrator → pool-manager → worker, aborts work if deadline exceeded.

**Key responsibilities:**
- Parse client deadline (X-Deadline header or deadline_ms field)
- Calculate remaining time at each hop
- Abort work if deadline already exceeded
- Return 504 Gateway Timeout if deadline missed
- Cancel downstream requests when deadline hit