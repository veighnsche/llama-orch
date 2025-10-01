# streaming

**Server-Sent Events (SSE) streaming coordination**

Manages SSE connections for streaming inference results from workers back to clients. Handles connection lifecycle, backpressure, and graceful degradation.

**Key responsibilities:**
- Maintain SSE connection pools per job
- Stream tokens from worker → orchestratord → client
- Handle disconnects and reconnects
- Apply backpressure when client is slow