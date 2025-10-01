# retry-policy

**Standardized retry policies across services**

Provides configurable retry logic with exponential backoff, jitter, and circuit breaking for all inter-service HTTP calls.

**Key responsibilities:**
- Exponential backoff (100ms, 200ms, 400ms, ...)
- Jitter to prevent thundering herd
- Retry on transient errors (503, connection timeout)
- Don't retry on permanent errors (400, 401, 404)
- Circuit breaker (stop retrying if service down)