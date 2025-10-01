# circuit-breaker

**Circuit breaker pattern for service resilience**

Stops calling a failing service temporarily, prevents cascading failures, and allows service to recover.

**Key responsibilities:**
- Track failure rate per downstream service
- Open circuit after N consecutive failures
- Half-open state (test if service recovered)
- Close circuit when service healthy again
- Fail fast (return error immediately when circuit open)