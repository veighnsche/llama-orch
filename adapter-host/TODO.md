# TODO — adapter-host

- Implement registry lifecycle (bind/rebind on reload/drain).
- Add facade submit/cancel/health/props with retries/backoff and circuit breaker.
- Integrate capability snapshot cache for `/v1/capabilities`.
- Emit narration logs (submit/cancel, retries, breaker trips) with taxonomy.
- Unit tests: cancel routing, retry jitter, breaker transitions.
