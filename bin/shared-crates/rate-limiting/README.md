# rate-limiting

**Request rate limiting middleware to prevent DoS attacks**

Provides configurable rate limiting per IP, per endpoint, and per authenticated user. Protects all services from request flooding.

**Key responsibilities:**
- Per-IP rate limiting (100 req/sec default)
- Per-user rate limiting (for authenticated requests)
- Sliding window or token bucket algorithm
- Return 429 with Retry-After header
- Integration with axum middleware