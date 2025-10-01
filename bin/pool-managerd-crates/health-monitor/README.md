# health-monitor

**Worker and pool health monitoring**

Continuously monitors worker health (heartbeats, memory, GPU), detects failures, and triggers recovery actions.

**Key responsibilities:**
- Monitor worker heartbeats (detect crashes)
- Check GPU health (CUDA errors, temperature)
- Monitor RAM/VRAM usage (detect leaks)
- Trigger worker restart on failure
- Mark pools as unhealthy/draining