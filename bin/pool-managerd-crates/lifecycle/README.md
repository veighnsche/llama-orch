# lifecycle

**Worker process lifecycle management**

Spawns, monitors, and manages worker-orcd processes. Handles graceful shutdown, crash recovery, and health monitoring.

**Key responsibilities:**
- Spawn worker-orcd processes with correct config
- Monitor worker health (heartbeats)
- Restart crashed workers
- Graceful shutdown on drain
- Pass callback URLs to workers