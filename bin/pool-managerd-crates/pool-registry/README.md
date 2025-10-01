# pool-registry

**Tracks pool state and readiness on this node**

Maintains the registry of pools (loaded models) on this node. Tracks which pools are ready, their replica counts, VRAM usage, and availability.

**Key responsibilities:**
- Register pools when workers become ready
- Track pool state (ready, draining, failed)
- Query available capacity per pool
- Provide pool snapshots for orchestratord