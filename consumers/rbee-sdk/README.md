# rbee-sdk

**TypeScript/JavaScript SDK for rbee**

> ⚠️ **Status:** Under development by TEAM-286  
> **Version:** 0.0.0 (not yet functional)

---

## Mission

Provide a production-ready TypeScript/JavaScript SDK for building web UIs and Node.js applications that interact with rbee (queen-rbee).

---

## For TEAM-286

**👉 START HERE:** Read `TEAM_286_MISSION.md` for complete instructions.

### Quick Start for Development

1. **Read required materials** (listed in TEAM_286_MISSION.md)
2. **Study rbee-keeper** (`/bin/00_rbee_keeper/`) - this is your reference implementation
3. **Review contracts** (`/bin/97_contracts/`) - these define the API
4. **Test heartbeat example** (`/bin/10_queen_rbee/examples/heartbeat_monitor.html`)
5. **Set up TypeScript project** in `ts/` directory
6. **Start with basic HTTP client**
7. **Add SSE streaming support**
8. **Implement operations one by one**

### Key Files to Study

**Must Read:**
- `/bin/CONTRACT_DEPENDENCY_ANALYSIS.md` - Contract hierarchy
- `/bin/97_contracts/operations-contract/src/lib.rs` - All operations
- `/bin/00_rbee_keeper/src/job_client.rs` - Reference implementation
- `/bin/TEAM_285_HEARTBEAT_MONITOR_READY.md` - Heartbeat system
- `/bin/ADDING_NEW_OPERATIONS.md` - API documentation

**Reference Code:**
- `/bin/99_shared_crates/job-client/src/lib.rs` - Job submission pattern
- `/bin/10_queen_rbee/src/http/heartbeat_stream.rs` - SSE streaming
- `/bin/10_queen_rbee/examples/heartbeat_monitor.html` - Working example

---

## Planned API (Not Yet Implemented)

### Installation (Future)

```bash
npm install @rbee/sdk
```

### Basic Usage (Future)

```typescript
import { RbeeClient } from '@rbee/sdk';

const client = new RbeeClient('http://localhost:8500');

// Run inference
const result = await client.infer({
  model: 'llama-3-8b',
  prompt: 'Hello, world!',
  stream: true,
});

// Stream tokens
for await (const token of result.stream()) {
  console.log(token);
}

// Monitor heartbeats
const monitor = client.heartbeats.stream();
monitor.on('update', (data) => {
  console.log('Workers:', data.workers_online);
});
```

---

## Current Status

**Completed:**
- ✅ Mission document created (TEAM_286_MISSION.md)
- ✅ Old code removed
- ✅ Clean slate for TEAM-286

**TODO (TEAM-286):**
- [ ] Set up TypeScript project structure
- [ ] Generate types from contracts
- [ ] Implement HTTP client
- [ ] Implement SSE streaming
- [ ] Implement operations
- [ ] Add React hooks
- [ ] Write examples
- [ ] Write tests
- [ ] Publish to npm

---

## Architecture

### Planned Structure

```
rbee-sdk/
├── ts/                    - TypeScript source
│   ├── client.ts         - Main RbeeClient
│   ├── operations.ts     - Operation builders
│   ├── types.ts          - TypeScript types
│   ├── sse.ts            - SSE utilities
│   ├── heartbeat.ts      - Heartbeat monitor
│   └── index.ts          - Public API
├── examples/             - Usage examples
├── tests/                - Tests
└── TEAM_286_MISSION.md   - Complete instructions
```

---

## Development

### Prerequisites

- Node.js 18+
- TypeScript 5+
- Understanding of rbee architecture

### Setup

```bash
cd consumers/rbee-sdk
npm install
npm run build
npm test
```

---

## License

GPL-3.0-or-later

---

## For Questions

See `TEAM_286_MISSION.md` for:
- Complete architecture guide
- Required reading list
- Implementation phases
- Design decisions
- Example code
- Testing strategy
- Success criteria

**Good luck, TEAM-286!** 🚀
