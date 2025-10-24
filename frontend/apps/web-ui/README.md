# rbee Web UI

**React-based dashboard for managing rbee infrastructure**

`frontend/apps/rbee-web-ui` — Web interface for queen, hives, workers, and models.

---

## Purpose

Provide a visual dashboard for managing rbee infrastructure, using the TypeScript SDK (`@rbee/sdk`) to communicate with queen-rbee.

**Use case:** Open `http://localhost:3002` to manage your rbee infrastructure visually.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Browser                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  rbee-web-ui (React/Next.js)                           │ │
│  │  ├─ Dashboard views                                    │ │
│  │  ├─ Hive management                                    │ │
│  │  ├─ Worker management                                  │ │
│  │  ├─ Model management                                   │ │
│  │  └─ Inference playground                               │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                         ↓ HTTP/SSE
┌──────────────────────────────────────────────────────────────┐
│  @rbee/sdk (TypeScript) - WASM compiled from Rust            │
│  ├─ RbeeClient.hiveList()                                    │
│  ├─ RbeeClient.workerSpawn()                                 │
│  ├─ RbeeClient.modelList()                                   │
│  └─ RbeeClient.infer() (streaming)                           │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│  queen-rbee HTTP API (http://localhost:8500)                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Status

**STUB APP** - Design phase only.

- ✅ App structure created (Next.js 15)
- ✅ Basic layout and placeholder cards
- ✅ Configured to use `@rbee/sdk`
- ✅ Configured to use `@rbee/ui` components
- ✅ Turborepo integration
- ⚠️ SDK integration pending (SDK not implemented yet)
- ⚠️ Real-time updates pending
- ⚠️ Streaming inference UI pending

---

## Features (Planned)

### 1. Dashboard Overview

**Real-time status cards:**
- Queen connection status
- Number of hives (online/offline)
- Number of workers (active/idle)
- Number of models (available/downloading)

### 2. Hive Management

- **List hives** with status indicators
- **Install new hive** (SSH connection form)
- **Start/stop hive** with live logs
- **Uninstall hive**
- **View hive details** (capabilities, resources)

### 3. Worker Management

- **List workers** across all hives
- **Spawn worker** (select model and device)
- **View worker details** (model, device, status)
- **Stop worker**
- **Live logs** from worker spawn/shutdown

### 4. Model Management

- **List models** with metadata
- **Download model** from HuggingFace
- **View download progress** (real-time)
- **Delete model**
- **Model details** (size, format, metadata)

### 5. Inference Playground

- **Prompt input** with syntax highlighting
- **Model selection** dropdown
- **Parameter controls** (temperature, top_p, max_tokens)
- **Streaming output** (token-by-token)
- **Chat history** (multi-turn conversations)
- **Export conversation** (JSON, Markdown)

### 6. System Monitoring

- **GPU utilization** graphs
- **Memory usage** graphs
- **Token throughput** metrics
- **Queue depth** visualization
- **Error logs** viewer

---

## Tech Stack

### Framework
- **Next.js 15** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type safety

### UI Components
- **@rbee/ui** - Shared component library (shadcn/ui based)
- **Radix UI** - Headless component primitives
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

### Data
- **@rbee/sdk** - TypeScript SDK (WASM from Rust)
- **SSE** - Server-Sent Events for streaming

### Build
- **Turborepo** - Monorepo build system
- **pnpm** - Package manager

---

## Development

### Install Dependencies

```bash
# From monorepo root
pnpm install
```

### Run Dev Server

```bash
# From monorepo root
cd frontend/apps/rbee-web-ui
pnpm dev
```

**Opens at:** `http://localhost:3002`

### Build

```bash
pnpm build
```

### Lint

```bash
pnpm lint
```

---

## Project Structure

```
rbee-web-ui/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Dashboard page
│   │   ├── hives/
│   │   │   └── page.tsx        # Hive management
│   │   ├── workers/
│   │   │   └── page.tsx        # Worker management
│   │   ├── models/
│   │   │   └── page.tsx        # Model management
│   │   └── inference/
│   │       └── page.tsx        # Inference playground
│   ├── components/
│   │   ├── dashboard/
│   │   ├── hive-list.tsx
│   │   ├── worker-list.tsx
│   │   ├── model-list.tsx
│   │   └── inference-form.tsx
│   ├── hooks/
│   │   ├── use-rbee-client.ts  # SDK client hook
│   │   ├── use-hives.ts        # Hive data hook
│   │   ├── use-workers.ts      # Worker data hook
│   │   └── use-models.ts       # Model data hook
│   └── lib/
│       └── rbee.ts             # SDK configuration
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.ts
└── README.md
```

---

## Implementation Roadmap

### Phase 1: SDK Integration (8-12 hours)

**Goal:** Connect to rbee SDK and display real data

**Prerequisites:**
- ⚠️ `@rbee/sdk` must be implemented first (22-32 hours)

**Tasks:**
- [ ] Create `use-rbee-client` hook
- [ ] Initialize SDK client with queen URL
- [ ] Handle connection states (connecting, connected, error)
- [ ] Add error boundaries
- [ ] Add loading states

**Example:**
```typescript
// src/hooks/use-rbee-client.ts
import { RbeeClient } from '@rbee/sdk';

export function useRbeeClient() {
  const [client, setClient] = useState<RbeeClient | null>(null);
  const [status, setStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');

  useEffect(() => {
    const queenUrl = process.env.NEXT_PUBLIC_QUEEN_URL || 'http://localhost:8500';
    const client = new RbeeClient(queenUrl);
    setClient(client);
    setStatus('connected');
  }, []);

  return { client, status };
}
```

### Phase 2: Dashboard Views (12-16 hours)

**Goal:** Implement all main views

**Tasks:**
- [ ] Dashboard overview page
- [ ] Hive management page
- [ ] Worker management page
- [ ] Model management page
- [ ] Inference playground page

### Phase 3: Real-Time Updates (8-12 hours)

**Goal:** Add live data updates

**Tasks:**
- [ ] Polling for status updates
- [ ] SSE for streaming events
- [ ] WebSocket for live metrics (future)
- [ ] Optimistic UI updates

### Phase 4: Advanced Features (16-20 hours)

**Goal:** Add advanced functionality

**Tasks:**
- [ ] System monitoring graphs
- [ ] Log viewer with filtering
- [ ] Configuration editor
- [ ] Export/import functionality

**Total Effort:** 44-60 hours (after SDK is ready)

---

## SDK Usage Examples

### Hive Management

```typescript
import { useRbeeClient } from '@/hooks/use-rbee-client';

function HiveList() {
  const { client } = useRbeeClient();
  const [hives, setHives] = useState([]);

  useEffect(() => {
    if (client) {
      client.hiveList().then(setHives);
    }
  }, [client]);

  const handleStart = async (alias: string) => {
    const stream = await client.hiveStart(alias);
    for await (const event of stream) {
      console.log(event); // Show in UI
    }
  };

  return (
    <div>
      {hives.map(hive => (
        <div key={hive.alias}>
          <h3>{hive.alias}</h3>
          <button onClick={() => handleStart(hive.alias)}>Start</button>
        </div>
      ))}
    </div>
  );
}
```

### Streaming Inference

```typescript
import { useRbeeClient } from '@/hooks/use-rbee-client';

function InferencePlayground() {
  const { client } = useRbeeClient();
  const [response, setResponse] = useState('');

  const handleInfer = async (prompt: string, model: string) => {
    setResponse('');
    const stream = await client.infer(prompt, model);
    
    for await (const token of stream) {
      setResponse(prev => prev + token);
    }
  };

  return (
    <div>
      <textarea placeholder="Enter prompt..." />
      <button onClick={() => handleInfer(prompt, model)}>Generate</button>
      <pre>{response}</pre>
    </div>
  );
}
```

---

## Configuration

### Environment Variables

```bash
# .env.local
NEXT_PUBLIC_QUEEN_URL=http://localhost:8500
```

### Queen URL Detection

**Priority:**
1. `NEXT_PUBLIC_QUEEN_URL` environment variable
2. `http://localhost:8500` (default)
3. User input (connection form)

---

## Design Principles

### 1. Simplicity First

- Clean, minimalist UI
- Focus on essential features
- No unnecessary complexity

### 2. Real-Time Feel

- Live status updates
- Streaming logs
- Optimistic UI updates

### 3. Developer-Friendly

- Clear error messages
- Detailed logs
- JSON export for debugging

### 4. Responsive

- Works on desktop and mobile
- Adaptive layouts
- Touch-friendly controls

---

## Future Enhancements

### 1. Advanced Monitoring

- GPU utilization graphs (time series)
- Memory usage tracking
- Token throughput metrics
- Queue visualization

### 2. Multi-Queen Support

- Connect to multiple queens
- Switch between queens
- Aggregate views

### 3. User Authentication

- Login/logout
- Role-based access
- API key management

### 4. Collaboration Features

- Share configurations
- Team workspaces
- Audit logs

---

## Dependencies

```json
{
  "dependencies": {
    "@rbee/sdk": "../../consumers/rbee-sdk",
    "@rbee/ui": "workspace:*",
    "@repo/tailwind-config": "workspace:*",
    "next": "15.5.5",
    "react": "19.2.0",
    "lucide-react": "^0.545.0"
  }
}
```

---

## References

**Related Documentation:**
- `.arch/SDK_ARCHITECTURE.md` - SDK design
- `consumers/rbee-sdk/README.md` - SDK documentation
- `packages/rbee-ui/README.md` - UI component library

**Similar Projects:**
- [Ollama Web UI](https://github.com/ollama-webui/ollama-webui)
- [LocalAI Web UI](https://github.com/mudler/LocalAI)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)

---

## Next Steps

1. **Wait for SDK implementation** - rbee-sdk must be functional first
2. **Implement Phase 1** - SDK integration (8-12 hours)
3. **Implement Phase 2** - Dashboard views (12-16 hours)
4. **Implement Phase 3** - Real-time updates (8-12 hours)
5. **Implement Phase 4** - Advanced features (16-20 hours)

**Total:** 44-60 hours (after SDK is ready)

---

**Created by:** TEAM-266  
**Status:** Stub app - ready for implementation  
**Depends on:** `@rbee/sdk` (22-32 hours to implement)
