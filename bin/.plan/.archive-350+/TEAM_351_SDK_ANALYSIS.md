# TEAM-351: SDK & React Package Analysis

**Date:** Oct 29, 2025  
**Question:** Do SDK and React packages have duplicated code that should be extracted?

---

## Analysis

### SDK Packages

#### Queen SDK (@rbee/queen-rbee-sdk)
- **Type:** WASM (Rust compiled to WebAssembly)
- **Size:** ~7 Rust files
- **Unique:** ✅ Queen-specific (WASM bindings, Rhai IDE, Heartbeat)
- **Hardcoded URL:** `http://localhost:7833`

#### Hive SDK (@rbee/rbee-hive-sdk)
- **Type:** TypeScript HTTP client
- **Size:** 41 lines
- **Functions:** `listModels()`, `downloadModel()`, `listWorkers()`, `spawnWorker()`
- **Hardcoded URL:** `http://localhost:7835`

#### Worker SDK (@rbee/llm-worker-sdk)
- **Type:** TypeScript HTTP client
- **Size:** 27 lines
- **Functions:** `infer()`, `getHealth()`
- **Hardcoded URL:** `http://localhost:8080`

---

## Duplication Found

### 🔴 Critical: Hardcoded URLs

**All 3 SDKs hardcode their base URLs:**

```typescript
// Hive SDK
fetch('http://localhost:7835/api/models')  // ❌ Hardcoded

// Worker SDK
fetch('http://localhost:8080/api/infer')   // ❌ Hardcoded

// Queen SDK (TypeScript wrapper)
// Base URL: http://localhost:7833           // ❌ Hardcoded
```

**Should use:** `@rbee/shared-config`
```typescript
import { getServiceUrl, PORTS } from '@rbee/shared-config'

const baseUrl = getServiceUrl('hive', 'prod')  // ✅ Dynamic
```

---

### 🟡 Potential: HTTP Client Pattern

**Hive and Worker SDKs both:**
- Use `fetch()` API
- Handle JSON responses
- Have similar error handling patterns
- Could share a base HTTP client

**Potential shared package:** `@rbee/http-client`
```typescript
// Shared HTTP client with:
- Base URL configuration
- Error handling
- Type-safe responses
- Retry logic
- Timeout handling
```

---

### 🟢 React Hooks Pattern

**All 3 React packages follow same pattern:**

```typescript
// Hive React
useModels()   // useState + useEffect + fetch
useWorkers()  // useState + useEffect + fetch + polling

// Worker React
useInference() // useState + async function

// Queen React
useRbeeSDK()     // useState + useEffect + WASM loading
useHeartbeat()   // useState + useEffect + polling
useRhaiScripts() // useState + async functions
```

**Observation:** These are service-specific hooks wrapping service-specific SDKs.
**Verdict:** ✅ **NO extraction needed** - Each is unique to its service.

---

## Recommendations

### ✅ Extract: Base HTTP Client

**Create:** `@rbee/http-client` or add to `@rbee/shared-config`

```typescript
// frontend/packages/shared-config/src/http-client.ts

export interface HttpClientConfig {
  baseUrl: string
  timeout?: number
  headers?: Record<string, string>
}

export class HttpClient {
  constructor(private config: HttpClientConfig) {}
  
  async get<T>(path: string): Promise<T> {
    const response = await fetch(`${this.config.baseUrl}${path}`, {
      signal: AbortSignal.timeout(this.config.timeout || 5000),
      headers: this.config.headers,
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return await response.json()
  }
  
  async post<T>(path: string, body: any): Promise<T> {
    const response = await fetch(`${this.config.baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...this.config.headers },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.config.timeout || 5000),
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return await response.json()
  }
}
```

**Usage in Hive SDK:**
```typescript
import { HttpClient } from '@rbee/shared-config'
import { getServiceUrl } from '@rbee/shared-config'

const client = new HttpClient({
  baseUrl: getServiceUrl('hive', 'prod'),
})

export async function listModels() {
  return await client.get<Model[]>('/api/models')
}
```

---

### ✅ Fix: Hardcoded URLs

**All SDKs should use `@rbee/shared-config`:**

```typescript
// Before (Hive SDK)
fetch('http://localhost:7835/api/models')

// After
import { getServiceUrl } from '@rbee/shared-config'
const baseUrl = getServiceUrl('hive', 'prod')
fetch(`${baseUrl}/api/models`)
```

---

### ❌ Don't Extract: React Hooks

**Reason:** Each React package is service-specific:
- Hive hooks: Models, Workers (Hive-specific)
- Worker hooks: Inference (Worker-specific)
- Queen hooks: WASM SDK, Heartbeat, Rhai (Queen-specific)

**Verdict:** Keep as-is. These are thin wrappers around service-specific SDKs.

---

## Summary

### Duplicated Code Found

| Type | Location | Should Extract? | Priority |
|------|----------|-----------------|----------|
| **Hardcoded URLs** | All 3 SDKs | ✅ Yes → Use `@rbee/shared-config` | 🔴 High |
| **HTTP Client Pattern** | Hive + Worker SDKs | ✅ Yes → Create `HttpClient` | 🟡 Medium |
| **React Hooks** | All 3 React packages | ❌ No (service-specific) | N/A |

---

## Action Items

### Phase 1 (Current - TEAM-351)
- ✅ Created `@rbee/shared-config` (ports)
- ✅ Created `@rbee/narration-client`
- ✅ Created `@rbee/iframe-bridge`
- ✅ Created `@rbee/dev-utils`

### Phase 2 (Next - TEAM-352+)
1. **Add HttpClient to `@rbee/shared-config`**
   - Base HTTP client class
   - Error handling
   - Timeout support
   - Type-safe responses

2. **Update Hive SDK** to use:
   - `getServiceUrl('hive', 'prod')` for base URL
   - `HttpClient` for fetch calls

3. **Update Worker SDK** to use:
   - `getServiceUrl('worker', 'prod')` for base URL
   - `HttpClient` for fetch calls

4. **Update Queen SDK TypeScript wrapper** to use:
   - `getServiceUrl('queen', 'prod')` for base URL

---

## Estimated Impact

### Lines of Code Saved
- **Hardcoded URLs:** ~10 lines per SDK = 30 lines
- **HTTP Client extraction:** ~50 lines (shared implementation)
- **Total:** ~80 lines saved

### Benefits
- ✅ Single source of truth for URLs
- ✅ Consistent error handling
- ✅ Easier testing (mock base URL)
- ✅ Type-safe HTTP calls
- ✅ Timeout handling
- ✅ Retry logic (future)

---

## Conclusion

**Answer:** YES, there is duplicated code in the SDK packages!

**What to extract:**
1. ✅ **Hardcoded URLs** → Use `@rbee/shared-config`
2. ✅ **HTTP Client pattern** → Create shared `HttpClient`

**What NOT to extract:**
- ❌ React hooks (service-specific, thin wrappers)

**Next Steps:**
1. Add `HttpClient` to `@rbee/shared-config`
2. Update all 3 SDKs to use shared config and client
3. Remove hardcoded URLs

---

**TEAM-351: SDK analysis complete. Extraction needed for HTTP client and URLs!** 🎯
