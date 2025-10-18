# Automatic Development Workflow

## The Problem You Had

**Before:** Manual, error-prone workflow
```bash
# Terminal 1
cd frontend/libs/rbee-ui
pnpm run dev:styles

# Terminal 2  
cd frontend/bin/commercial
pnpm run dev

# 😫 Two terminals, manual coordination, easy to forget
```

## The Solution (Idiomatic Turborepo)

**Now:** One command, automatic coordination
```bash
# From repo root
pnpm run dev:commercial

# ✨ Everything just works
```

## Visual Flow

```
┌─────────────────────────────────────────────────────────────┐
│  pnpm run dev:commercial                                    │
│  (from repo root)                                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─── concurrently orchestrates ───┐
                 │                                  │
        ┌────────▼────────┐              ┌─────────▼──────────┐
        │  @rbee/ui       │              │  @rbee/commercial  │
        │  pnpm run dev   │              │  pnpm run dev      │
        └────────┬────────┘              └─────────┬──────────┘
                 │                                  │
                 │                                  │
        ┌────────▼────────────────────┐   ┌────────▼──────────────────┐
        │ tailwindcss --watch         │   │ next dev                  │
        │                             │   │                           │
        │ Watches:                    │   │ Imports:                  │
        │ • src/**/*.tsx              │   │ • @rbee/ui/styles.css     │
        │ • src/**/*.ts               │   │   ↓                       │
        │                             │   │ • dist/index.css          │
        │ Outputs:                    │   │                           │
        │ • dist/index.css ───────────┼───┤ Serves:                   │
        │   (266KB, 9823 lines)       │   │ • http://localhost:3000   │
        └─────────────────────────────┘   └───────────────────────────┘
```

## What Happens When You Edit a Component

```
1. You edit: frontend/libs/rbee-ui/src/atoms/Button/Button.tsx
   │
   ├─ Add new Tailwind class: className="bg-purple-500"
   │
   ▼
2. [UI] Tailwind watcher detects change
   │
   ├─ Scans Button.tsx for classes
   ├─ Generates CSS for bg-purple-500
   ├─ Rebuilds dist/index.css
   │
   ▼
3. [UI] Output: "Done in 183ms"
   │
   ▼
4. [APP] Next.js detects dist/index.css changed
   │
   ├─ Hot Module Replacement (HMR)
   ├─ Injects new CSS
   │
   ▼
5. Browser updates automatically
   │
   └─ Purple button appears instantly! 🎉
```

## Terminal Output Example

```bash
$ pnpm run dev:commercial

[UI]  ≈ tailwindcss v4.1.14
[UI]  
[UI]  Done in 183ms
[APP] ▲ Next.js 15.5.4
[APP] - Local:        http://localhost:3000
[APP] - Environments: .env
[APP] 
[APP] ✓ Starting...
[APP] ✓ Ready in 2.3s
[UI]  
[UI]  Rebuilding...
[UI]  Done in 89ms
[APP] ⚡ Updated 1 module
```

## Key Benefits

### 1. **Zero Manual Steps**
- No "did I start the CSS watcher?"
- No switching terminals
- No forgetting to rebuild

### 2. **Instant Feedback**
```
Edit component → [UI] rebuilds → [APP] hot-reloads → See changes
     ↑                                                      │
     └──────────────── ~300ms total ──────────────────────┘
```

### 3. **Clear Visibility**
- `[UI]` prefix = CSS build output
- `[APP]` prefix = Next.js output
- Color-coded (cyan for UI, green for APP)

### 4. **Graceful Shutdown**
- Ctrl+C stops both processes
- No orphaned watchers
- Clean exit

## Comparison to Turborepo Examples

This matches the official pattern from:
- [vercel/turborepo/examples/with-tailwind](https://github.com/vercel/turborepo/tree/main/examples/with-tailwind)

**Their approach:**
```json
{
  "scripts": {
    "dev": "turbo run dev"
  }
}
```

**Our approach (without turbo CLI):**
```json
{
  "scripts": {
    "dev:commercial": "concurrently ... pnpm --filter @rbee/ui run dev ... pnpm --filter @rbee/commercial run dev"
  }
}
```

Same result, using pnpm's built-in workspace filtering instead of turbo CLI.

## Advanced: Adding More Apps

Want to add another app that uses `@rbee/ui`?

```json
{
  "scripts": {
    "dev:docs": "concurrently \"pnpm --filter @rbee/ui run dev\" \"pnpm --filter @rbee/user-docs run dev\"",
    "dev:all": "concurrently \"pnpm --filter @rbee/ui run dev\" \"pnpm --filter @rbee/commercial run dev\" \"pnpm --filter @rbee/user-docs run dev\""
  }
}
```

The UI CSS watcher runs once, all apps consume the same `dist/index.css`.

## Files That Make This Work

1. **Root `package.json`:**
   ```json
   {
     "scripts": {
       "dev:commercial": "concurrently --names \"UI,APP\" -c \"cyan,green\" \"pnpm --filter @rbee/ui run dev\" \"pnpm --filter @rbee/commercial run dev\""
     },
     "devDependencies": {
       "concurrently": "^9.1.2"
     }
   }
   ```

2. **`@rbee/ui` package.json:**
   ```json
   {
     "scripts": {
       "dev": "tailwindcss -i ./src/tokens/globals.css -o ./dist/index.css --watch"
     }
   }
   ```

3. **`@rbee/commercial` package.json:**
   ```json
   {
     "scripts": {
       "dev": "next dev"
     }
   }
   ```

4. **`@rbee/commercial` layout.tsx:**
   ```tsx
   import '@rbee/ui/styles.css'  // Pre-built CSS
   import './globals.css'        // App-specific
   ```

## Summary

✅ **One command:** `pnpm run dev:commercial`  
✅ **Automatic:** CSS rebuilds on component changes  
✅ **Fast:** Only rebuilds what changed  
✅ **Standard:** Matches Turborepo pattern  
✅ **No hacks:** No shell scripts, no manual steps  

**This is the idiomatic way.** 🎯
