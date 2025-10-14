# Frontend Development Quickstart

## Choose Your App

From the repository root:

```bash
# Commercial marketing site
pnpm run dev:commercial
# → UI CSS watcher + Commercial app at http://localhost:3000

# User documentation site
pnpm run dev:docs
# → UI CSS watcher + Docs app at http://localhost:3001

# Both sites simultaneously
pnpm run dev:frontend
# → UI CSS watcher + Commercial + Docs

# Just UI components (for Storybook)
pnpm run dev:ui
# → UI CSS watcher only
```

Each command automatically:
- 🎨 **Starts UI CSS watcher** - Rebuilds when you edit components in `@rbee/ui`
- 🚀 **Starts Next.js dev server(s)** - For the selected app(s)

## What You Get

### Automatic Hot Reload
1. Edit `frontend/libs/rbee-ui/src/organisms/HeroSection/HeroSection.tsx`
2. CSS rebuilds automatically (watch the [UI] output)
3. Next.js hot-reloads the page
4. See changes instantly at http://localhost:3000

### Colored Output

**Commercial:**
```
[UI]   ≈ tailwindcss v4.1.14
[UI]   Done in 183ms
[APP]  ▲ Next.js 15.5.4
[APP]  - Local:        http://localhost:3000
```

**Docs:**
```
[UI]   ≈ tailwindcss v4.1.14
[UI]   Done in 183ms
[DOCS] ▲ Next.js 15.5.4
[DOCS] - Local:        http://localhost:3001
```

**Both:**
```
[UI]   ≈ tailwindcss v4.1.14
[UI]   Done in 183ms
[COMM] ▲ Next.js 15.5.4 - http://localhost:3000
[DOCS] ▲ Next.js 15.5.4 - http://localhost:3001
```

## How It Works

The root `package.json` orchestrates both packages:

```json
{
  "scripts": {
    "dev:commercial": "concurrently --names \"UI,APP\" -c \"cyan,green\" \"pnpm --filter @rbee/ui run dev\" \"pnpm --filter @rbee/commercial run dev\""
  }
}
```

### Under the Hood

1. **`@rbee/ui` dev script:**
   ```bash
   tailwindcss -i ./src/tokens/globals.css -o ./dist/index.css --watch
   ```
   Watches all `.tsx` files in `src/`, rebuilds `dist/index.css` on changes

2. **`@rbee/commercial` dev script:**
   ```bash
   next dev
   ```
   Imports `@rbee/ui/styles.css` (which points to `dist/index.css`)

3. **`concurrently`:**
   Runs both in parallel, prefixes output, handles Ctrl+C gracefully

## Other Commands

### Just UI Watcher
```bash
pnpm run dev:ui
```

### Just Commercial App
```bash
pnpm --filter @rbee/commercial run dev
```

### Build for Production
```bash
pnpm run build:commercial
```
Builds UI CSS first, then the app.

### Storybook (UI components)
```bash
pnpm --filter @rbee/ui run storybook
```
Opens at http://localhost:6006

## Troubleshooting

### "Cannot find module '@rbee/ui/styles.css'"
Run the initial build:
```bash
pnpm --filter @rbee/ui run build
```

### Changes not reflecting
1. Check [UI] output - CSS should rebuild
2. Check [APP] output - Next.js should detect changes
3. Hard refresh browser (Ctrl+Shift+R)

### Port 3000 already in use
Kill the existing process:
```bash
pkill -f "next dev"
```

## Architecture

```
llama-orch/
├── package.json              # Root orchestration
├── pnpm-workspace.yaml       # Workspace config
└── frontend/
    ├── libs/
    │   └── rbee-ui/
    │       ├── src/           # Components, tokens
    │       ├── dist/
    │       │   └── index.css  # Built CSS (gitignored)
    │       └── package.json   # dev: watch CSS
    └── bin/
        └── commercial/
            ├── app/
            │   ├── layout.tsx # Imports @rbee/ui/styles.css
            │   └── globals.css
            └── package.json   # dev: next dev
```

## Why This Pattern?

✅ **Automatic:** No manual steps, no separate terminals  
✅ **Fast:** Only rebuilds CSS when components change  
✅ **Standard:** Matches official Turborepo examples  
✅ **Scalable:** Add more apps, they all share the same UI CSS  

## Next Steps

- Read [TURBOREPO_PATTERN.md](frontend/TURBOREPO_PATTERN.md) for deep dive
- Check [Turborepo Tailwind Guide](https://turborepo.com/docs/guides/tools/tailwind)
