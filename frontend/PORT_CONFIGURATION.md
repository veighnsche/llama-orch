# Frontend Port Configuration

## Port Assignments

All frontend development servers now use the 783X range:

| Service | Port | Command |
|---------|------|---------|
| **Storybook** | 7832 | `pnpm --filter @rbee/ui run storybook` |
| **Web UI** | 7833 | `pnpm --filter web-ui run dev` |
| **Commercial** | 7834 | `pnpm --filter @rbee/commercial run dev` |
| **User Docs** | 7835 | `pnpm --filter @rbee/user-docs run dev` |

## Quick Access URLs

- 🎨 **Storybook:** http://localhost:7832
- 🐝 **Web UI:** http://localhost:7833
- 🌐 **Commercial:** http://localhost:7834
- 📚 **User Docs:** http://localhost:7835

## Scripts Updated

### kill-dev-servers.sh
- Monitors ports: `7832, 7833, 7834, 7835`
- Kills Next.js, Vite, and Storybook processes

### clean-reinstall.sh
- Updated help text with new port numbers
- Lists all four dev servers

## Configuration Files

### Storybook (port 7832)
```json
// packages/rbee-ui/package.json
{
  "scripts": {
    "dev": "... storybook dev -p 7832 --no-open",
    "storybook": "storybook dev -p 7832 --no-open"
  }
}
```

### Web UI (port 7833)
```json
// apps/web-ui/package.json
{
  "scripts": {
    "dev": "vite --port 7833",
    "preview": "vite preview --port 7833"
  }
}
```

### Commercial (port 7834)
```json
// apps/commercial/package.json
{
  "scripts": {
    "dev": "next dev -p 7834"
  }
}
```

### User Docs (port 7835)
```json
// apps/user-docs/package.json
{
  "scripts": {
    "dev": "next dev -p 7835"
  }
}
```

## Why These Ports?

- **Sequential:** Easy to remember (7832-7835)
- **No conflicts:** Avoids common ports (3000, 5173, 6006, 8080, etc.)
- **Grouped:** All frontend services in same range
- **Future-proof:** Room for additional services (7836+)

## Starting All Services

```bash
# Start all dev servers (in separate terminals)
pnpm --filter @rbee/ui run storybook    # Terminal 1 - port 7832
pnpm --filter web-ui run dev            # Terminal 2 - port 7833
pnpm --filter @rbee/commercial run dev  # Terminal 3 - port 7834
pnpm --filter @rbee/user-docs run dev   # Terminal 4 - port 7835
```

Or use tmux/screen to run them all:

```bash
# Example tmux script
tmux new-session -d -s frontend
tmux send-keys -t frontend:0 'cd frontend && pnpm --filter @rbee/ui run storybook' C-m
tmux split-window -t frontend:0 -h
tmux send-keys -t frontend:0.1 'cd frontend && pnpm --filter web-ui run dev' C-m
tmux split-window -t frontend:0 -v
tmux send-keys -t frontend:0.2 'cd frontend && pnpm --filter @rbee/commercial run dev' C-m
tmux split-window -t frontend:0.1 -v
tmux send-keys -t frontend:0.3 'cd frontend && pnpm --filter @rbee/user-docs run dev' C-m
tmux attach -t frontend
```

## Killing All Services

```bash
cd frontend
./kill-dev-servers.sh
```

This will:
1. Kill all Next.js dev servers
2. Kill all Vite dev servers
3. Kill all Storybook instances
4. Kill any processes on ports 7832-7835
5. Verify all ports are free
