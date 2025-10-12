# Workspace Integration

**Created by:** TEAM-FE-011 (aka TEAM-DX-000)

## pnpm Scripts Integration

```json
{
  "scripts": {
    "dx:css": "dx css --class-exists cursor-pointer http://localhost:3000",
    "dx:verify": "dx snapshot --compare --name homepage http://localhost:3000",
    "dx:update": "dx snapshot --update --name homepage http://localhost:3000"
  }
}
```

## CI/CD Integration

```yaml
# .github/workflows/frontend.yml
- name: Install dx tool
  run: cargo install --path frontend/.dx-tool

- name: Start dev server
  run: pnpm dev &

- name: Verify CSS
  run: pnpm dx:verify
```

## Configuration

`.dxrc.json` in project root for defaults.

---

**Next:** See `06_IMPLEMENTATION_ROADMAP.md` for development timeline.
