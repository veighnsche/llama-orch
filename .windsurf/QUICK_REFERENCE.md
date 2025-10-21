# Language Server Quick Reference

## 🚨 Emergency Commands

```bash
# Kill runaway language servers
./.windsurf/language-server-diagnostics.sh kill

# Check current memory usage (threshold: 6GB)
./.windsurf/language-server-diagnostics.sh monitor

# Full diagnostic report
./.windsurf/language-server-diagnostics.sh report
```

## 📊 Quick Health Check

```bash
# Memory usage (should be < 6GB total)
ps aux | grep -E "(language_server|rust-analyzer)" | grep -v grep | awk '{sum+=$6} END {print sum/1024 " MB"}'

# Process count (should be 1-2)
pgrep -f language_server | wc -l

# Check for OOM events
dmesg | grep -i oom | tail -5
```

## ✅ Success Criteria

| Metric | Target | Command |
|--------|--------|---------|
| Memory usage | < 6GB | `.windsurf/language-server-diagnostics.sh monitor` |
| OOM events | 0 | `dmesg \| grep -i oom` |
| Stable runtime | > 1 hour | Watch process in `htop` |
| Indexing time | < 3 min | Time to first autocomplete |

## 🔧 Common Fixes

### Still High Memory?

1. **Full restart:**
   ```bash
   ./.windsurf/language-server-diagnostics.sh kill
   # Close and reopen Windsurf
   ```

2. **Clear cache:**
   ```bash
   rm -rf ~/.config/Windsurf/Cache/*
   # Restart Windsurf
   ```

3. **Verify exclusions active:**
   ```bash
   # Should NOT see target/ or node_modules/
   lsof -p $(pgrep -f language_server) | grep -E "(target|node_modules)" | wc -l
   # Expected: 0
   ```

### Need Proc-Macros Back?

**Edit `.vscode/settings.json`:**
```json
"rust-analyzer.procMacro.enable": true,  // Re-enable
"rust-analyzer.procMacro.attributes.enable": true
```

**Cost:** +2-3GB memory, slower indexing

### Working on Single Crate?

**Scope rust-analyzer to one crate:**
```json
"rust-analyzer.linkedProjects": [
  "./bin/10_queen_rbee/Cargo.toml"
]
```

## 📁 What's Excluded

| Directory | Size | Why Excluded |
|-----------|------|--------------|
| `target/` | 85GB | Rust build artifacts |
| `.venv-testing/` | 6.7GB | Python virtual env |
| `node_modules/` | 2.3GB | pnpm dependencies |
| `.git/` | ~500MB | Version control |
| `.cache/`, `.turbo/` | Variable | Build caches |

**Total saved:** ~94GB not being indexed

## 🔍 Monitoring Commands

```bash
# Real-time memory tracking
watch -n 5 './.windsurf/language-server-diagnostics.sh monitor'

# Process tree with memory
pstree -p $(pgrep -f language_server)

# Open file handles (should be < 1000)
lsof -p $(pgrep -f language_server) | wc -l

# Memory breakdown
cat /proc/$(pgrep -f language_server)/smaps_rollup
```

## 📖 Full Documentation

See: `.windsurf/LANGUAGE_SERVER_MEMORY_FIX.md`

## 🆘 If Nothing Works

```bash
# Nuclear option: Limit to single crate
echo '{"rust-analyzer.linkedProjects": ["./bin/10_queen_rbee/Cargo.toml"]}' > .vscode/settings.local.json

# Restart with clean state
pkill -9 -f language_server
rm -rf ~/.config/Windsurf/Cache/*
# Reopen Windsurf
```

---

**Last Updated:** 2025-10-21  
**Status:** ✅ Fix deployed, verified stable
