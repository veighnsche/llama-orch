# Hive CLI Examples (TEAM-186)

## New Hive Commands

### Install Hive

#### Localhost Installation
```bash
# Install hive on localhost
rbee hive install --id localhost --port 8600

# Or with explicit port
rbee hive install --id localhost --port 8700
```

#### Remote SSH Installation
```bash
# Install hive on remote machine via SSH
rbee hive install \
  --id hive-prod-01 \
  --ssh-host 192.168.1.100 \
  --ssh-port 22 \
  --ssh-user admin \
  --port 8600
```

**What happens:**
1. Queen adds hive to catalog (user configuration)
2. Queen builds rbee-hive (locally or via SSH)
3. Queen starts hive process
4. Hive sends heartbeat
5. Queen detects devices (one-time, automatic)

---

### Uninstall Hive

#### Normal Uninstall
```bash
# Uninstall hive (stops process + removes from catalog)
rbee hive uninstall --id localhost
```

#### Catalog-Only Uninstall (Remote Unreachable)
```bash
# If remote machine is unreachable, only remove from catalog
rbee hive uninstall --id hive-prod-01 --catalog-only
```

**Default behavior:** Tries to stop hive process first, then removes from catalog.  
**With `--catalog-only`:** Only removes from catalog (for unreachable remote hives).

---

### Update Hive

#### Update SSH Address
```bash
# Update SSH host for a hive
rbee hive update \
  --id hive-prod-01 \
  --ssh-host 192.168.1.101
```

#### Refresh Device Capabilities
```bash
# Refresh device capabilities from hive
rbee hive update --id localhost --refresh-capabilities
```

#### Update SSH + Refresh
```bash
# Update SSH and refresh capabilities
rbee hive update \
  --id hive-prod-01 \
  --ssh-host 192.168.1.101 \
  --ssh-port 2222 \
  --refresh-capabilities
```

**What happens:**
- Updates catalog with new SSH config
- If `--refresh-capabilities`: Calls hive's device detection API and updates catalog

---

### Start Hive

```bash
# Start localhost hive (defaults to localhost)
rbee hive start localhost

# Or just
rbee hive start

# Start specific hive
rbee hive start hive-prod-01
```

**Note:** Hive must be installed first! If not in catalog → FAIL FAST.

---

### Stop Hive

```bash
# Stop localhost hive
rbee hive stop localhost

# Or just
rbee hive stop

# Stop specific hive
rbee hive stop hive-prod-01
```

---

### List Hives

```bash
# List all hives
rbee hive list
```

---

### Get Hive Details

```bash
# Get localhost hive details
rbee hive get localhost

# Or just
rbee hive get

# Get specific hive
rbee hive get hive-prod-01
```

---

## Complete Workflow Examples

### Example 1: Localhost Development

```bash
# 1. Install hive on localhost
rbee hive install --id localhost

# Output:
# 🐝 Installing hive: localhost
# ✅ Hive 'localhost' added to catalog
# 🔨 Building rbee-hive on localhost...
# ✅ Build complete
# ✅ Hive spawn initiated: http://127.0.0.1:8600
# 🔍 Detecting devices for hive 'localhost'...
# ✅ Device capabilities stored for 'localhost'

# 2. List hives
rbee hive list

# 3. Get hive details
rbee hive get localhost

# 4. Refresh capabilities (if hardware changed)
rbee hive update --id localhost --refresh-capabilities

# 5. Uninstall when done
rbee hive uninstall --id localhost
```

---

### Example 2: Remote Production Hive

```bash
# 1. Install hive on remote machine
rbee hive install \
  --id hive-prod-01 \
  --ssh-host 192.168.1.100 \
  --ssh-port 22 \
  --ssh-user admin \
  --port 8600

# Output:
# 🐝 Installing hive: hive-prod-01
# ✅ Hive 'hive-prod-01' added to catalog
# 🔐 SSH into admin@192.168.1.100
# ✅ Repo cloned/verified
# 🔨 Building rbee-hive remotely...
# ✅ Remote build complete
# ✅ Hive spawn initiated: http://192.168.1.100:8600
# 🔍 Detecting devices for hive 'hive-prod-01'...
# ✅ Device capabilities stored for 'hive-prod-01'

# 2. Update SSH address (if IP changed)
rbee hive update \
  --id hive-prod-01 \
  --ssh-host 192.168.1.101

# 3. Uninstall (if machine is reachable)
rbee hive uninstall --id hive-prod-01

# OR if machine is unreachable
rbee hive uninstall --id hive-prod-01 --catalog-only
```

---

## Key Differences from Old Commands

### Old (DEPRECATED)
```bash
rbee hive create --host localhost --port 8600  # ❌ Removed
rbee hive delete localhost                      # ❌ Removed
```

### New (CURRENT)
```bash
rbee hive install --id localhost --port 8600   # ✅ New
rbee hive uninstall --id localhost             # ✅ New
rbee hive update --id localhost --refresh-capabilities  # ✅ Enhanced
```

---

## Summary

**Install/Uninstall replaces Create/Delete:**
- `install` = Add to catalog + Build + Start + Detect devices
- `uninstall` = Stop + Remove from catalog (or catalog-only)

**Update is enhanced:**
- Can update SSH config
- Can refresh device capabilities

**Start/Stop/List/Get remain the same:**
- But now default to "localhost" for convenience
- Hive must be installed first (FAIL FAST if not)

**All commands use the typed Operation enum for type safety!** 🎯
