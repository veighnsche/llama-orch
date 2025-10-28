# TEAM-292: Service Discovery & Default Ports

**Date:** Oct 25, 2025  
**Status:** ✅ COMPLETE

## Problem

Hives needed to be manually configured with the queen's URL. This required:
1. User to know queen's address
2. Manual configuration when starting hives
3. No automatic discovery mechanism

## Solution

Implemented **service discovery** where:
1. **Queen exposes `/v1/info` endpoint** with its address
2. **Bee keeper discovers queen's URL** automatically
3. **Bee keeper passes queen URL to hive** when starting it
4. **Hive automatically connects** to the correct queen

## Default Ports

- **Queen:** Always `8500` (default)
- **Hive:** Always `9000` (default)
- **Hive connects to:** `http://localhost:8500` (default)

## Architecture Flow

```
rbee-keeper
    ↓ 1. ensure_queen_running()
    ↓ 2. GET /v1/info
queen-rbee (port 8500)
    ↓ Returns: {"base_url": "http://localhost:8500", ...}
rbee-keeper
    ↓ 3. start_hive("localhost", ..., Some(queen_url))
rbee-hive (port 9000)
    ↓ Started with: --queen-url http://localhost:8500 --hive-id localhost
    ↓ 4. Sends heartbeats every 30s
    POST /v1/hive-heartbeat
queen-rbee
    ↓ Updates HiveRegistry
    ↓ Broadcasts via SSE
Web UI
    ✅ Shows "1 hive online"
```

## Files Changed

### Queen-rbee (Service Discovery)

**NEW:** `bin/10_queen_rbee/src/http/info.rs` (56 LOC)
- `/v1/info` endpoint
- Returns queen's base URL and port
- Enables service discovery

**MODIFIED:** `bin/10_queen_rbee/src/http/mod.rs`
- Added `info` module
- Re-exported `handle_info`

**MODIFIED:** `bin/10_queen_rbee/src/main.rs`
- Added `/v1/info` route to router

### Queen Lifecycle (Discovery Client)

**MODIFIED:** `bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs` (+40 LOC)
- Added `fetch_queen_url()` function
- Calls `/v1/info` after ensuring queen is running
- Falls back to default if discovery fails
- Updates handle with discovered URL

**MODIFIED:** `bin/05_rbee_keeper_crates/queen-lifecycle/src/types.rs`
- Added `with_discovered_url()` method to `QueenHandle`
- Allows updating URL after discovery

**MODIFIED:** `bin/05_rbee_keeper_crates/queen-lifecycle/Cargo.toml`
- Added `serde_json` dependency for parsing responses

### Hive Lifecycle (Queen URL Propagation)

**MODIFIED:** `bin/05_rbee_keeper_crates/hive-lifecycle/src/start.rs`
- Added `queen_url: Option<&str>` parameter to `start_hive()`
- Passes `--queen-url` and `--hive-id` to rbee-hive binary
- Works for both local and remote hives
- Defaults to `http://localhost:8500` if not provided

### Bee Keeper (Integration)

**MODIFIED:** `bin/00_rbee_keeper/src/handlers/hive.rs`
- Passes `queen_url` to `start_hive()` call
- Hive automatically gets correct queen address

**MODIFIED:** `bin/00_rbee_keeper/src/job_client.rs`
- Uses discovered queen URL from handle
- `queen_handle.base_url()` instead of input URL

## Usage

### Starting a Hive (Automatic)

```bash
# Bee keeper automatically discovers queen and configures hive
./rbee hive start localhost

# Behind the scenes:
# 1. Ensures queen is running on port 8500
# 2. Fetches queen URL from GET /v1/info
# 3. Starts hive with: --queen-url http://localhost:8500 --hive-id localhost
```

### Starting a Hive (Manual Override)

```bash
# Direct hive start with custom queen URL
./target/debug/rbee-hive \
  --port 9000 \
  --queen-url http://custom-queen:8500 \
  --hive-id my-hive
```

### Remote Hives

```bash
# Bee keeper passes queen URL to remote hive via SSH
./rbee hive start gpu-server-1

# Behind the scenes:
# SSH command: nohup rbee-hive --port 9000 --queen-url http://localhost:8500 --hive-id gpu-server-1
```

## Testing

```bash
# 1. Build all binaries
cargo build --bin queen-rbee --bin rbee-hive --bin rbee-keeper

# 2. Test queen info endpoint
curl http://localhost:8500/v1/info
# {"base_url":"http://localhost:8500","port":8500,"version":"0.1.0"}

# 3. Start hive via bee keeper
./rbee hive start localhost

# 4. Verify heartbeat
curl -s http://localhost:8500/v1/heartbeats/stream
# Should show: "hives_online":1,"hive_ids":["localhost"]

# 5. Check web UI
# Open http://localhost:3002/
# Should show: "1 hive online"
```

## Benefits

1. **Zero Configuration:** Hives automatically find queen
2. **Service Discovery:** Queen tells clients where it is
3. **Flexible Deployment:** Works for local and remote hives
4. **Future-Proof:** Easy to support multiple queens or custom ports
5. **Consistent Defaults:** Everyone uses 8500/9000 by default

## Future Enhancements

- **Multi-Queen Support:** Hives could discover multiple queens
- **Dynamic Ports:** Queen could tell hives its actual port
- **Load Balancing:** Hives could choose from multiple queen URLs
- **Health Checks:** Discovery could include queen health status

---

**TEAM-292 Signature:** Implemented service discovery for automatic queen-to-hive configuration.
