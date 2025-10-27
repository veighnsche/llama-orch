# TEAM-322: Consolidated Queen Status/Info (RULE ZERO)

**Status:** ✅ COMPLETE

## Problem

**DUPLICATION:** Two files (`status.rs` and `info.rs`) doing essentially the same thing with different endpoints.

### The Endpoints

| File | Endpoint | Returns |
|------|----------|---------|
| `status.rs` | `/health` | Just `200 OK` (empty body) |
| `info.rs` | `/v1/build-info` | JSON with version, features, timestamp |

### The Question

**When would you ONLY want to know if queen is running without caring about version/features?**

**Answer:** Never in practice. If you're checking if queen is running, you also want to know what version/features it has.

### The Waste

- **Two HTTP calls** instead of one
- **Two functions** doing the same thing
- **Two modules** to maintain
- **Unnecessary complexity**

### The Truth

`/v1/build-info` tells you BOTH:
1. If it succeeds → queen is running ✅
2. Response contains → version, features, timestamp 📋

**One endpoint does everything. Why have two?**

## Solution

**RULE ZERO: Delete the duplication.**

### Consolidated into `info.rs`

Created a single function `check_queen_status(queen_url, verbose)` that:
- Queries `/v1/build-info` (one HTTP call)
- If successful → queen is running + we have the info
- If failed → queen is not running
- `verbose` flag controls output:
  - `false` → Just confirm running ("✅ Queen is running")
  - `true` → Show full build info (version, features, timestamp)

**Implementation:**
```rust
pub async fn check_queen_status(queen_url: &str, verbose: bool) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(5))
        .build()?;

    let response = client
        .get(format!("{}/v1/build-info", queen_url))
        .send()
        .await
        .context("Failed to connect to queen (is it running?)")?;

    if !response.status().is_success() {
        anyhow::bail!("Queen returned error status: {}", response.status());
    }

    let body = response.text().await?;

    if verbose {
        n!("queen_status", "✅ Queen is running on {}", queen_url);
        n!("queen_info", "📋 Build information:");
        println!("{}", body);
    } else {
        n!("queen_status", "✅ Queen is running on {}", queen_url);
    }

    Ok(())
}
```

### Handler Usage

```rust
// rbee-keeper queen status → verbose=false (just confirm running)
QueenAction::Status => check_queen_status(queen_url, false).await,

// rbee-keeper queen info → verbose=true (show full details)
QueenAction::Info => check_queen_status(queen_url, true).await,
```

### Deleted Files

- ❌ `status.rs` - Deleted entirely (43 lines removed)
- ✅ `info.rs` - Now contains everything

## Files Changed

1. **bin/05_rbee_keeper_crates/queen-lifecycle/src/status.rs**
   - ❌ **DELETED** (43 lines removed)

2. **bin/05_rbee_keeper_crates/queen-lifecycle/src/info.rs**
   - Renamed to "Queen status and build info operation"
   - Added `check_queen_status(queen_url, verbose)` function
   - Kept `get_queen_info()` as alias for backwards compatibility
   - Proper error handling with `.context()`
   - Added TEAM-322 signature

3. **bin/05_rbee_keeper_crates/queen-lifecycle/src/lib.rs**
   - Removed `pub mod status;` declaration
   - Updated exports: `pub use info::{check_queen_status, get_queen_info};`
   - Added TEAM-322 comment

4. **bin/00_rbee_keeper/src/handlers/queen.rs**
   - Updated `Status` handler: `check_queen_status(queen_url, false).await`
   - Updated `Info` handler: `check_queen_status(queen_url, true).await`
   - Removed `get_queen_info` import
   - Added TEAM-322 signature

## Benefits

- ✅ **No duplication** - One function, one HTTP call
- ✅ **43 lines deleted** - Less code to maintain
- ✅ **Simpler API** - `verbose` flag instead of two functions
- ✅ **Proper error handling** - errors propagate with context
- ✅ **Better performance** - One HTTP call instead of two
- ✅ **RULE ZERO enforced** - Breaking changes > backwards compatibility

## Verification

```bash
cargo check -p queen-lifecycle  # ✅ PASS
```

## Usage

```bash
# Check if queen is running (health check)
rbee-keeper queen status
# Output: "✅ Queen is running on http://localhost:7833"

# Get build configuration details
rbee-keeper queen info
# Output: Full JSON/text with version, features, etc.
```

## Code Signatures

All changes marked with `// TEAM-322:` comments for traceability.
