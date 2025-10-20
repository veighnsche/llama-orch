# Hive Catalog - CRUD Pattern

**Date:** 2025-10-20  
**Team:** TEAM-158  
**Pattern:** Standard CRUD operations

---

## CRUD Operations

The `HiveCatalog` follows a standard CRUD (Create, Read, Update, Delete) pattern for managing hive records.

### **C**reate

```rust
/// Add a new hive to the catalog
pub async fn add_hive(&self, hive: HiveRecord) -> Result<()>
```

**Usage:**
```rust
let hive = HiveRecord {
    id: "localhost".to_string(),
    host: "127.0.0.1".to_string(),
    port: 8600,
    status: HiveStatus::Unknown,
    // ...
};
catalog.add_hive(hive).await?;
```

---

### **R**ead

```rust
/// Get a specific hive by ID
pub async fn get_hive(&self, id: &str) -> Result<Option<HiveRecord>>

/// List all hives in the catalog
pub async fn list_hives(&self) -> Result<Vec<HiveRecord>>
```

**Usage:**
```rust
// Get single hive
let hive = catalog.get_hive("localhost").await?;

// List all hives
let hives = catalog.list_hives().await?;
```

---

### **U**pdate

```rust
/// Update an entire hive record
pub async fn update_hive(&self, hive: HiveRecord) -> Result<()>

/// Update hive status only
pub async fn update_hive_status(&self, id: &str, status: HiveStatus) -> Result<()>

/// Update hive heartbeat timestamp only
pub async fn update_heartbeat(&self, id: &str, timestamp_ms: i64) -> Result<()>
```

**Usage:**
```rust
// Full update
let mut hive = catalog.get_hive("localhost").await?.unwrap();
hive.host = "192.168.1.100".to_string();
hive.port = 9000;
catalog.update_hive(hive).await?;

// Partial update - status only
catalog.update_hive_status("localhost", HiveStatus::Online).await?;

// Partial update - heartbeat only
let now = chrono::Utc::now().timestamp_millis();
catalog.update_heartbeat("localhost", now).await?;
```

---

### **D**elete

```rust
/// Remove a hive from the catalog
pub async fn remove_hive(&self, id: &str) -> Result<()>
```

**Usage:**
```rust
catalog.remove_hive("localhost").await?;
```

---

## Code Organization

The CRUD operations are clearly organized in `catalog.rs`:

```rust
impl HiveCatalog {
    // ========================================================================
    // Initialization
    // ========================================================================
    pub async fn new(db_path: &Path) -> Result<Self>

    // ========================================================================
    // CREATE
    // ========================================================================
    pub async fn add_hive(&self, hive: HiveRecord) -> Result<()>

    // ========================================================================
    // READ
    // ========================================================================
    pub async fn get_hive(&self, id: &str) -> Result<Option<HiveRecord>>
    pub async fn list_hives(&self) -> Result<Vec<HiveRecord>>

    // ========================================================================
    // UPDATE
    // ========================================================================
    pub async fn update_hive(&self, hive: HiveRecord) -> Result<()>
    pub async fn update_hive_status(&self, id: &str, status: HiveStatus) -> Result<()>
    pub async fn update_heartbeat(&self, id: &str, timestamp_ms: i64) -> Result<()>

    // ========================================================================
    // DELETE
    // ========================================================================
    pub async fn remove_hive(&self, id: &str) -> Result<()>
}
```

---

## Benefits of CRUD Pattern

### ✅ Predictability
- Standard operations everyone understands
- Easy to find what you need
- Clear naming conventions

### ✅ Completeness
- All basic operations covered
- Full update + partial updates
- Single record + bulk operations

### ✅ Maintainability
- Clear section boundaries
- Easy to add new operations
- Consistent structure

### ✅ Testability
- Each operation independently testable
- Clear inputs and outputs
- Easy to mock

---

## Test Coverage

All CRUD operations are tested:

```bash
$ cargo test -p queen-rbee-hive-catalog

running 7 tests
test tests::test_create_catalog ... ok         # Initialization
test tests::test_add_and_list_hives ... ok     # CREATE + READ
test tests::test_get_hive ... ok               # READ (single)
test tests::test_update_status ... ok          # UPDATE (partial)
test tests::test_update_heartbeat ... ok       # UPDATE (partial)
test tests::test_update_hive ... ok            # UPDATE (full)
test tests::test_remove_hive ... ok            # DELETE

test result: ok. 7 passed; 0 failed
```

---

## API Summary

| Operation | Method | Returns |
|-----------|--------|---------|
| **Create** | `add_hive(hive)` | `Result<()>` |
| **Read (single)** | `get_hive(id)` | `Result<Option<HiveRecord>>` |
| **Read (all)** | `list_hives()` | `Result<Vec<HiveRecord>>` |
| **Update (full)** | `update_hive(hive)` | `Result<()>` |
| **Update (status)** | `update_hive_status(id, status)` | `Result<()>` |
| **Update (heartbeat)** | `update_heartbeat(id, timestamp)` | `Result<()>` |
| **Delete** | `remove_hive(id)` | `Result<()>` |

---

## Design Decisions

### Why 3 Update Methods?

1. **`update_hive()`** - Full record update
   - Use when you have the complete record
   - Updates all fields except `id` and `created_at_ms`

2. **`update_hive_status()`** - Status only
   - Use for status transitions (Unknown → Online)
   - Lightweight, doesn't require full record

3. **`update_heartbeat()`** - Heartbeat only
   - Use for frequent heartbeat updates
   - Optimized for high-frequency calls
   - Doesn't require full record

### Why Separate Read Methods?

1. **`get_hive()`** - Single record by ID
   - Returns `Option<HiveRecord>` (None if not found)
   - Use when you know the ID

2. **`list_hives()`** - All records
   - Returns `Vec<HiveRecord>` (empty if none)
   - Use for listing/iteration

---

## Future Enhancements

Possible additions while maintaining CRUD pattern:

```rust
// READ - Query operations
pub async fn find_hives_by_status(&self, status: HiveStatus) -> Result<Vec<HiveRecord>>
pub async fn find_online_hives(&self) -> Result<Vec<HiveRecord>>

// READ - Count operations
pub async fn count_hives(&self) -> Result<usize>
pub async fn count_by_status(&self, status: HiveStatus) -> Result<usize>

// UPDATE - Bulk operations
pub async fn update_all_statuses(&self, status: HiveStatus) -> Result<()>

// DELETE - Bulk operations
pub async fn remove_offline_hives(&self) -> Result<usize>
```

---

**TEAM-158: CRUD pattern implemented. Clear, complete, tested.**
