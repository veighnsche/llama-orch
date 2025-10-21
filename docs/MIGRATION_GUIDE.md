# Migration Guide: SQLite to File-Based Config

**Created by:** TEAM-198

## Overview

rbee has migrated from SQLite-based hive catalog to file-based configuration.

## What Changed

### Before (SQLite)
- Hives stored in SQLite database
- CLI commands wrote to database
- `--id`, `--ssh-host`, `--ssh-user` flags required

### After (File-Based)
- Hives defined in `~/.config/rbee/hives.conf`
- User manually edits config file
- `-h <alias>` flag references config

## Migration Steps

### Step 1: Export Existing Hives

If you have existing hives in the SQLite database:

```bash
# List current hives
./rbee hive list

# Note down the details for each hive
```

### Step 2: Create hives.conf

Create `~/.config/rbee/hives.conf` and add your hives:

```ssh-config
Host <alias>
    HostName <ip-or-hostname>
    Port <ssh-port>
    User <ssh-user>
    HivePort <hive-port>
```

Example:

```ssh-config
Host localhost
    HostName 127.0.0.1
    Port 22
    User vince
    HivePort 8081

Host workstation
    HostName 192.168.1.100
    Port 22
    User admin
    HivePort 8081
```

### Step 3: Update CLI Commands

**Old commands:**
```bash
./rbee hive install --id my-hive --ssh-host 192.168.1.100 --ssh-user admin --port 8081
./rbee hive start --id my-hive
./rbee hive stop --id my-hive
```

**New commands:**
```bash
# First add to hives.conf, then:
./rbee hive install -h my-hive
./rbee hive start -h my-hive
./rbee hive stop -h my-hive
```

### Step 4: Verify Migration

```bash
# List hives (should read from hives.conf)
./rbee hive list

# Test a hive
./rbee hive ssh-test -h <alias>
```

## Breaking Changes

### CLI Arguments

| Old | New |
|-----|-----|
| `--id <id>` | `-h <alias>` or `--host <alias>` |
| `--ssh-host <host>` | Defined in `hives.conf` |
| `--ssh-port <port>` | Defined in `hives.conf` |
| `--ssh-user <user>` | Defined in `hives.conf` |
| `--port <port>` | `HivePort` in `hives.conf` |
| `--binary-path <path>` | `BinaryPath` in `hives.conf` |

### Programmatic Access

If you have scripts that use the CLI:

**Before:**
```bash
./rbee hive install --id hive1 --ssh-host 192.168.1.100 --ssh-user admin --port 8081
```

**After:**
```bash
# Add to hives.conf first (one-time setup)
cat >> ~/.config/rbee/hives.conf << EOF
Host hive1
    HostName 192.168.1.100
    Port 22
    User admin
    HivePort 8081
EOF

# Then use alias
./rbee hive install -h hive1
```

## FAQ

### Q: Can I still use the old CLI commands?

**A:** No, the old SQLite-based commands are removed. You must use the new file-based config.

### Q: Where is the SQLite database?

**A:** It's no longer used. The `hive-catalog` crate has been removed.

### Q: Can I automate hive registration?

**A:** Yes, by programmatically editing `~/.config/rbee/hives.conf`. It's a simple text file.

### Q: What happens to my old database?

**A:** It's ignored. You can safely delete it after migration.

### Q: Do I need to reinstall hives?

**A:** No, just add them to `hives.conf`. If they're already running, refresh capabilities:

```bash
./rbee hive refresh-capabilities -h <alias>
```

## Rollback

If you need to rollback (not recommended):

1. Checkout the previous version: `git checkout <old-commit>`
2. Rebuild: `cargo build --workspace`
3. Your old SQLite database should still work

## Support

If you encounter issues during migration:

1. Check `docs/HIVE_CONFIGURATION.md` for usage guide
2. Verify `hives.conf` syntax
3. Run with debug logging: `RUST_LOG=debug ./rbee hive list`
4. Open an issue on GitHub
