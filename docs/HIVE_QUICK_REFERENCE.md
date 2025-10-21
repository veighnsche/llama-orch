# Hive Quick Reference

**Created by:** TEAM-198

## Config Files

```
~/.config/rbee/
├── config.toml          # Queen settings
├── hives.conf           # Hive definitions (edit this)
└── capabilities.yaml    # Auto-generated (don't edit)
```

## hives.conf Format

```ssh-config
Host <alias>
    HostName <ip-or-hostname>
    Port <ssh-port>
    User <ssh-user>
    HivePort <hive-port>
    BinaryPath <path>  # Optional
```

## Commands

| Command | Description |
|---------|-------------|
| `./rbee hive install -h <alias>` | Install hive |
| `./rbee hive start -h <alias>` | Start hive |
| `./rbee hive stop -h <alias>` | Stop hive |
| `./rbee hive uninstall -h <alias>` | Uninstall hive |
| `./rbee hive list` | List all hives |
| `./rbee hive ssh-test -h <alias>` | Test SSH connection |
| `./rbee hive refresh-capabilities -h <alias>` | Refresh device info |

## Workflow

1. **Add hive to config:**
   ```bash
   cat >> ~/.config/rbee/hives.conf << EOF
   Host my-hive
       HostName 192.168.1.100
       Port 22
       User admin
       HivePort 8081
   EOF
   ```

2. **Install:**
   ```bash
   ./rbee hive install -h my-hive
   ```

3. **Use:**
   ```bash
   ./rbee hive start -h my-hive
   ```

4. **Remove:**
   ```bash
   ./rbee hive uninstall -h my-hive
   # Then edit hives.conf to remove the Host entry
   ```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Alias not found" | Add to `hives.conf` |
| "Duplicate aliases" | Make aliases unique |
| "Binary not found" | Build with `cargo build --bin rbee-hive` |
| "Connection failed" | Check if hive is running |
