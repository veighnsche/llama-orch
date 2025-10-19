# queen-rbee

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** Daemon for managing rbee-hive instances across multiple machines

## Overview

queen-rbee is a daemon that manages multiple rbee-hive instances across different machines via SSH.

## Dependencies

- queen-rbee-crates/ssh-client
- queen-rbee-crates/hive-registry
- queen-rbee-crates/worker-registry
- queen-rbee-crates/hive-lifecycle
- queen-rbee-crates/http-server
- queen-rbee-crates/preflight

## Usage

```bash
# TODO: Add usage examples
queen-rbee --config /path/to/config.toml
```

## Implementation Status

- [ ] Core functionality
- [ ] Tests
- [ ] Documentation
- [ ] Examples

## Notes

This is a daemon-only binary. It manages rbee-hive lifecycle and provides HTTP API.
