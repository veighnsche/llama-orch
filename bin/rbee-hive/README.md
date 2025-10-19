# rbee-hive

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** Daemon for managing LLM worker instances on a single machine

## Overview

rbee-hive is a daemon that manages multiple LLM worker instances on a single machine.

**CRITICAL:** This is a DAEMON ONLY binary. NO CLI functionality!

## Dependencies

- rbee-hive-crates/worker-lifecycle
- rbee-hive-crates/worker-registry
- rbee-hive-crates/model-catalog
- rbee-hive-crates/model-provisioner
- rbee-hive-crates/monitor
- rbee-hive-crates/http-server
- rbee-hive-crates/download-tracker
- rbee-hive-crates/device-detection

## Usage

```bash
# Daemon arguments only
rbee-hive --config /path/to/config.toml
```

## Implementation Status

- [ ] Core functionality
- [ ] Tests
- [ ] Documentation
- [ ] Examples

## Notes

**NO CLI crate!** This is daemon-only. All management is done via HTTP API or queen-rbee.
