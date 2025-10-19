# TEAM-123: Duplicate Step Fixes

## Strategy

For each duplicate, keep the implementation that:
1. Makes real API calls (not stubs)
2. Is more complete/detailed
3. Is in the more specific file (e.g., validation.rs over cli_commands.rs)

## Fixes Applied

### 1. ✅ `the exit code is {int}` 
- Kept: cli_commands.rs:261 (then_exit_code_is)
- Removed: cli_commands.rs:384 (then_exit_code)

### 2. ✅ `I send request with node {string}`
- Kept: validation.rs:219 (makes HTTP call)
- Removed: cli_commands.rs:394 (stub)

### 3. ✅ `request is accepted`
- Kept: validation.rs:156
- Removed: authentication.rs:770

## Remaining (18 duplicates)

Run this command to fix them all:

```bash
cd /home/vince/Projects/llama-orch/test-harness/bdd/src/steps

# Fix each duplicate by commenting out the stub/weaker implementation
# Keep the one with real API calls
```

### Quick Fix List:

1. `validation fails` - Keep error_handling.rs:745, remove worker_preflight.rs:354
2. `worker returns to idle state` - Keep lifecycle.rs:562, remove integration.rs:290  
3. `error message does not contain {string}` - Keep validation.rs:353, remove error_handling.rs:1438
4. `worker is processing inference request` - Keep error_handling.rs:961, remove deadline_propagation.rs:311
5. `queen-rbee logs warning {string}` - Keep audit_logging.rs:385, remove audit_logging.rs:589
6. `I send POST to {string} without Authorization header` - Keep authentication.rs:782, remove authentication.rs:28
7. `I send {int} authenticated requests` - Keep authentication.rs:814, remove authentication.rs:690
8. `I send GET to {string} without Authorization header` - Keep authentication.rs:798, remove authentication.rs:352
9. `rbee-hive reports worker {string} with capabilities {string}` - Keep queen_rbee_registry.rs:126, remove worker_registration.rs:89
10. `rbee-hive continues running (does NOT crash)` - Keep errors.rs:113, remove error_handling.rs:1465
11. `queen-rbee starts with config:` - Keep configuration_management.rs:655, remove secrets.rs:51
12. `the response contains {int} worker(s)` - Keep queen_rbee_registry.rs:237, remove worker_registration.rs:115
13. `rbee-hive spawns a worker process` - Keep lifecycle.rs:540, remove pid_tracking.rs:18
14. `rbee-hive detects worker crash` - Keep lifecycle.rs:574, remove error_handling.rs:524
15. `log contains {string}` - Keep configuration_management.rs:669, remove authentication.rs:123
16. `r#` (4 duplicates!) - Need to examine each one
17. `systemd credential exists at {string}` - Keep secrets.rs:363, remove secrets.rs:82

## Automated Fix Script

Since there are 18 remaining, I'll create an xtask command to auto-fix them.
