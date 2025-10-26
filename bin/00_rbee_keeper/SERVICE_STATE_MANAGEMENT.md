# Service State Management

**Version:** 1.0  
**Date:** 2025-10-26  
**Status:** SPECIFICATION

## Overview

This document defines the state model for Queen and Hive services in rbee-keeper UI.

## Service States

Both Queen and Hive services can be in one of the following states:

### Installation States

| State | Description | UI Indicator |
|-------|-------------|--------------|
| **Not Installed** | Binary not found in `~/.local/bin` | Gray badge "Not Installed" |
| **Installed** | Binary exists but not running | Blue badge "Ready" |
| **Out of Date** | Binary version < latest available | Yellow badge "Update Available" |

### Runtime States

| State | Description | UI Indicator |
|-------|-------------|--------------|
| **Healthy** | Service running and responding to health checks | Green badge "Healthy" |
| **Unhealthy** | Service running but failing health checks | Red badge "Unhealthy" |
| **Stopped** | Service not running (but installed) | Gray badge "Stopped" |

## State Transitions

```
Not Installed → (install) → Installed/Ready
Installed/Ready → (start) → Healthy
Healthy → (health check fails) → Unhealthy
Healthy → (stop) → Stopped
Unhealthy → (stop) → Stopped
Stopped → (start) → Healthy
Installed/Ready → (update available) → Out of Date
Out of Date → (update) → Installed/Ready
```

## State Detection Logic

### Queen Service

**Installation Check:**
```bash
# Check if binary exists
test -f ~/.local/bin/queen-rbee
```

**Version Check:**
```bash
# Get installed version
~/.local/bin/queen-rbee --version

# Compare with git repo version
cd ~/Projects/llama-orch && git describe --tags
```

**Health Check:**
```bash
# HTTP health endpoint
curl -f http://localhost:8500/health
```

### Hive Service

**Installation Check:**
```bash
# Check if binary exists
test -f ~/.local/bin/rbee-hive
```

**Version Check:**
```bash
# Get installed version
~/.local/bin/rbee-hive --version

# Compare with git repo version
cd ~/Projects/llama-orch && git describe --tags
```

**Health Check:**
```bash
# HTTP health endpoint
curl -f http://localhost:7835/health
```

## State Priority

When multiple states apply, display in this priority order:

1. **Unhealthy** (highest priority - service is broken)
2. **Healthy** (service is running)
3. **Out of Date** (service needs update)
4. **Stopped** (service is installed but not running)
5. **Not Installed** (lowest priority - nothing to manage)

## UI Implementation

### ServiceCard Component

The `ServiceCard` component displays a status badge in the top-right corner of the card header.

**Badge Variants:**
- `destructive` - Red (Unhealthy)
- `default` - Green (Healthy)
- `secondary` - Gray (Not Installed, Stopped)
- `outline` - Yellow/Amber (Out of Date)

**Example:**
```tsx
<ServiceCard
  title="Queen"
  description="Smart API server"
  details="Job router that dispatches inference requests..."
  servicePrefix="queen"
  status="healthy" // or "unhealthy" | "stopped" | "not-installed" | "out-of-date"
  onCommandClick={handleCommand}
  disabled={isExecuting}
/>
```

## State Polling

**Frequency:**
- Installation state: Check once on page load
- Version state: Check once on page load
- Health state: Poll every 5 seconds while service is expected to be running

**Optimization:**
- Stop health polling when service is known to be stopped
- Resume health polling after start command is issued
- Debounce rapid state changes (e.g., during startup)

## Error Handling

**Health Check Failures:**
- 1 failed check → Keep current "Healthy" state (transient failure)
- 2 consecutive failed checks → Transition to "Unhealthy"
- 3 consecutive failed checks → Transition to "Stopped" (assume crashed)

**Installation Check Failures:**
- Permission denied → Show error toast, keep current state
- Binary not executable → Show "Not Installed" state with error message

## Future Enhancements

1. **Version Details:** Show installed version vs latest version in tooltip
2. **Uptime Tracking:** Display how long service has been running
3. **Auto-Recovery:** Automatically restart unhealthy services
4. **Notifications:** Desktop notifications for state changes
5. **Historical State:** Log state transitions for debugging

## Implementation Checklist

- [ ] Add `status` prop to `ServiceCard` component
- [ ] Add `Badge` component to card header (top-right)
- [ ] Create `useServiceState` hook for state detection
- [ ] Implement installation check (binary exists)
- [ ] Implement version check (compare with git)
- [ ] Implement health check polling (HTTP endpoint)
- [ ] Add state priority logic
- [ ] Add error handling for failed checks
- [ ] Add state polling with 5s interval
- [ ] Add debouncing for rapid state changes
- [ ] Update `ServicesPage` to pass status prop
- [ ] Add loading states during checks
- [ ] Add tooltips with detailed state info

## References

- Component: `/bin/00_rbee_keeper/ui/src/components/ServiceCard.tsx`
- Page: `/bin/00_rbee_keeper/ui/src/pages/ServicesPage.tsx`
- Tauri Commands: `/bin/00_rbee_keeper/src-tauri/src/main.rs`
