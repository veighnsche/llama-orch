# Visual Regression Testing

**Created by:** TEAM-FE-011 (aka TEAM-DX-000)

## Overview

Snapshot testing for HTML/CSS without visual screenshots. Detects structural and style changes.

## Core Features

### 1. Snapshot Creation

```bash
dx snapshot --create --name "homepage" http://localhost:3000
```

Captures DOM structure, computed styles, and element positions.

### 2. Snapshot Comparison

```bash
dx snapshot --compare --name "homepage" http://localhost:3000
```

Compares current state against baseline, reports differences.

### 3. Snapshot Update

```bash
dx snapshot --update --name "homepage" http://localhost:3000
```

Updates baseline when changes are intentional.

## Implementation

Uses `insta` crate for snapshot testing with JSON serialization.

---

**Next:** See `05_INTEGRATION.md` for workspace integration.
