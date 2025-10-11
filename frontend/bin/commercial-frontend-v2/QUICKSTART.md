# Quick Start Guide

**Created by:** TEAM-FE-000  
**Date:** 2025-10-11

## Prerequisites

You need a package manager installed. This project uses pnpm (workspace monorepo).

### Install pnpm (if not installed)

```bash
# Using npm
npm install -g pnpm

# Or using curl
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

## Installation

From the project root:

```bash
# Install all workspace dependencies
pnpm install
```

Or from this directory:

```bash
# Install this package only
pnpm install
```

## Verify Build

```bash
# Type check
pnpm type-check

# Build
pnpm build

# Should complete without errors
```

## Development

```bash
# Start dev server
pnpm dev

# Open http://localhost:5173
```

## Next Steps

1. **Read WORKFLOW.md** - Understand the 13-team sequence
2. **Read .handoffs/TEAM-FE-000-PROJECT-MANAGER.md** - Project manager handoff
3. **Start with TEAM-FE-001-CONTENT-STRATEGY** - First team in sequence

## Troubleshooting

### "Cannot find module 'orchyra-storybook'"

Run `pnpm install` from the **project root** to install workspace dependencies.

### "Cannot find module 'orchyra-frontend-tooling'"

Same as above - workspace packages need to be installed from root.

### Build errors

Make sure you're using Node.js 20.19+ or 22.12+:

```bash
node --version
```

## Project Structure

```
commercial-frontend-v2/
├── .handoffs/           # Team handoff documents
├── .content/            # Content deliverables
├── .design/             # Design deliverables
├── .qa/                 # QA deliverables
├── src/                 # Source code
├── README.md            # Project overview
├── WORKFLOW.md          # 13-team workflow sequence
└── QUICKSTART.md        # This file
```

## Ready to Start

✅ Scaffold complete  
✅ Dependencies configured  
✅ Workflow defined  
✅ Documentation ready

**Next:** TEAM-FE-001-CONTENT-STRATEGY
