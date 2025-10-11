# rbee Commercial Frontend v2

**Created by:** TEAM-FE-000 (Project Manager)  
**Date:** 2025-10-11  
**Status:** Scaffold Complete - Ready for Department Workflow

## Project Overview

Complete rewrite of the rbee commercial frontend based on stakeholder documentation. This project follows a professional agency workflow with clear department handoffs.

## Tech Stack

- **Framework:** Vue 3 + TypeScript
- **Build Tool:** Vite
- **Router:** Vue Router 4
- **Component Library:** orchyra-storybook (workspace package)
- **Design Tokens:** orchyra-storybook/styles/tokens.css
- **Tooling:** orchyra-frontend-tooling (workspace package)

## Installation

From this directory:

```bash
pnpm install
```

## Development

```bash
# Start dev server
pnpm dev

# Type check
pnpm type-check

# Build for production
pnpm build

# Preview production build
pnpm preview

# Lint
pnpm lint
pnpm lint:fix

# Format
pnpm format
pnpm format:fix
```

## Project Structure

```
commercial-frontend-v2/
├── src/
│   ├── assets/          # CSS, images, fonts
│   ├── components/      # Vue components
│   ├── composables/     # Vue composables
│   ├── layouts/         # Layout components
│   ├── router/          # Vue Router configuration
│   ├── views/           # Page components
│   ├── App.vue          # Root component
│   └── main.ts          # Application entry point
├── public/              # Static assets
├── index.html           # HTML entry point
├── package.json         # Dependencies
├── vite.config.ts       # Vite configuration
└── tsconfig.*.json      # TypeScript configuration
```

## Design System Integration

This project uses the shared `orchyra-storybook` component library:

```vue
<script setup lang="ts">
// ✅ CORRECT: Use workspace package
import { Button } from 'orchyra-storybook/stories'

// ❌ WRONG: Never use relative imports
// import Button from '../../../libs/storybook/stories/Button/Button.vue'
</script>

<style>
/* ✅ CORRECT: Import design tokens */
@import 'orchyra-storybook/styles/tokens.css';
</style>
```

## Stakeholder Documentation

Source material for content and messaging:
- `/home/vince/Projects/llama-orch/.business/stakeholders/STAKEHOLDER_STORY.md`
- `/home/vince/Projects/llama-orch/.business/stakeholders/AGENTIC_AI_USE_CASE.md`
- `/home/vince/Projects/llama-orch/.business/stakeholders/TECHNICAL_DEEP_DIVE.md`
- `/home/vince/Projects/llama-orch/.business/stakeholders/ENGINEERING_GUIDE.md`

## Department Workflow

See `WORKFLOW.md` for the complete department sequence and handoff process.

## Next Steps

1. Review `WORKFLOW.md` for department sequence
2. First department: **Content Strategy** (TEAM-FE-001-CONTENT-STRATEGY)
3. Follow handoff checklist for each department

## Notes

- All new files include `// Created by: TEAM-FE-000` signature
- Storybook components are imported via workspace package
- Design tokens are centralized in orchyra-storybook
- No placeholder content or lorem ipsum - waiting for real content from Content Strategy team
