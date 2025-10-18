# Tailwind CSS Fix - Commercial Frontend

## ⚠️ DEPRECATED - See TURBOREPO_PATTERN.md

This document describes the initial `@source` directive fix, which has been **replaced** with the official Turborepo pattern.

**Current implementation:** See `/home/vince/Projects/llama-orch/frontend/TURBOREPO_PATTERN.md`

---

## Original Problem (SOLVED)
The commercial frontend at `http://localhost:3000/` was not applying Tailwind CSS styles to components imported from `@rbee/ui`, while Storybook at `http://localhost:6006/` was working correctly.

## Original Solution (REPLACED)
Initially fixed with `@source` directive, but this is **not** the Turborepo recommended pattern.

## Current Solution (TURBOREPO PATTERN)
The UI package now:
1. Pre-builds its CSS: `pnpm run build:styles`
2. Exports compiled CSS: `"./styles.css": "./dist/index.css"`
3. Consumer apps import: `import '@rbee/ui/styles.css'`

**No `@source` directive needed!**

## Migration
- ✅ UI package builds its own CSS
- ✅ Commercial app imports pre-built CSS
- ✅ Removed `@source` directive
- ✅ Follows official Turborepo pattern

## Reference
- [Current Implementation](../../../TURBOREPO_PATTERN.md)
- [Turborepo Tailwind Guide](https://turborepo.com/docs/guides/tools/tailwind)
- [Example: with-tailwind](https://github.com/vercel/turborepo/tree/main/examples/with-tailwind)
