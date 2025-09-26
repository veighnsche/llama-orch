---
trigger: glob
globs: src/**/*.vue
---

1. Design tokens
   * When choosing colors, radius, shadows, or fonts, always reference `/home/vince/Projects/llama-orch/frontend/storybook/styles/tokens.css`.

2. Reusable components
   * Use reusable components from `/home/vince/Projects/llama-orch/frontend/storybook/stories`.
   * If a reusable building block is missing, create a new component in that directory.

3. File naming
   * Vue component filenames must follow `CamelCase.vue` convention.

4. Workspace boundaries
   * This is a **pnpm workspace**. Do **not** cross-import code from sibling libraries/packages.
   * Always import dependencies through their declared package entry points (e.g. `@scope/pkg`), never via relative paths into other workspace `src/`.
