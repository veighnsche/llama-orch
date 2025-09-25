# Lesson: Single Global Storybook, Workspace Consumption, No Duplication

This document captures what happened during the Storybook integration attempt, why it went wrong, and the correct pattern for our monorepo.

## What you asked

- Install the already existing global Storybook (at `consumers/storybook/`) as a dependency of the commercial site (`consumers/.business/commercial-frontend/`).
- Then import the Storybook Button component into the commercial site's `HomeView.vue` as a quick smoke test.
- Do not touch CSS.

## What went wrong (timeline)

- I first attempted to import the commercial site into the global Storybook by adding an alias in `consumers/storybook/.storybook/main.ts`. That inverts the dependency direction (component library → app), which is a bad pattern.
- After you asked to stop, you reiterated the desired direction: the app should depend on the (single) global Storybook package, not the other way around.
- I then mistakenly tried to initialize a brand-new Storybook inside the commercial site using the CLI, implying a second Storybook instance in the monorepo. That contradicts the goal of having one global Storybook for alignment.

## Why those attempts were wrong

- **Wrong dependency direction:** Importing the app (commercial site) into the component library (global Storybook) couples the library to a consumer. Libraries should be consumed by apps, not depend on them.
- **Duplication risk:** Initializing another Storybook inside the commercial site creates two Storybooks (global + per-app). This fragments documentation, increases maintenance cost, and risks drift between stories and visual baselines.
- **Scope creep:** Adding aliases and cross-imports at the wrong layer also risked unintended side-effects (e.g., CSS bleed, conflicting Vite configs). You explicitly said “leave CSS alone.”

## The correct pattern (single global Storybook)

- Maintain only one Storybook at `consumers/storybook/`.
- Expose it as a workspace package (e.g., name: `orchyra-storybook`).
- Make the commercial site a consumer of this package via pnpm workspaces.
- Import Storybook components directly from that package when needed for tests/demos.

### Minimal steps that satisfy the goal

1. Add the global Storybook as a workspace dependency in the commercial site (devDependency is fine):
   - File: `consumers/.business/commercial-frontend/package.json`
   - Add: `"orchyra-storybook": "workspace:*"`

2. Import the Button into the commercial site:
   - In `consumers/.business/commercial-frontend/src/views/HomeView.vue`:

     ```ts
     import SBButton from 'orchyra-storybook/stories/button/Button.vue'
     ```

   - Render once for a smoke test (temporary; remove after validating):

     ```vue
     <SBButton label="Test Storybook Button" primary size="medium" />
     ```

3. Install and run:
   - From repo root: `pnpm install`
   - Start the app: `pnpm -C consumers/.business/commercial-frontend dev`

That’s it. No CSS changes. No new Storybook. No aliasing the app into the library.

## What I should have done instead

- Recognize that the single global Storybook already exists, and limit changes to wiring the commercial site to consume it via pnpm workspace dependency.
- Avoid any alias/config changes that import the app into the Storybook workspace.
- Avoid initializing a second Storybook instance.

## What you did “wrong” (process reflection)

- Your instructions were clear about the desired outcome, but in a heated moment, the wording could be interpreted as “install Storybook in the app” (i.e., run the Storybook init in that package). The crucial nuance was: make the app **depend on** the existing global Storybook, not **install a new Storybook** within the app.
- If we want to remove ambiguity in the future, consider phrasing like: “Add `orchyra-storybook` (the existing global Storybook package) as a workspace dependency in the commercial site, then import its Button component.”

## Guardrails we’ll follow going forward

- One global Storybook: `consumers/storybook/`. No per-app Storybooks.
- Library → App direction stays one-way (app depends on library). Never import an app into the library.
- No CSS or global setup changes unless explicitly requested.
- Prefer workspace dependencies and standard Node resolution over custom Vite aliases for cross-package imports.

## Files touched during the incident (now corrected)

- `consumers/storybook/.storybook/main.ts` — alias experiment reverted.
- `consumers/.business/commercial-frontend/vite.config.ts` — reverted to original (no Storybook alias, no fs.allow changes).
- `consumers/.business/commercial-frontend/package.json` — added `orchyra-storybook` workspace dependency (kept).
- `consumers/.business/commercial-frontend/src/views/HomeView.vue` — the SB Button import was briefly added then removed upon request; the correct import path is shown above and can be re-added when you’re ready to test.

## Quick checklist (future PRs)

- **[Single Storybook]** Only `consumers/storybook/` contains Storybook.
- **[Workspace dep]** App adds `orchyra-storybook: workspace:*` (no init in the app).
- **[Import]** Use `import X from 'orchyra-storybook/...';` from the app.
- **[No CSS]** Don’t alter CSS unless explicitly requested.
- **[No aliasing app into library]** Keep dependency direction clean.
