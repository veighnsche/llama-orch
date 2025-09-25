# Orchyra Global Storybook

A single Storybook workspace for all Vue packages in the monorepo. Initial integration is with the commercial frontend at `consumers/.business/commercial-frontend/`.

## Stack

- Framework: `@storybook/vue3-vite`
- Builder: Vite
- Package manager: pnpm (workspace)
- Addons: docs, a11y, vitest (optional)

## Run

From the repo root:

```sh
pnpm --filter orchyra-storybook storybook
```

Then open <http://localhost:6006>.

If Playwright prompts for sudo to install system dependencies, you can skip that and just install the browser binary:

```sh
pnpm --filter orchyra-storybook exec playwright install chromium
```

## How components are imported

We alias the commercial app’s `src/` as both `@commercial` and `@` in `.storybook/main.js`:

```js
// .storybook/main.js
viteFinal(config) {
  config.resolve.alias = {
    ...(config.resolve.alias || {}),
    '@commercial': '<repo>/consumers/.business/commercial-frontend/src',
    '@': '<same as above>',
  }
  return config
}
```

This allows stories to import components like:

```ts
import Button from '@commercial/components/ui/Button.vue'
```

Global styles `tokens.css` and `blueprint.css` are loaded in `.storybook/preview.js`.

## Router support

A memory-history `vue-router` is registered in `preview.js` so `RouterLink` renders in stories.

## Adding more Vue packages later

- Add the package path to the workspace (pnpm-workspace.yaml), if not already present.
- Add a Vite alias in `.storybook/main.js`, e.g. `@marketing` -> `consumers/marketing-site/src`.
- Write stories that import from that alias.

## Creating stories for commercial UI

See `stories/ButtonCommercial.stories.ts` for the Button story with controls for `variant`, `size`, `iconOnly`, `block`, and `as/href/to`.

## Notes

- We enabled `docgen: 'vue-component-meta'` for improved controls/doc gen.
- If you hit missing peer deps, run `pnpm -w install` at the repo root.
