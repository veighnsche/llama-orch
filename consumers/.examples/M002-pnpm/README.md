# m002-pnpm

This example runs with Node + pnpm using `tsx` to execute TypeScript directly.

## Install

Run these from the repo root to ensure workspace linking:

```bash
pnpm install
```

## Build the local utils package once (first time)

```bash
pnpm --filter @llama-orch/utils run build
```

## Dev loop

Terminal A – rebuild `@llama-orch/utils` on changes:

```bash
pnpm --filter @llama-orch/utils run dev:watch
```

Terminal B – run this example with watch:

```bash
pnpm --filter m002-pnpm run dev
```

Requirements: Node >= 20.
