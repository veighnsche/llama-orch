# Projectstructuur & Scripts

```
website/
  package.json
  src/
    app.tsx            # Hono setup + routes
    pages/             # JSX pages (Home, Legal, Agency, POC, Pricing, ...)
    components/        # Hero, ValueProps, Steps, TierCards, KPI, Forms, StatusWidget
    styles/            # CSS (vanilla/utility)
    data/              # Static JSON (fallback metrics, SLA tiers)
    api/status.ts      # Pages Function (returns JSON metrics)
  public/
    images/
```

package.json (indicatief):
```json
{
  "name": "website",
  "private": true,
  "scripts": {
    "dev": "hono --serve src/app.tsx",
    "build": "node scripts/prerender.mjs",
    "preview": "npx serve dist"
  },
  "devDependencies": {
    "hono": "^4",
    "esbuild": "^0.21",
    "remark": "^15",
    "remark-html": "^16"
  }
}
```

scripts/prerender.mjs (concept):
```js
import { renderToString } from 'hono/jsx/dom/server';
import routes from '../src/routes.js'; // registry: elk item { path, Component, props }
import fs from 'node:fs/promises';

const out = 'dist';
await fs.mkdir(out, { recursive: true });
for (const { path, Component, props } of routes) {
  const html = '<!doctype html>' + renderToString(<Component {...props} />);
  const target = path === '/' ? `${out}/index.html` : `${out}${path}/index.html`;
  await fs.mkdir(target.replace(/\\/index.html$/, ''), { recursive: true });
  await fs.writeFile(target, html);
}
```

Status fallback: `src/data/status_fallback.json` → gebruikt door StatusWidget SSR.
