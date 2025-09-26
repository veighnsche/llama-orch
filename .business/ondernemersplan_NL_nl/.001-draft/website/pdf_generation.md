# PDF Generatie — uit MDX (Build/CI)

Doel: automatisch PDF’s leveren (download) die 1:1 overeenkomen met de online content.

## Strategieën
- Lokale build (snelste start): Node script gebruikt Puppeteer om de prerenderde HTML te “printen naar PDF”.
- CI build (aanbevolen): GitHub Actions job draait na SSG build en voegt PDF’s toe aan `dist/downloads/` vóór deploy.
- Cloudflare Pages: headless Chrome niet beschikbaar in Pages build; daarom PDF in CI of lokaal pre‑commit.

## Node dependencies
- `puppeteer` (Actions/local) of `puppeteer-core` + extern Chromium in CI image.
- Alternatief zonder browser (minder mooi): `pdfkit` (tekst‑gebaseerd), alleen als fallback.

## Script (concept)
```js
// scripts/render-pdf.mjs
import { launch } from 'puppeteer';
const pages = [
  { url: 'file://' + process.cwd() + '/dist/plans/ondernemersplan/index.html', out: 'dist/downloads/ondernemersplan.pdf' },
  { url: 'file://' + process.cwd() + '/dist/plans/financieel-plan/index.html', out: 'dist/downloads/financieel-plan.pdf' }
];
const browser = await launch({ args: ['--no-sandbox'] });
const page = await browser.newPage();
for (const p of pages) {
  await page.goto(p.url, { waitUntil: 'networkidle0' });
  await page.pdf({ path: p.out, format: 'A4', printBackground: true });
}
await browser.close();
```

## GitHub Actions (schets)
```yaml
name: build-site
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - run: pnpm i
      - run: pnpm build   # SSG → dist/
      - run: mkdir -p dist/downloads && node scripts/render-pdf.mjs
      - name: Upload artifact to Pages
        run: npx wrangler pages deploy dist --project-name <project> --branch ${{ github.ref_name }}
        env:
          CLOUDFLARE_API_TOKEN: ${{ secrets.CF_API_TOKEN }}
          CLOUDFLARE_ACCOUNT_ID: ${{ secrets.CF_ACCOUNT_ID }}
```

## Download links
- Voeg knoppen toe op `/plans/*` die verwijzen naar `/downloads/ondernemersplan.pdf` en `/downloads/financieel-plan.pdf`.
