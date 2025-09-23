# CI Pipeline — Build, PDF, Deploy

## Doel
- Volautomatisch: SSG build → PDF genereren → deploy naar Cloudflare Pages met de PDF’s inbegrepen.

## Workflow (GitHub Actions)
- Trigger: push op `main` of release tag.
- Jobs:
  1) Setup Node, pnpm; build SSG (`pnpm build`) → `dist/`
  2) PDF job: `node scripts/render-pdf.mjs` → `dist/downloads/*.pdf`
  3) Deploy job: `wrangler pages deploy dist` met API token/account ID

## Secrets
- `CF_API_TOKEN`, `CF_ACCOUNT_ID` (Pages deploy)  
- (optioneel) `MAIL_*` voor contact function tests

## Artefacten
- Bewaar `dist/` als artifact om precies te zien wat live staat.
