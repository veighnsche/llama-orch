# Build, Deploy & Cloudflare Pages Setup

## Build & Deploy
- Build: `pnpm build` → prerender alle routes (SSG) met Hono
- Deploy: Cloudflare Pages (CI), Functions voor `/api/status` en formulieren
- Secrets: Turnstile keys + mail API keys via Pages Environment Variables

## Cloudflare Pages setup (stappen)
- Repo koppelen → Project (Pages)
- Build command: `pnpm install && pnpm build`
- Build output: `dist`
- Pages Functions: map `functions/` (Node compat on), routes `/api/status`, `/api/contact`
- Env vars (Production/Preview): `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY`, `MAIL_PROVIDER=mailchannels`, `MAIL_FROM`, `MAIL_TO`
- Analytics: enable Cloudflare Web Analytics (cookieless)

### Contact function (concept)
```ts
export const onRequestPost: PagesFunction = async (ctx) => {
  const body = await ctx.request.json();
  // verify Turnstile with TURNSTILE_SECRET_KEY
  // send email via MailChannels (or SendGrid)
  return new Response(JSON.stringify({ ok: true }), { headers: { 'content-type': 'application/json' } });
};
```

### Status function (concept)
```ts
export const onRequestGet: PagesFunction = async () => {
  // optional: read KV/R2; else return static numbers
  return Response.json({ uptime_pct: 99.7, p50_latency_ms: 280, sla_clients_count: 6, updated_at: new Date().toISOString() });
};
```
