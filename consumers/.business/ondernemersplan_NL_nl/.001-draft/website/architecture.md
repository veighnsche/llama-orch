# Platform & Architectuur

- Hosting: Cloudflare Pages (SSG) + Pages Functions (voor lichte API’s)
- Framework: Hono + JSX templates (SSR/SSG tijdens build)
- Statische assets: bundling via esbuild/Vite (CSS minimal, geen framework CSS nodig)
- Analyses: Cloudflare Web Analytics (cookieless) + optioneel Plausible (EU)
- Security: HTTPS, security headers (CSP licht), Cloudflare Turnstile op formulieren
