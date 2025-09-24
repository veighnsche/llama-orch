# Draft 002 — Minimal static site (display + email capture)

This is a no-build, static HTML/CSS/JS site intended for Draft 002. It focuses on maximum visitor impact with minimum complexity: clear proposition, value props, steps, pricing/SLA tiers, vertical variants, checklists, and a simple email capture.

## Files
- `index.html` — one-page layout with anchors (Hero, Legal, Agency, POC, Pricing & SLA, Checklists, Contact)
- `styles.css` — lean responsive styling (< 10KB, no frameworks)
- `main.js` — progressive enhancement for email submissions (JSON POST or mailto fallback)

## Configure email submission (choose one)

- Option A: Form endpoint (e.g., Formspree, CF Worker, simple API)
  1) Set a POST endpoint on the `<body>` via `data-form-endpoint`:
     ```html
     <body data-form-endpoint="https://formspree.io/f/yourId">
     ```
  2) Submissions will send `{ email, consent, page_url, ts }` as JSON.

- Option B: Mailto fallback (no external service)
  1) Set a target email address on the `<body>` via `data-mail-to`:
     ```html
     <body data-mail-to="you@company.com">
     ```
  2) The form will open the user’s email client prefilled with details.

You can set both. The script tries JSON POST first; if it fails, it falls back to `mailto:`.

## Local preview
No build step needed. Open `index.html` directly in a browser, or use a tiny static server to preserve anchor behavior across browsers:

```bash
python3 -m http.server --directory /home/vince/Projects/llama-orch/consumers/.business/ondernemersplan_NL_nl/.002-draft/website 8080
```
Then visit: http://localhost:8080

## Customizing copy
The text is derived from:
- `.001-draft/marketing/landing_page_NL.md`
- `.001-draft/marketing/landing_vertical_legal_NL.md`
- `.001-draft/marketing/landing_vertical_agency_SI_NL.md`
- `.001-draft/marketing/poc_pakket_NL.md`
- `.001-draft/2_5_6_marketingmix_en_doelen.md`

Edit the corresponding sections in `index.html` to keep the messaging aligned.

## Checklists downloads
Links in the Checklists section are placeholders. To enable downloads:
- Put your PDFs in a `files/` subfolder here, e.g. `website/files/dpia_checklist.pdf`.
- Update the links under `#checklists` accordingly.

## Design tokens
Tweak color and spacing in `styles.css` under `:root { ... }`.

## Notes
- No cookies, analytics, or CAPTCHAs are included here. For production, add Cloudflare Web Analytics (cookieless) and Turnstile to forms as needed.
- This draft intentionally avoids any build tooling to keep it portable and reviewable inside the repo.
