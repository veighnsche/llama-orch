// Cloudflare Worker entry that serves the Vue SPA via the static assets binding.
// It tries the requested asset first; if not found and the client accepts HTML, it
// falls back to /index.html (SPA routing). This works in both dev (wrangler dev)
// and production deployments.

export default {
  async fetch(request, env) {
    const url = new URL(request.url)

    // Only GET/HEAD are valid for static assets / SPA fallback
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response('Method Not Allowed', { status: 405 })
    }

    // Try to serve the exact asset first
    let res = await env.ASSETS.fetch(request)

    // Heuristic: if the asset wasn’t found (404), and the client accepts HTML, and the
    // path looks like an app route (no file extension), fall back to index.html.
    if (res.status === 404) {
      const acceptsHtml = (request.headers.get('Accept') || '').includes('text/html')
      const looksLikeRoute = !url.pathname.split('/').pop()?.includes('.')
      if (acceptsHtml && looksLikeRoute) {
        const indexReq = new Request(new URL('/index.html', url.origin).toString(), request)
        res = await env.ASSETS.fetch(indexReq)
      }
    }

    return res
  },
} satisfies ExportedHandler<Env>
