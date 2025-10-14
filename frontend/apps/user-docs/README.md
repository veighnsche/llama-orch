# rbee User Documentation

This is the documentation site for **rbee** (Private LLM Hosting in the Netherlands), built with [Next.js 15](https://nextjs.org) and [Nextra](https://nextra.site).

## ⚠️ SETUP INCOMPLETE

**The Nextra theme layout is not rendering correctly.** See [`NEXTRA_SETUP_ISSUE.md`](./NEXTRA_SETUP_ISSUE.md) for full details and debugging instructions.

## Architecture

- **App Router** (`app/`) - Landing page and marketing content
- **Pages Router** (`pages/`) - Documentation powered by Nextra
- **Deployment** - Cloudflare Workers via OpenNext

## Getting Started

### Local Development

```bash
# Install dependencies
pnpm install

# Run dev server
pnpm dev
```

Open [http://localhost:3100](http://localhost:3100) to view the documentation.

### Building for Production

```bash
# Build for Cloudflare Workers
pnpm build

# Preview locally
pnpm preview
```

## Documentation Setup

### Adding a New Doc Page

1. Create a new `.mdx` file in `pages/`:
   ```bash
   pages/
     my-new-page.mdx
   ```

2. Add metadata to `pages/_meta.json`:
   ```json
   {
     "my-new-page": "My New Page Title"
   }
   ```

3. Write your content using MDX:
   ```mdx
   # My New Page
   
   Content goes here...
   ```

### Organizing Pages

For nested pages, create a directory with its own `_meta.json`:

```bash
pages/
  guide/
    _meta.json
    overview.mdx
    deployment.mdx
```

### Using Custom Components

Custom MDX components are defined in `mdx-components.tsx`. To use them:

```mdx
<Callout type="warning">
  This is a warning message.
</Callout>
```

**TODO:** Wire shared UI components from `packages/ui` when available.

## Cloudflare Deployment

This app is configured for Cloudflare Workers deployment:

- `open-next.config.ts` - OpenNext configuration
- `wrangler.jsonc` - Cloudflare Workers settings
- `next.config.ts` - Includes `images.unoptimized = true` for Workers compatibility

Deploy with:

```bash
pnpm deploy
```

## Project Structure

```
user-docs/
├── app/              # App Router (landing page)
├── pages/            # Pages Router (Nextra docs)
│   ├── _meta.json
│   ├── index.mdx
│   ├── getting-started.mdx
│   └── guide/
├── public/           # Static assets
├── theme.config.tsx  # Nextra theme configuration
├── mdx-components.tsx # Custom MDX components
└── next.config.ts    # Next.js + Nextra config
```

## Learn More

- [Nextra Documentation](https://nextra.site) - Learn about Nextra features
- [Next.js Documentation](https://nextjs.org/docs) - Next.js features and API
- [OpenNext Cloudflare](https://opennext.js.org/cloudflare) - Cloudflare Workers adapter
