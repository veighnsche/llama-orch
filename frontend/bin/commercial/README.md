# rbee Commercial Frontend

**Cloudflare Workers-Ready Next.js Marketing Website**

This is the production marketing website for rbee - an open-source AI orchestration platform. Built with Next.js 15 and deployed to Cloudflare Workers using the `@opennextjs/cloudflare` adapter.

## 🚀 Quick Start

### Development
```bash
pnpm dev
```
Opens [http://localhost:3000](http://localhost:3000) with Turbopack for instant HMR.

### Production Build
```bash
pnpm run build
```
Builds optimized static pages and Cloudflare Workers-compatible output.

### Preview
```bash
pnpm run preview
```
Preview the production build locally before deploying.

### Deploy to Cloudflare
```bash
pnpm run deploy
```
Builds and deploys to Cloudflare Workers.

## 📁 Project Structure

**This project follows Atomic Design methodology** - see [ATOMIC_DESIGN.md](./ATOMIC_DESIGN.md) for details.

```
commercial/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout with Navigation + Theme
│   ├── page.tsx           # Home page
│   ├── developers/        # Developer landing page
│   ├── enterprise/        # Enterprise solutions page
│   ├── features/          # Features showcase
│   ├── gpu-providers/     # GPU provider marketplace
│   ├── pricing/           # Pricing tiers
│   └── use-cases/         # Use case examples
├── components/            # 107 React components (Atomic Design)
│   ├── atoms/            # 57 basic UI elements (Button, Input, etc.)
│   ├── molecules/        # 26 simple combinations (ThemeToggle, Cards)
│   ├── organisms/        # 23 complex sections (Navigation, Footer)
│   ├── templates/        # Page layouts (to be added)
│   └── providers/        # 1 context provider (ThemeProvider)
├── lib/                  # Utility functions
├── hooks/                # Custom React hooks
└── styles/               # Global styles and tokens
```

## 🎨 Tech Stack

- **Framework:** Next.js 15 (App Router)
- **Runtime:** Cloudflare Workers (via @opennextjs/cloudflare)
- **Styling:** TailwindCSS v4
- **UI Components:** Radix UI + shadcn/ui
- **Icons:** Lucide React
- **Fonts:** Geist Sans & Geist Mono
- **Theme:** next-themes (dark/light mode)
- **Forms:** react-hook-form + zod
- **Charts:** Recharts
- **Animations:** tailwindcss-animate + tw-animate-css

## 📦 Key Dependencies

- `@opennextjs/cloudflare` - Cloudflare Workers adapter
- `@radix-ui/react-*` - 27 accessible UI primitives
- `class-variance-authority` - Type-safe variant styling
- `lucide-react` - Icon library
- `next-themes` - Theme management
- `recharts` - Data visualization
- `sonner` - Toast notifications
- `vaul` - Drawer component

## 🌐 Routes

All routes are pre-rendered as static pages:

- `/` - Home page with all marketing sections
- `/developers` - Developer-focused landing page
- `/enterprise` - Enterprise solutions
- `/features` - Feature showcase
- `/gpu-providers` - GPU provider marketplace info
- `/pricing` - Pricing tiers and comparison
- `/use-cases` - Use case examples

## 🎯 Features

- ✅ **Zero Vendor Lock-in** - Deploys to Cloudflare Workers, not Vercel
- ✅ **Static Generation** - All pages pre-rendered for performance
- ✅ **Dark Mode** - System-aware theme switching
- ✅ **Responsive** - Mobile-first design
- ✅ **Accessible** - Radix UI primitives with ARIA support
- ✅ **Type-Safe** - Full TypeScript coverage
- ✅ **Fast Refresh** - Turbopack for instant updates
- ✅ **SEO Optimized** - Metadata and semantic HTML

## 🔧 Configuration

### Cloudflare Workers
Configured via `wrangler.jsonc` and `open-next.config.ts`.

### ESLint
Custom rules in `eslint.config.mjs` to allow marketing copy with quotes/apostrophes.

### TypeScript
Path aliases configured: `@/*` maps to project root.

### Tailwind
Design tokens and theme variables in `app/globals.css`.

## 📝 Migration Notes

This project was migrated from `commercial-old` to remove Vercel-specific dependencies. See `MIGRATION_COMPLETE.md` for full details.

**Key Changes:**
- Removed `@vercel/analytics`
- Added `@opennextjs/cloudflare` adapter
- Updated build/deploy scripts for Cloudflare Workers
- Maintained 100% feature parity

## 🚢 Deployment

### Cloudflare Workers
```bash
# First time setup
pnpm run cf-typegen  # Generate Cloudflare types

# Deploy
pnpm run deploy
```

### Environment Variables
Configure in `.dev.vars` for local development and Cloudflare dashboard for production.

## 🛠️ Development

### Adding Components
Uses shadcn/ui CLI:
```bash
npx shadcn@latest add [component-name]
```

### Code Style
- ESLint + TypeScript for linting
- Prettier-compatible (via ESLint rules)
- Consistent component patterns

### Testing
```bash
pnpm run lint     # ESLint check
pnpm run build    # Build verification
```

## 📚 Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [Cloudflare Workers](https://developers.cloudflare.com/workers/)
- [@opennextjs/cloudflare](https://opennext.js.org/cloudflare)
- [Radix UI](https://www.radix-ui.com/)
- [shadcn/ui](https://ui.shadcn.com/)
- [TailwindCSS](https://tailwindcss.com/)

## 📄 License

GPL-3.0-or-later (see project root LICENSE file)
