# Commercial Frontend - Nuxt Implementation

**Created by:** TEAM-FE-010  
**Date:** 2025-10-12

## Overview

This is the Nuxt 4 implementation of the commercial frontend, using pre-built view templates from `rbee-storybook/stories/templates`.

## What Was Implemented

### 1. Dependencies
- Added `rbee-storybook` workspace dependency
- Configured Tailwind CSS v4 with `@tailwindcss/vite`
- Set up Cloudflare deployment target

### 2. Styling
Updated `app/assets/css/main.css`:
- Imported Tailwind CSS
- Imported design tokens from `rbee-storybook/styles/tokens.css`
- Added CSS custom properties for theming

### 3. Pages (Nuxt Auto-routing)
Created 7 pages in `app/pages/`:
- `index.vue` - Home page (/)
- `developers.vue` - Developers page (/developers)
- `enterprise.vue` - Enterprise page (/enterprise)
- `features.vue` - Features page (/features)
- `pricing.vue` - Pricing page (/pricing)
- `providers.vue` - GPU providers page (/providers)
- `use-cases.vue` - Use cases page (/use-cases)

Each page imports and assembles organisms from `rbee-storybook/stories`.

### 4. App Root
Updated `app/app.vue` to use `<NuxtPage />` for automatic routing.

## Key Differences from Vite Template

The template README at `frontend/libs/storybook/stories/templates/README.md` provides Vite-based instructions. Here are the Nuxt-specific adaptations:

| Aspect | Vite (Template) | Nuxt (This Implementation) |
|--------|-----------------|----------------------------|
| **Routing** | Manual Vue Router setup in `src/router/index.ts` | Automatic file-based routing via `app/pages/` |
| **App Root** | `<RouterView />` in `App.vue` | `<NuxtPage />` in `app.vue` |
| **CSS Import** | In `src/assets/main.css` | In `app/assets/css/main.css` |
| **CSS Config** | `nuxt.config.ts` css array | Same, but Nuxt-specific path resolution |
| **Pages Location** | `src/views/` | `app/pages/` |
| **Dev Server** | `pnpm run dev` (Vite) | `pnpm run dev` (Nuxt) |

## Running the App

```bash
# Install dependencies (from monorepo root or this directory)
pnpm install

# Start dev server
pnpm run dev
# Opens at http://localhost:3000 (or 3001 if 3000 is taken)

# Build for production
pnpm run build

# Preview production build locally
pnpm run preview

# Deploy to Cloudflare
pnpm run deploy
```

## Verification Checklist

- ✅ All 7 pages created in `app/pages/`
- ✅ Tailwind CSS configured and working
- ✅ Design tokens imported from storybook
- ✅ `rbee-storybook` workspace dependency added
- ✅ Nuxt auto-routing configured
- ✅ Dev server runs without errors
- ✅ All components import from `rbee-storybook/stories`

## File Structure

```
frontend/bin/commercial/
├── app/
│   ├── app.vue                 # Root component with <NuxtPage />
│   ├── assets/
│   │   └── css/
│   │       └── main.css        # Global styles + design tokens
│   └── pages/                  # Auto-routed pages
│       ├── index.vue           # Home (/)
│       ├── developers.vue      # /developers
│       ├── enterprise.vue      # /enterprise
│       ├── features.vue        # /features
│       ├── pricing.vue         # /pricing
│       ├── providers.vue       # /providers
│       └── use-cases.vue       # /use-cases
├── nuxt.config.ts              # Nuxt configuration
├── package.json                # Dependencies
└── wrangler.jsonc              # Cloudflare deployment config
```

## Next Steps

1. **Navigation Component**: Add a global navigation component (import from storybook)
2. **Meta Tags**: Add SEO meta tags using Nuxt's `useHead()` composable
3. **Analytics**: Integrate analytics tracking
4. **Error Pages**: Create custom 404 and error pages
5. **Performance**: Add lazy loading for heavy components
6. **Testing**: Set up Vitest for component testing

## Notes

- All view templates are maintained in `frontend/libs/storybook/stories/templates/`
- Components are exported from `frontend/libs/storybook/stories/index.ts`
- Design tokens are in `frontend/libs/storybook/styles/tokens.css`
- This implementation follows the monorepo workspace pattern with proper boundaries
