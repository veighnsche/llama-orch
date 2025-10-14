# Nextra v4 Setup - RESOLVED ✅

## Problem Summary

The user-docs app had Nextra v4 installed but **the theme layout was not rendering**. The MDX content displayed as unstyled HTML without the Nextra docs theme UI (no navbar, sidebar, search, theme switcher, etc.).

## Solution

**Root Cause**: Nextra v4 is designed for Next.js App Router, not Pages Router. The initial setup attempted to use Pages Router which is incompatible with Nextra v4's architecture.

## Current State

### What Works
- ✅ Next.js 15 dev server runs on port 3100
- ✅ MDX files compile and render content
- ✅ Tailwind CSS v4 is installed and working
- ✅ Dependencies installed: `nextra@4.6.0`, `nextra-theme-docs@4.6.0`
- ✅ Build completes successfully

### What Now Works ✅
- ✅ **Nextra theme layout fully applied** - pages render with complete Nextra UI
- ✅ **pageMap auto-generated** - Nextra's automatic page navigation structure working
- ✅ **All theme UI components** - navbar, sidebar, footer, theme toggle, table of contents
- ✅ **CSS fully loading** - `nextra-theme-docs/style.css` properly applied
- ✅ **Navigation working** - sidebar navigation and routing functional
- ✅ **Theme switcher** - Light/Dark/System theme toggle working
- ✅ **Code syntax highlighting** - Code blocks properly highlighted

## Technical Details

### Current Setup

**File Structure:**
```
frontend/bin/user-docs/
├── app/
│   ├── layout.tsx                    # Root layout with Nextra CSS
│   ├── page.tsx                      # Redirects to /docs
│   └── docs/
│       ├── layout.tsx                # Docs layout with Nextra theme
│       ├── _meta.ts                  # Navigation metadata
│       ├── page.mdx                  # Docs homepage
│       ├── getting-started/
│       │   └── page.mdx
│       └── guide/
│           ├── _meta.ts
│           ├── overview/
│           │   └── page.mdx
│           └── deployment/
│               └── page.mdx
├── mdx-components.tsx                # MDX components integration
├── next.config.ts                    # Wrapped with nextra()
└── package.json
```

**`next.config.ts`:**
```typescript
import nextra from "nextra";

const withNextra = nextra({
  defaultShowCopyCode: true,
  search: { codeblocks: false },
});

const nextConfig: NextConfig = {
  images: { unoptimized: true }, // Cloudflare Workers compatibility
};

export default withNextra(nextConfig);
```

**`app/layout.tsx` (Root):**
```typescript
import type { Metadata } from "next";
import "nextra-theme-docs/style.css";

export const metadata: Metadata = {
  title: "rbee Documentation",
  description: "Documentation for rbee - Private LLM Hosting in the Netherlands",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
```

**`app/docs/layout.tsx` (Docs Layout with Nextra Theme):**
```typescript
import { Layout } from 'nextra-theme-docs';
import { getPageMap } from 'nextra/page-map';

export default async function DocsLayout({ children }: { children: React.ReactNode }) {
  const pageMap = await getPageMap('/docs');
  
  return (
    <Layout
      pageMap={pageMap}
      docsRepositoryBase="https://github.com/veighnsche/llama-orch/tree/main/frontend/bin/user-docs"
      sidebar={{ defaultMenuCollapseLevel: 1 }}
      footer={
        <span>
          {new Date().getFullYear()} © rbee. Private LLM Hosting in the Netherlands.
        </span>
      }
    >
      {children}
    </Layout>
  );
}
```

**`app/docs/_meta.ts` (Navigation):**
```typescript
export default {
  index: 'Welcome',
  'getting-started': 'Getting Started',
  guide: 'Guide'
}
```

**`mdx-components.tsx`:**
```typescript
import { useMDXComponents as getDocsComponents } from 'nextra-theme-docs';

export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    ...components,
  };
}
```

## Implementation Steps Taken

### 1. Migrated from Pages Router to App Router
- Moved all MDX content from `pages/` to `app/docs/`
- Restructured to use Next.js 15 App Router file conventions (`page.mdx`, `layout.tsx`)
- Deleted obsolete Pages Router files (`pages/_app.tsx`, `theme.config.tsx`)

### 2. Created Proper Nextra Layout
- Created `app/docs/layout.tsx` that imports `Layout` from `nextra-theme-docs`
- Used `getPageMap('/docs')` to generate navigation structure
- Configured theme options directly in the Layout component props

### 3. Set Up Navigation Metadata
- Created `_meta.ts` files for navigation structure:
  - `app/docs/_meta.ts` for top-level navigation
  - `app/docs/guide/_meta.ts` for nested guide pages
- Nextra automatically generates sidebar from these files

### 4. Configured MDX Components
- Created `mdx-components.tsx` that integrates Nextra's MDX components
- Ensures all Nextra features (callouts, code blocks, etc.) work correctly

### 5. Updated Root Layout
- Modified `app/layout.tsx` to import Nextra CSS
- Added `suppressHydrationWarning` for theme switcher compatibility

## Key Learnings

1. **Nextra v4 requires App Router** - It is not compatible with Pages Router
2. **Layout component needs pageMap** - Generated via `getPageMap()` from `nextra/page-map`
3. **_meta.ts files drive navigation** - Simple object exports define sidebar structure
4. **Theme config is passed as props** - Not a separate `theme.config.tsx` file
5. **File naming matters** - Must use `page.mdx` not `index.mdx` in App Router

## Design Cohesion Notes

The commercial site uses honeycomb yellow (hue: 45) as the primary color. This should be applied to Nextra via custom CSS or theme configuration. Future work should:

1. Extract shared design tokens to `frontend/libs/shared-components`
2. Apply custom Nextra theming to match commercial site colors
3. Ensure consistent typography across both applications

## Success Criteria - Status

- [x] Nextra theme layout renders (navbar, sidebar, footer visible)
- [x] Navigation works (sidebar shows all pages from `_meta.ts`)
- [x] Theme switcher works (light/dark/system mode)
- [x] Code syntax highlighting works
- [ ] Search functionality configured (requires additional setup)
- [ ] Design fully cohesive with commercial site (needs custom theming)
- [x] Build succeeds without errors
- [x] Dev server runs on port 3100

---

**Created**: 2025-10-14  
**Resolved**: 2025-10-14  
**Status**: ✅ RESOLVED - Nextra v4 theme fully functional with App Router
