# Next.js to React/Vite Migration Guide

## Quick Reference: What Changed

### Package.json Scripts
```json
// Next.js (web-ui.old)
{
  "dev": "next dev -p 3002",
  "build": "next build",
  "start": "next start"
}

// React/Vite (web-ui)
{
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview"
}
```

### Imports

#### Routing
```tsx
// ❌ Next.js
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';

// ✅ React Router
import { useNavigate, useLocation, Link } from 'react-router-dom';
```

#### Client Directives
```tsx
// ❌ Next.js - needed at top of file
'use client';

// ✅ React/Vite - not needed (everything is client-side)
```

### Routing Patterns

#### Navigation
```tsx
// ❌ Next.js
<Link href="/dashboard">Dashboard</Link>

// ✅ React Router
<Link to="/dashboard">Dashboard</Link>
```

#### Programmatic Navigation
```tsx
// ❌ Next.js
const router = useRouter();
router.push('/dashboard');

// ✅ React Router
const navigate = useNavigate();
navigate('/dashboard');
```

#### Current Route
```tsx
// ❌ Next.js
const pathname = usePathname();
const isActive = pathname === '/dashboard';

// ✅ React Router
const location = useLocation();
const isActive = location.pathname === '/dashboard';
```

### File Structure

#### Next.js (App Router)
```
src/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Home page (/)
│   ├── dashboard/
│   │   └── page.tsx        # /dashboard
│   ├── keeper/
│   │   └── page.tsx        # /keeper
│   └── globals.css
└── components/
```

#### React/Vite
```
src/
├── App.tsx                 # Root component with routes
├── main.tsx                # Entry point
├── globals.css
├── pages/
│   ├── DashboardPage.tsx   # /dashboard
│   ├── KeeperPage.tsx      # /keeper
│   ├── SettingsPage.tsx    # /settings
│   └── HelpPage.tsx        # /help
├── components/
├── hooks/
└── stores/
```

### Layout Pattern

#### Next.js
```tsx
// app/layout.tsx
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <SidebarProvider>
            <AppSidebar />
            <SidebarInset>{children}</SidebarInset>
          </SidebarProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
```

#### React/Vite
```tsx
// App.tsx
function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <SidebarProvider>
          <AppSidebar />
          <SidebarInset>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              {/* ... */}
            </Routes>
          </SidebarInset>
        </SidebarProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}
```

### CSS Imports

#### Next.js
```tsx
// app/layout.tsx
import "./globals.css";
import "@rbee/ui/styles.css";
```

#### React/Vite
```tsx
// main.tsx
import './globals.css';
import '@rbee/ui/styles.css';
```

### Metadata

#### Next.js
```tsx
// app/layout.tsx
export const metadata: Metadata = {
  title: "rbee Web UI",
  description: "Dashboard for managing rbee infrastructure",
};
```

#### React/Vite
```html
<!-- index.html -->
<head>
  <title>rbee Web UI</title>
</head>
```

### Environment Variables

#### Next.js
```
NEXT_PUBLIC_API_URL=http://localhost:8500
```

#### Vite
```
VITE_API_URL=http://localhost:8500
```

Access in code:
```tsx
// ❌ Next.js
const apiUrl = process.env.NEXT_PUBLIC_API_URL;

// ✅ Vite
const apiUrl = import.meta.env.VITE_API_URL;
```

## Migration Checklist

- [x] Install React Router dependencies
- [x] Create `vite.config.ts`
- [x] Create `postcss.config.js` for Tailwind
- [x] Update `package.json` scripts
- [x] Convert `app/layout.tsx` → `App.tsx` with `<BrowserRouter>`
- [x] Convert `app/page.tsx` → redirect route
- [x] Convert `app/*/page.tsx` → `pages/*Page.tsx`
- [x] Update all `usePathname()` → `useLocation()`
- [x] Update all `useRouter()` → `useNavigate()`
- [x] Update all `<Link href>` → `<Link to>`
- [x] Remove all `'use client'` directives
- [x] Update CSS imports in `main.tsx`
- [x] Test all routes
- [x] Test theme switching
- [x] Test heartbeat monitoring
- [x] Build for production

## Common Gotchas

### 1. WASM Support
React/Vite requires `vite-plugin-wasm` for WASM modules:
```ts
// vite.config.ts
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  plugins: [wasm()],
});
```

### 2. CSS Minification
Tailwind v4 arbitrary values may break with lightningcss:
```ts
// vite.config.ts
export default defineConfig({
  build: {
    cssMinify: false,
  },
});
```

### 3. Package Exports
TypeScript needs explicit `.ts` extensions in package.json exports:
```json
{
  "exports": {
    "./hooks/*": "./src/hooks/*.ts"  // ← .ts extension required
  }
}
```

### 4. Redirects
Next.js automatic redirects need manual implementation:
```tsx
// ❌ Next.js (automatic)
// app/page.tsx redirects to /dashboard via router.push()

// ✅ React Router (explicit)
<Route path="/" element={<Navigate to="/dashboard" replace />} />
```

## Performance Comparison

### Build Time
- **Next.js:** ~8-12s (with optimization)
- **Vite:** ~6s (with Rolldown)

### Dev Server Startup
- **Next.js:** ~2-3s
- **Vite:** ~400ms

### Bundle Size
- **Next.js:** Similar (depends on optimization)
- **Vite:** 373 KB JS + 293 KB CSS (gzipped: 119 KB + 36 KB)

## Benefits of React/Vite

1. **Faster Development:** HMR is instant with Vite
2. **Simpler Routing:** No file-based routing complexity
3. **Smaller Runtime:** No Next.js framework overhead
4. **Better WASM Support:** Native WASM plugin
5. **Modern Tooling:** Rolldown bundler (Rust-based)

## Drawbacks

1. **No SSR:** Client-side only (not needed for dashboard)
2. **No API Routes:** Need separate backend (already have queen-rbee)
3. **Manual SEO:** No automatic meta tags (not needed for dashboard)

## Conclusion

For rbee Web UI (internal dashboard), React/Vite is the better choice:
- ✅ Faster development experience
- ✅ Simpler architecture
- ✅ Better WASM support
- ✅ No SSR needed (authenticated dashboard)
- ✅ Smaller bundle size
