# Font Loading Fixed - Using CDN Instead of Local Files

## Problem

The initial approach of loading fonts from `node_modules` failed because:

**Error:**
```
Module not found: Can't resolve '../../node_modules/geist/dist/fonts/geist-sans/Geist-Variable.woff2'
```

**Root Cause:**
- Next.js (Webpack) couldn't resolve relative paths from compiled CSS (`dist/index.css`) to `node_modules`
- The path `../../node_modules/geist/...` was correct from source but broke after CSS compilation
- Different bundlers (Next.js Webpack vs Vite for Storybook) handle paths differently

## Solution

**Use CDN for font files instead of local node_modules paths.**

### Changes Made

**File:** `packages/rbee-ui/src/tokens/fonts.css`

**Before (BROKEN):**
```css
@font-face {
    font-family: "Geist Sans";
    src: url("../../node_modules/geist/dist/fonts/geist-sans/Geist-Variable.woff2") format("woff2");
    /* ❌ Path breaks after CSS compilation */
}
```

**After (WORKING):**
```css
@font-face {
    font-family: "Geist Sans";
    src: url("https://cdn.jsdelivr.net/npm/geist@1.5.1/dist/fonts/geist-sans/Geist-Variable.woff2") format("woff2");
    /* ✅ CDN URL works everywhere */
}

@font-face {
    font-family: "Geist Mono";
    src: url("https://cdn.jsdelivr.net/npm/geist@1.5.1/dist/fonts/geist-mono/GeistMono-Variable.woff2") format("woff2");
    /* ✅ CDN URL works everywhere */
}
```

## Why This Works

### 1. No Path Resolution Issues
- CDN URLs are absolute, not relative
- Works identically in Next.js, Storybook, and any bundler
- No dependency on file system structure

### 2. Version Locked
- Using `geist@1.5.1` in URL matches package.json dependency
- Fonts won't change unexpectedly
- Can update version when upgrading package

### 3. Performance Benefits
- CDN caching across sites
- Parallel downloads (not blocked by main bundle)
- jsDelivr has global CDN edge nodes

### 4. Works Everywhere
- ✅ Next.js (Webpack)
- ✅ Storybook (Vite)
- ✅ Any future bundler
- ✅ No build configuration needed

## Font Loading Strategy

All three fonts now use external sources:

```css
/* Geist Sans - from jsDelivr CDN */
@font-face {
    font-family: "Geist Sans";
    src: url("https://cdn.jsdelivr.net/npm/geist@1.5.1/dist/fonts/geist-sans/Geist-Variable.woff2");
}

/* Geist Mono - from jsDelivr CDN */
@font-face {
    font-family: "Geist Mono";
    src: url("https://cdn.jsdelivr.net/npm/geist@1.5.1/dist/fonts/geist-mono/GeistMono-Variable.woff2");
}

/* Source Serif 4 - from Google Fonts */
@import url("https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&display=swap");
```

## Verification

Build succeeded:
```bash
cd packages/rbee-ui
pnpm build:styles
# ✅ Success - no module resolution errors
```

Both environments now work:
```bash
# Storybook
cd packages/rbee-ui && pnpm storybook
# ✅ Fonts load from CDN

# Commercial app
cd apps/commercial && pnpm dev
# ✅ Fonts load from CDN
```

## Trade-offs

### Pros
- ✅ Works in all bundlers without configuration
- ✅ No path resolution issues
- ✅ CDN caching and performance
- ✅ Simple and maintainable

### Cons
- ⚠️ Requires internet connection (not an issue for web apps)
- ⚠️ External dependency on jsDelivr (highly reliable CDN)

### Why This is Acceptable

1. **Web apps require internet anyway** - Users are already online
2. **jsDelivr is extremely reliable** - 99.9% uptime, used by millions of sites
3. **Fallback fonts work** - CSS variables include system font fallbacks
4. **Version locked** - Won't break unexpectedly

## Alternative Considered: next/font

**Why NOT use next/font:**
- Only works in Next.js, not Storybook
- Violates single source of truth principle
- Would require duplication across apps
- Doesn't work with our centralized rbee-ui approach

## Files Changed

1. `packages/rbee-ui/src/tokens/fonts.css` - Changed to CDN URLs
2. `packages/rbee-ui/dist/index.css` - Rebuilt with CDN URLs

## Related Documentation

- Font centralization: `frontend/FONT_CENTRALIZATION_COMPLETE.md`
- Original serif fix: `frontend/FONT_SYSTEM_FIX_COMPLETE.md`
- Font analysis: `packages/rbee-ui/FONT_FIX_ANALYSIS.md`
