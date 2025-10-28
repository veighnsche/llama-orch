# Removal of next-themes Dependency

## Problem
The Tauri app was using `next-themes`, a Next.js-specific package that:
1. Won't work properly in production Tauri builds
2. Adds unnecessary dependencies for a desktop app
3. Relies on Next.js server-side rendering features

## Root Cause
The `ThemeToggle` component from `@rbee/ui/molecules` was hardcoded to use `next-themes`, which was designed for the Next.js commercial website, not for Tauri apps.

## Solution
Created Tauri-compatible theme management using React Context and localStorage.

### Files Created

#### 1. Local ThemeProvider
**File:** `src/components/ThemeProvider.tsx`

```tsx
// Tauri-compatible theme provider using localStorage
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = localStorage.getItem("theme");
    return (stored as Theme) || "system";
  });

  // Apply theme class to <html> element
  // Listen for system theme changes
  // Save to localStorage
}

export function useTheme() {
  return useContext(ThemeContext);
}
```

**Features:**
- ✅ Persists theme preference in localStorage
- ✅ Supports "light", "dark", and "system" modes
- ✅ Listens for system theme changes
- ✅ No server-side dependencies
- ✅ Works in Tauri production builds

#### 2. Local ThemeToggle
**File:** `src/components/ThemeToggle.tsx`

```tsx
// Tauri-compatible theme toggle
import { useTheme } from "./ThemeProvider";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  // Toggle between light and dark
}
```

**Features:**
- ✅ Same API as rbee-ui version
- ✅ Same visual appearance
- ✅ Uses local ThemeProvider instead of next-themes

### Files Modified

#### 1. Main Entry Point
**File:** `src/main.tsx`

```diff
- import { ThemeProvider } from "next-themes";
+ import { ThemeProvider } from "./components/ThemeProvider";

  <ThemeProvider>
    <App />
  </ThemeProvider>
```

#### 2. Commands Sidebar
**File:** `src/components/CommandsSidebar.tsx`

```diff
- import { ThemeToggle } from "@rbee/ui/molecules";
+ import { ThemeToggle } from "./ThemeToggle";
```

#### 3. Package Dependencies
**File:** `package.json`

```diff
  "dependencies": {
-   "next-themes": "^0.4.6",
    "react": "^19.1.1",
```

## How It Works

### Theme Storage
```
localStorage.setItem("theme", "dark")  // User preference
```

### Theme Application
```
document.documentElement.classList.add("dark")  // CSS class on <html>
```

### System Theme Detection
```
window.matchMedia("(prefers-color-scheme: dark)").matches
```

### Theme Switching Flow
1. User clicks theme toggle button
2. `setTheme("light")` called
3. Theme saved to localStorage
4. CSS class updated on `<html>` element
5. Tailwind's dark mode styles apply automatically

## Benefits

### Production-Ready
- ✅ No Next.js dependencies
- ✅ Works in Tauri production builds
- ✅ No server-side rendering issues
- ✅ Smaller bundle size

### Feature Parity
- ✅ Same functionality as next-themes
- ✅ Same visual appearance
- ✅ System theme detection
- ✅ Theme persistence

### Maintainability
- ✅ Simple, self-contained implementation
- ✅ No external dependencies for theming
- ✅ Easy to debug and modify
- ✅ Clear separation from web-specific code

## Testing

Verified with Puppeteer:
- ✅ App loads correctly
- ✅ Theme toggle button appears
- ✅ Clicking toggle switches themes
- ✅ Dark mode → Light mode works
- ✅ Theme persists across reloads (localStorage)

## Future Considerations

### rbee-ui Package Issue
The `@rbee/ui/molecules/ThemeToggle` component is tightly coupled to `next-themes`. This should be refactored to:

1. **Option A:** Make it framework-agnostic
   - Accept theme state as props
   - Let consuming apps provide their own theme context

2. **Option B:** Split into platform-specific versions
   - `ThemeToggle.next.tsx` for Next.js apps
   - `ThemeToggle.react.tsx` for React apps
   - Export the appropriate one based on environment

3. **Option C:** Remove from rbee-ui entirely
   - Theme management is app-specific
   - Each app should implement its own solution

**Recommendation:** Option A - make it accept props so it works everywhere.

## Related Files
- `src/components/ThemeProvider.tsx` (84 lines)
- `src/components/ThemeToggle.tsx` (42 lines)
- `src/main.tsx` (modified)
- `src/components/CommandsSidebar.tsx` (modified)
- `package.json` (next-themes removed)
