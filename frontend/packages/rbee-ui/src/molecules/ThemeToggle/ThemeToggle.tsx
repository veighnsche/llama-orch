'use client'

import { IconButton } from '@rbee/ui/atoms/IconButton'
import { Moon, Sun } from 'lucide-react'
import * as React from 'react'

/**
 * Framework-agnostic ThemeToggle component
 * 
 * This component works with any theme provider that follows the standard theme API.
 * It detects the current theme from the DOM and toggles between light and dark modes.
 * 
 * **Next.js with next-themes:**
 * ```tsx
 * import { ThemeProvider } from 'next-themes'
 * import { ThemeToggle } from '@rbee/ui/molecules'
 * 
 * <ThemeProvider attribute="class">
 *   <ThemeToggle />
 * </ThemeProvider>
 * ```
 * 
 * **Tauri/Vite with @rbee/ui/providers:**
 * ```tsx
 * import { ThemeProvider } from '@rbee/ui/providers'
 * import { ThemeToggle } from '@rbee/ui/molecules'
 * 
 * <ThemeProvider>
 *   <ThemeToggle />
 * </ThemeProvider>
 * ```
 */
export function ThemeToggle() {
  const [mounted, setMounted] = React.useState(false)
  const [theme, setTheme] = React.useState<'light' | 'dark'>('light')

  // Avoid hydration mismatch
  React.useEffect(() => {
    setMounted(true)
    
    // Detect initial theme from DOM
    const root = document.documentElement
    const initialTheme = root.classList.contains('dark') ? 'dark' : 'light'
    setTheme(initialTheme)

    // Watch for theme changes (from other sources)
    const observer = new MutationObserver(() => {
      const currentTheme = root.classList.contains('dark') ? 'dark' : 'light'
      setTheme(currentTheme)
    })

    observer.observe(root, {
      attributes: true,
      attributeFilter: ['class'],
    })

    return () => observer.disconnect()
  }, [])

  const toggleTheme = () => {
    const root = document.documentElement
    const newTheme = theme === 'dark' ? 'light' : 'dark'
    
    // Update DOM
    root.classList.remove('light', 'dark')
    root.classList.add(newTheme)
    
    // Update localStorage (works with both next-themes and @rbee/ui/providers)
    localStorage.setItem('theme', newTheme)
    
    // Update state
    setTheme(newTheme)

    // Dispatch custom event for theme providers that listen to it
    window.dispatchEvent(new CustomEvent('theme-change', { detail: { theme: newTheme } }))
  }

  if (!mounted) {
    return (
      <IconButton aria-label="Toggle theme" title="Toggle theme">
        <Sun className="size-5" aria-hidden />
      </IconButton>
    )
  }

  return (
    <IconButton onClick={toggleTheme} aria-label="Toggle theme" title="Toggle theme">
      {theme === 'dark' ? (
        <Sun className="size-5 transition-transform duration-300" aria-hidden />
      ) : (
        <Moon className="size-5 transition-transform duration-300" aria-hidden />
      )}
    </IconButton>
  )
}
