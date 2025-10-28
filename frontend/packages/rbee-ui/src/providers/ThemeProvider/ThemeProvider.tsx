/**
 * Framework-agnostic theme provider
 * Compatible with Next.js (next-themes) and other React environments (Tauri, Vite, etc.)
 *
 * Usage:
 * - Next.js: Use next-themes ThemeProvider (already installed)
 * - Tauri/Vite: Use this ThemeProvider
 */

import { createContext, type ReactNode, useContext, useEffect, useState } from 'react'

type Theme = 'light' | 'dark' | 'system'

interface ThemeContextType {
  theme: Theme
  setTheme: (theme: Theme) => void
  resolvedTheme: 'light' | 'dark'
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export interface ThemeProviderProps {
  children: ReactNode
  defaultTheme?: Theme
  storageKey?: string
  attribute?: string
}

/**
 * Universal ThemeProvider for non-Next.js environments
 *
 * @example
 * // Tauri/Vite app
 * import { ThemeProvider } from '@rbee/ui/providers'
 *
 * <ThemeProvider>
 *   <App />
 * </ThemeProvider>
 */
export function ThemeProvider({
  children,
  defaultTheme = 'system',
  storageKey = 'theme',
  attribute = 'class',
}: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window === 'undefined') return defaultTheme
    const stored = localStorage.getItem(storageKey)
    return (stored as Theme) || defaultTheme
  })

  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('dark')

  useEffect(() => {
    const root = document.documentElement

    // Determine the actual theme to apply
    let effectiveTheme: 'light' | 'dark'

    if (theme === 'system') {
      effectiveTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    } else {
      effectiveTheme = theme
    }

    setResolvedTheme(effectiveTheme)

    // Apply theme attribute
    if (attribute === 'class') {
      root.classList.remove('light', 'dark')
      root.classList.add(effectiveTheme)
    } else {
      root.setAttribute(attribute, effectiveTheme)
    }

    // Save to localStorage
    localStorage.setItem(storageKey, theme)
  }, [theme, storageKey, attribute])

  // Listen for system theme changes
  useEffect(() => {
    if (theme !== 'system') return

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = (e: MediaQueryListEvent) => {
      const newTheme = e.matches ? 'dark' : 'light'
      setResolvedTheme(newTheme)
      const root = document.documentElement

      if (attribute === 'class') {
        root.classList.remove('light', 'dark')
        root.classList.add(newTheme)
      } else {
        root.setAttribute(attribute, newTheme)
      }
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [theme, attribute])

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
  }

  return <ThemeContext.Provider value={{ theme, setTheme, resolvedTheme }}>{children}</ThemeContext.Provider>
}

/**
 * Hook to access theme context
 * Works with both next-themes and universal ThemeProvider
 */
export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}
