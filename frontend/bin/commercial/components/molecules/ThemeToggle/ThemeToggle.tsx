'use client'

import * as React from 'react'
import { Moon, Sun } from 'lucide-react'
import { useTheme } from 'next-themes'
import { IconButton } from '@/components/atoms/IconButton/IconButton'

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  // Avoid hydration mismatch
  React.useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <IconButton
        aria-label="Toggle theme"
        title="Toggle theme"
      >
        <Sun className="size-5" aria-hidden />
      </IconButton>
    )
  }

  return (
    <IconButton
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      aria-label="Toggle theme"
      title="Toggle theme"
    >
      {theme === 'dark' ? (
        <Sun className="size-5 transition-transform duration-300" aria-hidden />
      ) : (
        <Moon className="size-5 transition-transform duration-300" aria-hidden />
      )}
    </IconButton>
  )
}
