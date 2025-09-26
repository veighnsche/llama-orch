export type ThemeMode = 'system' | 'light' | 'dark'

const STORAGE_KEY = 'orchyra:theme'

function getStoredTheme(): ThemeMode | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
    return null
  } catch {
    return null
  }
}

function setStoredTheme(mode: ThemeMode) {
  try {
    localStorage.setItem(STORAGE_KEY, mode)
  } catch {
    // ignore
  }
}

function applyTheme(mode: ThemeMode) {
  const root = document.documentElement
  if (mode === 'light' || mode === 'dark') {
    root.setAttribute('data-theme', mode)
    // Keep class parity with Histoire preview and other ecosystems (e.g., Tailwind)
    root.classList.toggle('dark', mode === 'dark')
    root.classList.toggle('light', mode === 'light')
  } else {
    // system: remove explicit override so @media fallback takes effect
    root.removeAttribute('data-theme')
    root.classList.remove('dark', 'light')
  }
}

export function initThemeFromStorage() {
  const stored = getStoredTheme()
  applyTheme(stored ?? 'system')
}

export function useTheme() {
  const mql = window.matchMedia('(prefers-color-scheme: dark)')

  function get(): ThemeMode {
    return (getStoredTheme() ?? 'system') as ThemeMode
  }

  function set(mode: ThemeMode) {
    setStoredTheme(mode)
    applyTheme(mode)
  }

  function effective(): 'light' | 'dark' {
    const mode = get()
    if (mode === 'dark') return 'dark'
    if (mode === 'light') return 'light'
    return mql.matches ? 'dark' : 'light'
  }

  // Optional: let callers subscribe to system changes when in system mode
  function onSystemChange(cb: (isDark: boolean) => void) {
    const handler = (e: MediaQueryListEvent) => {
      if (get() === 'system') cb(e.matches)
    }
    try {
      mql.addEventListener('change', handler)
      return () => mql.removeEventListener('change', handler)
    } catch {
      // Safari < 14
      // @ts-expect-error
      mql.addListener && mql.addListener(handler)
      return () => {
        // @ts-expect-error
        mql.removeListener && mql.removeListener(handler)
      }
    }
  }

  return { get, set, effective, onSystemChange }
}
