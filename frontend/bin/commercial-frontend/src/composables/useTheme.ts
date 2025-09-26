export type ThemeMode = 'light' | 'dark'

const STORAGE_KEY = 'orchyra:theme'

function getStoredTheme(): ThemeMode | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark') return v as ThemeMode
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
  root.setAttribute('data-theme', mode)
  // Keep class parity with ecosystems that rely on .dark/.light
  root.classList.toggle('dark', mode === 'dark')
  root.classList.toggle('light', mode === 'light')
}

export function initThemeFromStorage() {
  const stored = getStoredTheme()
  if (stored) {
    applyTheme(stored)
    return
  }
  // Initial decision from system preference
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  const initial: ThemeMode = prefersDark ? 'dark' : 'light'
  applyTheme(initial)
  // Persist the initial decision so subsequent loads are stable
  setStoredTheme(initial)
}

export function useTheme() {
  function get(): ThemeMode {
    return (getStoredTheme() ?? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) as ThemeMode
  }

  function set(mode: ThemeMode) {
    setStoredTheme(mode)
    applyTheme(mode)
  }

  function toggle() {
    const next: ThemeMode = get() === 'dark' ? 'light' : 'dark'
    set(next)
  }

  return { get, set, toggle }
}
