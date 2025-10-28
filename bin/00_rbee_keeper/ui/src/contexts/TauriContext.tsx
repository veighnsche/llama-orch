// Context to track if app is running in Tauri environment
import { createContext, type ReactNode, useContext } from 'react'

interface TauriContextValue {
  isTauri: boolean
}

const TauriContext = createContext<TauriContextValue | undefined>(undefined)

export function TauriProvider({ children }: { children: ReactNode }) {
  // Check if running in Tauri environment (only once at mount)
  const isTauri = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window

  return <TauriContext.Provider value={{ isTauri }}>{children}</TauriContext.Provider>
}

export function useTauri() {
  const context = useContext(TauriContext)
  if (context === undefined) {
    throw new Error('useTauri must be used within a TauriProvider')
  }
  return context
}
