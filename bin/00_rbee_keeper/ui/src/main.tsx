// TEAM-295: Main entry point
// Import order: globals.css (Tailwind + theme), then UI components, then App

import { ThemeProvider } from '@rbee/ui/providers'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { TauriProvider } from './contexts/TauriContext'
import './globals.css'
import '@rbee/ui/styles.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TauriProvider>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </TauriProvider>
  </StrictMode>,
)
