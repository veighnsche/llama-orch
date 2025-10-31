// TEAM-375: Main entry point
// Import order matches Queen: app CSS first, then UI CSS
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import '@rbee/ui/styles.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
