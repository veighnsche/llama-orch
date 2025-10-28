// TEAM-334: Custom titlebar for rbee-keeper
// Replaces native window decorations with React component
// Provides window controls (minimize, maximize, close) and drag area

import { Button } from '@rbee/ui/atoms'
import { BrandLogo } from '@rbee/ui/molecules'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { AlertCircle, Minus, Square, X } from 'lucide-react'
import { Component, type ReactNode } from 'react'
import { useTauri } from '../contexts/TauriContext'

// Error boundary for CustomTitlebar
class CustomTitlebarErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('CustomTitlebar error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI - minimal titlebar
      return (
        <div
          data-tauri-drag-region
          className="h-10 bg-background border-b border-border flex items-center justify-between px-3 select-none"
        >
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <AlertCircle className="h-4 w-4" />
            <span>rbee</span>
          </div>
          <div className="text-xs text-destructive">Titlebar error: {this.state.error?.message}</div>
        </div>
      )
    }

    return this.props.children
  }
}

function CustomTitlebarContent() {
  const { isTauri } = useTauri()

  // Only get window if in Tauri environment
  const appWindow = isTauri ? getCurrentWindow() : null

  const handleMinimize = () => {
    appWindow?.minimize()
  }

  const handleMaximize = () => {
    appWindow?.toggleMaximize()
  }

  const handleClose = () => {
    appWindow?.close()
  }

  return (
    <div
      data-tauri-drag-region
      className="h-10 bg-background border-b border-border flex items-center justify-between px-3 select-none"
    >
      {/* Left side - Brand logo */}
      <div className="flex items-center gap-2">
        <BrandLogo size="sm" />
        {!isTauri && <span className="text-xs text-muted-foreground">(Browser Mode)</span>}
      </div>

      {/* Right side - Window controls (only in Tauri) */}
      {isTauri && (
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon-sm" onClick={handleMinimize} aria-label="Minimize">
            <Minus />
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={handleMaximize} aria-label="Maximize">
            <Square className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={handleClose}
            aria-label="Close"
            className="hover:bg-destructive hover:text-destructive-foreground"
          >
            <X />
          </Button>
        </div>
      )}
    </div>
  )
}

export function CustomTitlebar() {
  return (
    <CustomTitlebarErrorBoundary>
      <CustomTitlebarContent />
    </CustomTitlebarErrorBoundary>
  )
}
