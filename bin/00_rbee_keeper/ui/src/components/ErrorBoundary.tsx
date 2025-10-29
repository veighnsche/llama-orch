// TEAM-350: Error boundary to catch and display React errors gracefully
// Prevents the entire app from crashing when a component errors

import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Alert, AlertDescription, AlertTitle } from '@rbee/ui/atoms'
import { AlertCircle } from 'lucide-react'

interface Props {
  children: ReactNode
  fallbackMessage?: string
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    }
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary] Caught error:', error, errorInfo)
    this.setState({ error, errorInfo })
  }

  render() {
    if (this.state.hasError) {
      return (
        <Alert variant="destructive" className="m-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Component Error</AlertTitle>
          <AlertDescription>
            <div className="space-y-2">
              <p>
                {this.props.fallbackMessage || 'Something went wrong in this component.'}
              </p>
              {this.state.error && (
                <details className="text-xs">
                  <summary className="cursor-pointer font-semibold">Error Details</summary>
                  <pre className="mt-2 p-2 bg-muted rounded overflow-auto">
                    {this.state.error.toString()}
                    {this.state.errorInfo?.componentStack}
                  </pre>
                </details>
              )}
            </div>
          </AlertDescription>
        </Alert>
      )
    }

    return this.props.children
  }
}
