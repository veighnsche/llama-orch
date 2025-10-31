// TEAM-375: Generic parent ↔ iframe communication utilities

import { createMessageReceiver } from './receiver'
import type { IframeMessage } from './types'

/**
 * PARENT SIDE: Broadcast message to all iframes
 * Generic function - not tied to any specific message type
 */
export function broadcastToIframes(message: IframeMessage) {
  const iframes = document.querySelectorAll('iframe')
  
  iframes.forEach((iframe) => {
    if (iframe.contentWindow) {
      iframe.contentWindow.postMessage(message, '*')
    }
  })
}

/**
 * CHILD SIDE: Setup message receiver from parent
 * Generic receiver - handles any message type
 * Returns cleanup function
 */
export function receiveFromParent(
  onMessage: (message: IframeMessage) => void,
  options?: { allowedOrigins?: string[], debug?: boolean }
): () => void {
  return createMessageReceiver({
    allowedOrigins: options?.allowedOrigins || ['*'],
    debug: options?.debug,
    onMessage,
  })
}

// ============================================================================
// THEME-SPECIFIC HELPERS (built on top of generic utilities)
// ============================================================================

/**
 * PARENT SIDE: Watch for theme changes and broadcast to iframes
 * Returns cleanup function
 */
export function broadcastThemeChanges(): () => void {
  const root = document.documentElement
  
  const sendTheme = () => {
    const theme = root.classList.contains('dark') ? 'dark' : 'light'
    broadcastToIframes({
      type: 'THEME_CHANGE',
      source: 'keeper',
      timestamp: Date.now(),
      theme,
    })
  }
  
  // Send initial theme
  sendTheme()
  
  // Watch for changes
  const observer = new MutationObserver(sendTheme)
  observer.observe(root, {
    attributes: true,
    attributeFilter: ['class'],
  })
  
  return () => observer.disconnect()
}

/**
 * CHILD SIDE: Receive theme changes from parent
 * Returns cleanup function
 */
export function receiveThemeChanges(): () => void {
  return receiveFromParent((msg) => {
    if (msg.type === 'THEME_CHANGE' && msg.source === 'keeper') {
      const root = document.documentElement
      root.classList.remove('light', 'dark')
      root.classList.add(msg.theme)
      // TEAM-375: Save to localStorage so next page load uses correct theme
      localStorage.setItem('theme', msg.theme)
    }
  })
}
