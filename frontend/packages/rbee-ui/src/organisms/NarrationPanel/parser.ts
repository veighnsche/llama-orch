// TEAM-384: Narration parser - converts raw SSE lines to structured events
// Handles both rbee-keeper (JSON) and rbee-hive (raw text) formats

import type { NarrationEvent } from './types'

/**
 * Parse raw SSE line into structured NarrationEvent
 *
 * Handles two formats:
 *
 * 1. Raw text format (rbee-hive):
 *    ```
 *    lifecycle_local::rebuild::rebuild_daemon rebuild_start
 *    🔄 Rebuilding rbee-hive locally
 *    ```
 *
 * 2. JSON format (rbee-keeper via narration-bridge):
 *    ```json
 *    {
 *      "level": "info",
 *      "human": "🔄 Rebuilding rbee-hive locally",
 *      "action": "rebuild_start",
 *      "fn_name": "lifecycle_local::rebuild::rebuild_daemon",
 *      ...
 *    }
 *    ```
 *
 * @param line - Raw SSE line (text or JSON string)
 * @returns Structured NarrationEvent
 */
export function parseNarrationLine(line: string): NarrationEvent {
  // Try JSON parse first (rbee-keeper format)
  try {
    const json = JSON.parse(line)
    if (json.human || json.message) {
      return {
        level: normalizeLevel(json.level),
        message: json.human || json.message,
        timestamp: json.timestamp || new Date().toISOString(),
        actor: json.actor || null,
        action: json.action || null,
        context: json.context || json.job_id || null,
        human: json.human || json.message,
        fn_name: json.fn_name || null,
        target: json.target || null,
      }
    }
  } catch {
    // Not JSON, continue to text parsing
  }

  // Parse raw text format (rbee-hive)
  const lines = line.split('\n')

  if (lines.length < 2) {
    // Malformed line, return as-is
    return {
      level: 'info',
      message: line,
      timestamp: new Date().toISOString(),
      actor: null,
      action: null,
      context: null,
      human: line,
      fn_name: null,
      target: null,
    }
  }

  const [header, ...messageLines] = lines
  if (!header) {
    return {
      level: 'info',
      message: line,
      timestamp: new Date().toISOString(),
      actor: null,
      action: null,
      context: null,
      human: line,
      fn_name: null,
      target: null,
    }
  }
  const parts = header.split(' ')
  const fn_name = parts[0] || null
  const action = parts[1] || null
  const message = messageLines.join('\n')

  // Extract actor from fn_name (first segment before ::)
  const actor = fn_name?.split('::')[0] || null

  // Detect level from emoji or action
  const level = detectLevel(message, action)

  return {
    level,
    message,
    timestamp: new Date().toISOString(),
    actor,
    action,
    context: null,
    human: message,
    fn_name,
    target: null,
  }
}

/**
 * Normalize level string to valid level type
 */
function normalizeLevel(level?: string): 'error' | 'warn' | 'info' | 'debug' {
  const normalized = level?.toLowerCase()
  switch (normalized) {
    case 'error':
      return 'error'
    case 'warn':
    case 'warning':
      return 'warn'
    case 'debug':
      return 'debug'
    default:
      return 'info'
  }
}

/**
 * Detect log level from message content and action
 *
 * Uses emoji and keywords to infer level:
 * - ❌, Error, error, failed → error
 * - ⚠️, Warning, warn → warn
 * - 🔍, debug → debug
 * - Everything else → info
 */
function detectLevel(message: string, action: string | null): 'error' | 'warn' | 'info' | 'debug' {
  const lowerMessage = message.toLowerCase()
  const lowerAction = action?.toLowerCase() || ''

  // Error indicators
  if (
    message.includes('❌') ||
    lowerMessage.includes('error') ||
    lowerMessage.includes('failed') ||
    lowerAction.includes('error') ||
    lowerAction.includes('failed')
  ) {
    return 'error'
  }

  // Warning indicators
  if (
    message.includes('⚠️') ||
    lowerMessage.includes('warning') ||
    lowerMessage.includes('warn') ||
    lowerAction.includes('warn')
  ) {
    return 'warn'
  }

  // Debug indicators
  if (message.includes('🔍') || lowerAction.includes('debug')) {
    return 'debug'
  }

  // Default to info
  return 'info'
}

/**
 * Extract function name from ANSI-formatted string
 *
 * Format: "\x1b[1mfunction_name\x1b[0m \x1b[2maction\x1b[0m\nmessage"
 * The \x1b[1m...\x1b[0m is the function name in bold
 *
 * Used by rbee-keeper when receiving events via narration-bridge
 */
export function extractFnNameFromFormatted(formatted?: string): string | null {
  if (!formatted) return null

  // Match text between ESC[1m (bold) and ESC[0m (reset)
  const match = formatted.match(/\x1b\[1m([^\x1b]+)\x1b\[0m/)
  return match ? (match[1] ?? null) : null
}
