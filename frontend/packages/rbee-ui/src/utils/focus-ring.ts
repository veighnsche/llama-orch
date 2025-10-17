/**
 * Unified focus ring utility for consistent focus states across all interactive atoms.
 * 
 * Uses CSS variables from theme-tokens.css:
 * - --focus-ring-color
 * - --focus-ring-width
 * - --focus-ring-offset
 * - --bg-canvas (for ring offset color)
 * 
 * Apply to: Button, Input, Textarea, Select, Checkbox, Radio, Switch, 
 *           Tabs.Trigger, MenuItem, Link, and any focusable element.
 */

export const focusRing = 
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"

/**
 * Variant for elements that need tighter focus (e.g., small icons, compact controls)
 */
export const focusRingTight = 
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background"

/**
 * Variant for destructive actions (uses destructive color instead of brand)
 */
export const focusRingDestructive = 
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-destructive focus-visible:ring-offset-2 focus-visible:ring-offset-background"

/**
 * Brand-consistent link styling with amber underline and focus states
 * Apply to: Button link variant, NavLink, inline CTAs
 */
export const brandLink = 
  "text-[color:var(--primary)] underline underline-offset-2 decoration-amber-300 hover:text-[color:var(--accent)] hover:decoration-amber-400 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
