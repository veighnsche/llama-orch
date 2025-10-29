/**
 * TEAM-356: Singleflight pattern - ensure only one load at a time
 * 
 * Prevents multiple concurrent loads of the same SDK package.
 * If multiple callers request the same package, only one load executes
 * and all callers receive the same result.
 */

import type { GlobalSlot } from './types'

const GLOBAL_SLOTS = new Map<string, GlobalSlot<any>>()

/**
 * Get or create global slot for package
 * 
 * @param packageName - Package name to get slot for
 * @returns Global slot for the package
 */
export function getGlobalSlot<T>(packageName: string): GlobalSlot<T> {
  if (!GLOBAL_SLOTS.has(packageName)) {
    GLOBAL_SLOTS.set(packageName, {})
  }
  return GLOBAL_SLOTS.get(packageName)!
}

/**
 * Clear global slot (for testing)
 * 
 * @param packageName - Package name to clear
 */
export function clearGlobalSlot(packageName: string): void {
  GLOBAL_SLOTS.delete(packageName)
}

/**
 * Clear all global slots (for testing)
 */
export function clearAllGlobalSlots(): void {
  GLOBAL_SLOTS.clear()
}
