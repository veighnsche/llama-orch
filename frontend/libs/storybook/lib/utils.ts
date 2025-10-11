// Created by: TEAM-FE-001
import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Merge Tailwind CSS classes with clsx
 * Matches the cn() utility from shadcn/ui React
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
