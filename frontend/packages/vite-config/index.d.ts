// TEAM-294: Type declarations for @repo/vite-config
import type { UserConfig } from 'vite';

/**
 * Create a Vite config for React + Tailwind apps
 * @param overrides - Optional config overrides
 * @returns Vite UserConfig
 */
export function createViteConfig(overrides?: UserConfig): UserConfig;
