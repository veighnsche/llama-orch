/**
 * TEAM-356: Vitest configuration for @rbee/sdk-loader
 */
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: ['**/*.test.ts', '**/dist/**', '**/node_modules/**'],
    },
  },
})
