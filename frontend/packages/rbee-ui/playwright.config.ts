import { defineConfig, devices } from '@playwright/experimental-ct-react'

/**
 * CRITICAL: DO NOT run `pnpm test:ct` with blocking=true!
 * It opens an interactive HTML report server that hangs.
 *
 * CORRECT usage:
 * 1. Run in background: `pnpm test:ct &` (capture PID)
 * 2. Or run to file: `pnpm test:ct > test.log 2>&1`
 * 3. Then check results: `cat test-results/.last-run.json`
 *
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: './src',
  testMatch: '**/*.spec.tsx',
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: 'html',

  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',

    /* Port to use for Playwright component testing. */
    ctPort: 3100,
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
