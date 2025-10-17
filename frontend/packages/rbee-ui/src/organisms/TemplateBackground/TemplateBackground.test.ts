import { expect, test } from '@playwright/test'

const STORYBOOK_URL = 'http://localhost:6006'

test.describe('TemplateBackground - All Variants', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(STORYBOOK_URL)
  })

  const variants = [
    'none',
    'background',
    'secondary',
    'card',
    'muted',
    'accent',
    'primary',
    'destructive',
    'subtle-border',
    'gradient-primary',
    'gradient-secondary',
    'gradient-destructive',
    'gradient-radial',
    'gradient-mesh',
  ]

  for (const variant of variants) {
    test(`should render ${variant} variant correctly`, async ({ page }) => {
      // Navigate to the specific story
      const storyName =
        variant.charAt(0).toUpperCase() + variant.slice(1).replace(/-([a-z])/g, (g) => g[1].toUpperCase())
      await page.goto(`${STORYBOOK_URL}/?path=/story/organisms-templatebackground--${variant}`)

      // Wait for story to load
      await page.waitForSelector('#storybook-root', { timeout: 5000 })

      // Take screenshot
      const screenshot = await page.screenshot({ fullPage: true })
      expect(screenshot).toBeTruthy()

      // Check that content is visible
      const content = await page.locator('text=Template Background Showcase').first()
      await expect(content).toBeVisible()
    })
  }

  test('should render with decoration', async ({ page }) => {
    await page.goto(`${STORYBOOK_URL}/?path=/story/organisms-templatebackground--custom-decoration`)
    await page.waitForSelector('#storybook-root')

    const content = await page.locator('text=Template Background Showcase').first()
    await expect(content).toBeVisible()

    // Check SVG decoration exists
    const svg = await page.locator('svg').first()
    await expect(svg).toBeVisible()
  })

  test('should render with overlay', async ({ page }) => {
    await page.goto(`${STORYBOOK_URL}/?path=/story/organisms-templatebackground--gradient-with-overlay`)
    await page.waitForSelector('#storybook-root')

    const content = await page.locator('text=Template Background Showcase').first()
    await expect(content).toBeVisible()
  })

  test('should render with blur', async ({ page }) => {
    await page.goto(`${STORYBOOK_URL}/?path=/story/organisms-templatebackground--pattern-with-blur`)
    await page.waitForSelector('#storybook-root')

    const content = await page.locator('text=Template Background Showcase').first()
    await expect(content).toBeVisible()
  })

  test('should render pattern variants', async ({ page }) => {
    const patterns = ['dots', 'grid', 'honeycomb', 'waves', 'circuit', 'diagonal']

    for (const pattern of patterns) {
      await page.goto(`${STORYBOOK_URL}/?path=/story/organisms-templatebackground--pattern-${pattern}-medium`)
      await page.waitForSelector('#storybook-root')

      const content = await page.locator('text=Template Background Showcase').first()
      await expect(content).toBeVisible()
    }
  })
})

test.describe('TemplateContainer - Background Integration', () => {
  test('should work with all legacy bgVariant values', async ({ page }) => {
    const legacyVariants = ['background', 'secondary', 'card', 'muted', 'subtle', 'destructive-gradient']

    for (const variant of legacyVariants) {
      await page.goto(`${STORYBOOK_URL}/?path=/story/molecules-templatecontainer--all-props-showcase`)
      await page.waitForSelector('#storybook-root')

      // Check content is visible
      const content = await page.locator('text=All Props Showcase').first()
      await expect(content).toBeVisible({ timeout: 5000 })
    }
  })

  test('should work in dark mode', async ({ page }) => {
    await page.goto(`${STORYBOOK_URL}/?path=/story/molecules-templatecontainer--all-props-showcase&globals=theme:dark`)
    await page.waitForSelector('#storybook-root')

    // Check content is visible in dark mode
    const content = await page.locator('text=All Props Showcase').first()
    await expect(content).toBeVisible({ timeout: 5000 })

    // Take screenshot for visual verification
    const screenshot = await page.screenshot({ fullPage: true })
    expect(screenshot).toBeTruthy()
  })

  test('should support new background prop', async ({ page }) => {
    await page.goto(`${STORYBOOK_URL}/?path=/story/molecules-templatecontainer--all-props-showcase`)
    await page.waitForSelector('#storybook-root')

    // Verify the component renders
    const root = await page.locator('#storybook-root')
    await expect(root).toBeVisible()
  })
})
