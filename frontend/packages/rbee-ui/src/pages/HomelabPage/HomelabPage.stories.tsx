import type { Meta, StoryObj } from '@storybook/react'
import HomelabPage from './HomelabPage'

const meta = {
  title: 'Pages/HomelabPage',
  component: HomelabPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Homelab & Self-Hosting page targeting homelab enthusiasts, self-hosters, and privacy advocates. Showcases SSH-based orchestration, multi-backend support, and complete privacy.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof HomelabPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Homelab page with all sections:
 * - Hero with network topology visualization
 * - Email capture for setup guide
 * - Problem section (homelab complexity)
 * - Solution section (unified orchestration)
 * - How It Works (4-step setup guide)
 * - Cross-Node Orchestration visualization
 * - Multi-Backend GPU support
 * - Power cost calculator
 * - Use cases (single PC, multi-node, hybrid)
 * - Security & privacy features
 * - FAQ section
 * - Final CTA
 */
export const Default: Story = {}

/**
 * Mobile view of the Homelab page.
 * All templates are responsive and adapt to mobile screens.
 */
export const Mobile: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
  },
}

/**
 * Tablet view of the Homelab page.
 */
export const Tablet: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'tablet',
    },
  },
}

/**
 * Dark mode view of the Homelab page.
 * Tests dark mode compatibility across all templates.
 */
export const DarkMode: Story = {
  parameters: {
    backgrounds: { default: 'dark' },
  },
  globals: {
    theme: 'dark',
  },
}
