import type { Meta, StoryObj } from '@storybook/react'
import PrivacyPage from './PrivacyPage'

const meta = {
  title: 'Pages/PrivacyPage',
  component: PrivacyPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Privacy Policy page with legal content about data collection, usage, and user rights. Includes GDPR compliance information and contact details.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof PrivacyPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Privacy Policy page with all sections:
 * - Hero with privacy overview
 * - Data collection practices
 * - Data usage and sharing
 * - User rights (GDPR)
 * - Cookie policy
 * - Contact information
 * - Last updated date
 */
export const Default: Story = {}
