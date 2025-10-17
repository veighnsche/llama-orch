import type { Meta, StoryObj } from '@storybook/react'
import LegalPage from './LegalPage'

const meta = {
  title: 'Pages/LegalPage',
  component: LegalPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Legal page hub with links to Terms of Service, Privacy Policy, and other legal documents. Provides overview of legal information and compliance.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof LegalPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Legal page with all sections:
 * - Hero with legal overview
 * - Links to legal documents
 * - License information (GPL-3.0)
 * - Contact information
 * - Last updated dates
 */
export const Default: Story = {}
