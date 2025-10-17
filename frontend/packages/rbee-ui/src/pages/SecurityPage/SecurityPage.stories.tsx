import type { Meta, StoryObj } from '@storybook/react'
import SecurityPage from './SecurityPage'

const meta = {
  title: 'Pages/SecurityPage',
  component: SecurityPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Security page showcasing rbee security features, architecture, and best practices. Covers process isolation, encryption, audit trails, and security compliance.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof SecurityPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Security page with all sections:
 * - Hero with security overview
 * - Security architecture
 * - Process isolation
 * - Encryption and data protection
 * - Audit trails and logging
 * - Security best practices
 * - Vulnerability disclosure
 * - FAQ section
 * - Final CTA
 */
export const Default: Story = {}
