import type { Meta, StoryObj } from '@storybook/react'
import CommunityPage from './CommunityPage'

const meta = {
  title: 'Pages/CommunityPage',
  component: CommunityPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Community page showcasing the open-source rbee community. Features contribution opportunities, community channels, events, and ways to get involved.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CommunityPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Community page with all sections:
 * - Hero with community introduction
 * - Community channels (Discord, GitHub, Forums)
 * - Contribution opportunities
 * - Community events and meetups
 * - Success stories from contributors
 * - FAQ section
 * - Final CTA to join
 */
export const Default: Story = {}
