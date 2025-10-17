import type { Meta, StoryObj } from '@storybook/react'
import DevOpsPage from './DevOpsPage'

const meta = {
  title: 'Pages/DevOpsPage',
  component: DevOpsPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'DevOps page targeting infrastructure engineers and platform teams. Showcases deployment automation, monitoring, scaling, and operational excellence.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof DevOpsPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete DevOps page with all sections:
 * - Hero with infrastructure overview
 * - Deployment automation
 * - Monitoring and observability
 * - Scaling and performance
 * - Infrastructure as code
 * - DevOps use cases
 * - Integration examples
 * - FAQ section
 * - Final CTA
 */
export const Default: Story = {}
