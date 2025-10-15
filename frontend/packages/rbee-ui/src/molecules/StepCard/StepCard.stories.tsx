import type { Meta, StoryObj } from '@storybook/react'
import { CheckCircle, Rocket, Server, Shield } from 'lucide-react'
import { StepCard } from './StepCard'

const meta: Meta<typeof StepCard> = {
  title: 'Molecules/StepCard',
  component: StepCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The StepCard molecule displays a numbered step with icon, title, intro, and deliverables list.

## Used In
- Deployment guides
- Getting started flows
- Process documentation
				`,
      },
    },
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof StepCard>

export const Default: Story = {
  args: {
    index: 1,
    icon: <Shield className="h-6 w-6" />,
    title: 'Security Setup',
    intro: 'Configure authentication and encryption',
    items: ['Generate TLS certificates', 'Set up API keys', 'Configure firewall'],
  },
}

export const WithNumber: Story = {
  args: {
    index: 2,
    icon: <Server className="h-6 w-6" />,
    title: 'Deploy Infrastructure',
    intro: 'Set up your GPU workers',
    items: ['Install Docker', 'Pull rbee images', 'Configure workers'],
  },
}

export const Active: Story = {
  args: {
    index: 3,
    icon: <CheckCircle className="h-6 w-6" />,
    title: 'Verify Deployment',
    intro: 'Test your setup',
    items: ['Run health checks', 'Test API endpoints', 'Monitor metrics'],
  },
}

export const Completed: Story = {
  args: {
    index: 4,
    icon: <Rocket className="h-6 w-6" />,
    title: 'Go Live',
    intro: 'Launch your AI infrastructure',
    items: ['Enable production mode', 'Configure monitoring', 'Set up alerts'],
    isLast: true,
  },
}
