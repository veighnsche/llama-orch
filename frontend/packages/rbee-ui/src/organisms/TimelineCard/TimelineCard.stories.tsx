import type { Meta, StoryObj } from '@storybook/react'
import { TimelineCard } from './TimelineCard'

const meta = {
  title: 'Organisms/TimelineCard',
  component: TimelineCard,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof TimelineCard>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    heading: 'Deployment Timeline',
    description: 'Your custom deployment schedule',
    progress: 25,
    weeks: [
      { week: 'Week 1', phase: 'Infrastructure Setup' },
      { week: 'Week 2', phase: 'Model Deployment' },
      { week: 'Week 3', phase: 'Testing & Validation' },
      { week: 'Week 4', phase: 'Go Live' },
    ],
  },
}

export const EarlyProgress: Story = {
  args: {
    heading: 'Project Phases',
    description: 'Track your implementation progress',
    progress: 10,
    weeks: [
      { week: 'Phase 1', phase: 'Discovery' },
      { week: 'Phase 2', phase: 'Design' },
      { week: 'Phase 3', phase: 'Development' },
      { week: 'Phase 4', phase: 'Launch' },
    ],
  },
}

export const MidProgress: Story = {
  args: {
    heading: 'Onboarding Journey',
    description: 'Complete these steps to get started',
    progress: 50,
    weeks: [
      { week: 'Step 1', phase: 'Account Setup' },
      { week: 'Step 2', phase: 'Configuration' },
      { week: 'Step 3', phase: 'Integration' },
      { week: 'Step 4', phase: 'Production Ready' },
    ],
  },
}

export const NearComplete: Story = {
  args: {
    heading: 'Migration Timeline',
    description: 'Almost there!',
    progress: 75,
    weeks: [
      { week: 'Q1', phase: 'Planning' },
      { week: 'Q2', phase: 'Execution' },
      { week: 'Q3', phase: 'Optimization' },
      { week: 'Q4', phase: 'Scale' },
    ],
  },
}

export const ManyWeeks: Story = {
  args: {
    heading: '12-Week Rollout',
    description: 'Enterprise deployment schedule',
    progress: 33,
    weeks: [
      { week: 'Week 1-2', phase: 'Infrastructure Provisioning' },
      { week: 'Week 3-4', phase: 'Security Hardening' },
      { week: 'Week 5-6', phase: 'Model Deployment' },
      { week: 'Week 7-8', phase: 'Integration Testing' },
      { week: 'Week 9-10', phase: 'User Acceptance Testing' },
      { week: 'Week 11-12', phase: 'Production Launch' },
    ],
  },
}
