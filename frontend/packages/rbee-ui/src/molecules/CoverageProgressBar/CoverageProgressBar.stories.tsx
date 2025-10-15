import type { Meta, StoryObj } from '@storybook/react'
import { CoverageProgressBar } from './CoverageProgressBar'

const meta = {
  title: 'Molecules/CoverageProgressBar',
  component: CoverageProgressBar,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    passing: {
      control: { type: 'number', min: 0 },
    },
    total: {
      control: { type: 'number', min: 1 },
    },
  },
} satisfies Meta<typeof CoverageProgressBar>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    label: 'BDD Coverage',
    passing: 42,
    total: 62,
  },
}

export const FullCoverage: Story = {
  args: {
    label: 'Test Coverage',
    passing: 100,
    total: 100,
  },
}

export const NoCoverage: Story = {
  args: {
    label: 'Integration Tests',
    passing: 0,
    total: 50,
  },
}

export const PartialCoverage: Story = {
  args: {
    label: 'Unit Tests',
    passing: 25,
    total: 100,
  },
}

export const HighCoverage: Story = {
  args: {
    label: 'Code Coverage',
    passing: 95,
    total: 100,
  },
}

export const LowCoverage: Story = {
  args: {
    label: 'E2E Tests',
    passing: 5,
    total: 100,
  },
}

export const CustomLabel: Story = {
  args: {
    label: 'Acceptance Criteria Met',
    passing: 18,
    total: 24,
  },
}

export const SmallTotal: Story = {
  args: {
    label: 'Critical Features',
    passing: 3,
    total: 5,
  },
}

export const LargeTotal: Story = {
  args: {
    label: 'All Test Suites',
    passing: 847,
    total: 1203,
  },
}

export const AlmostComplete: Story = {
  args: {
    label: 'Sprint Goals',
    passing: 49,
    total: 50,
  },
}

export const JustStarted: Story = {
  args: {
    label: 'New Feature Tests',
    passing: 1,
    total: 30,
  },
}
