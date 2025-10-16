import type { Meta, StoryObj } from '@storybook/react'
import { TimelineStep } from './TimelineStep'

const meta: Meta<typeof TimelineStep> = {
  title: 'Molecules/TimelineStep',
  component: TimelineStep,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
## Overview
TimelineStep molecule displays a single step in a timeline or sequence with timestamp, title, and optional description.

## When to Use
- Process flows and sequences
- Cancellation/lifecycle timelines
- Step-by-step guides
- Event logs with timestamps

## Composition
- **Timestamp**: Small muted label (e.g., "t+0ms", "Step 1")
- **Title**: Bold heading with optional ReactNode for custom styling
- **Description**: Optional muted text for additional context
- **Variant**: Visual variants for success/warning/error states
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    timestamp: {
      control: 'text',
      description: 'Timestamp or step label',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    title: {
      control: 'text',
      description: 'Step title/heading (can be ReactNode)',
      table: {
        type: { summary: 'ReactNode' },
        category: 'Content',
      },
    },
    description: {
      control: 'text',
      description: 'Optional step description',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    variant: {
      control: 'select',
      options: ['default', 'success', 'warning', 'error'],
      description: 'Visual variant',
      table: {
        type: { summary: 'default | success | warning | error' },
        defaultValue: { summary: 'default' },
        category: 'Appearance',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof TimelineStep>

export const Default: Story = {
  args: {
    timestamp: 't+0ms',
    title: 'Client sends POST /v1/cancel',
    description: 'Idempotent request.',
  },
}

export const WithCode: Story = {
  args: {
    timestamp: 't+0ms',
    title: (
      <>
        Client sends <code className="bg-muted px-1 rounded text-xs">POST /v1/cancel</code>
      </>
    ),
    description: 'Idempotent request.',
  },
}

export const Success: Story = {
  args: {
    timestamp: 't+120ms',
    title: <span className="text-chart-3">Worker idle ✓</span>,
    description: 'Ready for next task.',
    variant: 'success',
  },
}

export const Warning: Story = {
  args: {
    timestamp: 't+50ms',
    title: 'SSE disconnect detected',
    description: 'Stream closes ≤ 1s.',
    variant: 'warning',
  },
}

export const Error: Story = {
  args: {
    timestamp: 't+100ms',
    title: <span className="text-destructive">Connection failed</span>,
    description: 'Retry with exponential backoff.',
    variant: 'error',
  },
}

export const WithoutDescription: Story = {
  args: {
    timestamp: 'Step 1',
    title: 'Initialize worker process',
  },
}

export const CancellationSequence: Story = {
  render: () => (
    <div className="max-w-5xl">
      <ol className="grid gap-3 sm:grid-cols-4 text-sm" aria-label="Cancellation sequence">
        <TimelineStep
          timestamp="t+0ms"
          title={
            <>
              Client sends <code className="bg-muted px-1 rounded text-xs">POST /v1/cancel</code>
            </>
          }
          description="Idempotent request."
        />
        <TimelineStep timestamp="t+50ms" title="SSE disconnect detected" description="Stream closes ≤ 1s." />
        <TimelineStep
          timestamp="t+80ms"
          title="Immediate cleanup"
          description="Stop tokens, release slot, log event."
        />
        <TimelineStep
          timestamp="t+120ms"
          title={<span className="text-chart-3">Worker idle ✓</span>}
          description="Ready for next task."
          variant="success"
        />
      </ol>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Example of TimelineStep used in a 4-step cancellation sequence grid.',
      },
    },
  },
}
