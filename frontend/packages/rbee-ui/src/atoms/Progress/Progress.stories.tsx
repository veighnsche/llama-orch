// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { Progress } from './Progress'

const meta: Meta<typeof Progress> = {
  title: 'Atoms/Progress',
  component: Progress,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A progress bar component built on Radix UI Progress primitive.

## Features
- Smooth transitions for value changes
- Accessible with proper ARIA attributes
- Customizable height and colors
- Supports 0-100 value range
- Indeterminate state support

## Used In
- File uploads
- Loading states
- Multi-step forms
- Download progress
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    value: {
      control: { type: 'range', min: 0, max: 100, step: 1 },
      description: 'Progress value (0-100)',
    },
    className: {
      control: 'text',
      description: 'Additional CSS classes',
    },
  },
}

export default meta
type Story = StoryObj<typeof Progress>

export const Default: Story = {
  args: {
    value: 60,
  },
}

export const AllStates: Story = {
  render: () => (
    <div className="flex flex-col gap-6 w-80">
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Empty</span>
          <span className="text-muted-foreground">0%</span>
        </div>
        <Progress value={0} />
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Starting</span>
          <span className="text-muted-foreground">15%</span>
        </div>
        <Progress value={15} />
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>In Progress</span>
          <span className="text-muted-foreground">45%</span>
        </div>
        <Progress value={45} />
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Almost Done</span>
          <span className="text-muted-foreground">85%</span>
        </div>
        <Progress value={85} />
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Complete</span>
          <span className="text-muted-foreground">100%</span>
        </div>
        <Progress value={100} />
      </div>
    </div>
  ),
}

export const WithLabel: Story = {
  render: () => (
    <div className="flex flex-col gap-6 w-96">
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Uploading model weights</span>
          <span className="text-sm text-muted-foreground">2.4 GB / 4.8 GB</span>
        </div>
        <Progress value={50} />
        <p className="text-xs text-muted-foreground">Estimated time remaining: 3 minutes</p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Processing dataset</span>
          <span className="text-sm text-muted-foreground">7,500 / 10,000 items</span>
        </div>
        <Progress value={75} />
        <p className="text-xs text-muted-foreground">Processing at 250 items/sec</p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Model deployment</span>
          <span className="text-sm text-green-600 dark:text-green-400">Complete</span>
        </div>
        <Progress value={100} />
        <p className="text-xs text-muted-foreground">Deployment successful</p>
      </div>
    </div>
  ),
}

export const Indeterminate: Story = {
  render: () => (
    <div className="flex flex-col gap-6 w-80">
      <div className="space-y-2">
        <span className="text-sm font-medium">Loading...</span>
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-primary/20">
          <div className="h-full w-1/3 bg-primary animate-pulse" />
        </div>
        <p className="text-xs text-muted-foreground">Connecting to server</p>
      </div>

      <div className="space-y-2">
        <span className="text-sm font-medium">Processing</span>
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-primary/20">
          <div
            className="h-full w-1/4 bg-primary animate-[shimmer_2s_ease-in-out_infinite]"
            style={{
              backgroundImage: 'linear-gradient(90deg, transparent, currentColor, transparent)',
            }}
          />
        </div>
        <p className="text-xs text-muted-foreground">Please wait...</p>
      </div>
    </div>
  ),
}

export const CustomSizes: Story = {
  render: () => (
    <div className="flex flex-col gap-6 w-80">
      <div className="space-y-2">
        <span className="text-xs text-muted-foreground">Extra Small (1px)</span>
        <Progress value={60} className="h-px" />
      </div>

      <div className="space-y-2">
        <span className="text-xs text-muted-foreground">Small (1.5px)</span>
        <Progress value={60} className="h-1.5" />
      </div>

      <div className="space-y-2">
        <span className="text-xs text-muted-foreground">Default (2px)</span>
        <Progress value={60} />
      </div>

      <div className="space-y-2">
        <span className="text-xs text-muted-foreground">Medium (3px)</span>
        <Progress value={60} className="h-3" />
      </div>

      <div className="space-y-2">
        <span className="text-xs text-muted-foreground">Large (4px)</span>
        <Progress value={60} className="h-4" />
      </div>
    </div>
  ),
}

export const InMultiStepForm: Story = {
  render: () => (
    <div className="w-full max-w-2xl p-6 border rounded-lg">
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Deploy Your Model</h3>
        <p className="text-sm text-muted-foreground">Step 2 of 4: Configure Resources</p>
      </div>

      <Progress value={50} className="mb-8" />

      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <div className="flex items-center justify-center size-8 rounded-full bg-primary text-primary-foreground text-sm font-medium">
            ✓
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium">Select Model</p>
            <p className="text-xs text-muted-foreground">llama-3.1-8b selected</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center justify-center size-8 rounded-full bg-primary text-primary-foreground text-sm font-medium">
            2
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium">Configure Resources</p>
            <p className="text-xs text-muted-foreground">In progress...</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center justify-center size-8 rounded-full border-2 border-muted text-muted-foreground text-sm font-medium">
            3
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-muted-foreground">Review & Deploy</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center justify-center size-8 rounded-full border-2 border-muted text-muted-foreground text-sm font-medium">
            4
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-muted-foreground">Confirmation</p>
          </div>
        </div>
      </div>
    </div>
  ),
}
