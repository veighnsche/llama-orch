import { Button } from '@rbee/ui/atoms/Button'
import type { Meta, StoryObj } from '@storybook/react'
import { toast } from 'sonner'
import { Toaster } from './Sonner'

const meta: Meta<typeof Toaster> = {
  title: 'Atoms/Sonner',
  component: Toaster,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Toaster>

export const Default: Story = {
  render: () => (
    <div>
      <Toaster />
      <Button onClick={() => toast('This is a default toast notification')}>Show Toast</Button>
    </div>
  ),
}

export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-col gap-2">
      <Toaster />
      <Button onClick={() => toast('Default notification')}>Default</Button>
      <Button onClick={() => toast.success('Operation completed successfully')}>Success</Button>
      <Button onClick={() => toast.error('An error occurred')}>Error</Button>
      <Button onClick={() => toast.info('Here is some information')}>Info</Button>
      <Button onClick={() => toast.warning('This is a warning')}>Warning</Button>
    </div>
  ),
}

export const WithAction: Story = {
  render: () => (
    <div>
      <Toaster />
      <Button
        onClick={() =>
          toast('Event scheduled', {
            action: {
              label: 'Undo',
              onClick: () => toast('Undo clicked'),
            },
          })
        }
      >
        Show Toast with Action
      </Button>
    </div>
  ),
}

export const WithDescription: Story = {
  render: () => (
    <div>
      <Toaster />
      <Button
        onClick={() =>
          toast('New message received', {
            description: 'You have a new message from John Doe. Click to view details.',
          })
        }
      >
        Show Toast with Description
      </Button>
    </div>
  ),
}
