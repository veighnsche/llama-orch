import { Button } from '@rbee/ui/atoms/Button'
import { useToast } from '@rbee/ui/hooks/use-toast'
import type { Meta, StoryObj } from '@storybook/react'
import { Toaster } from './Toaster'

const meta: Meta<typeof Toaster> = {
  title: 'Atoms/Toaster',
  component: Toaster,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Toaster>

function ToasterDemo() {
  const { toast } = useToast()

  return (
    <div>
      <Button
        onClick={() => {
          toast({
            title: 'Notification',
            description: 'This is a toast notification.',
          })
        }}
      >
        Show Toast
      </Button>
      <Toaster />
    </div>
  )
}

export const Default: Story = {
  render: () => <ToasterDemo />,
}

function MultipleToastsDemo() {
  const { toast } = useToast()

  return (
    <div className="flex flex-col gap-2">
      <Button
        onClick={() => {
          toast({
            title: 'First Toast',
            description: 'This is the first toast.',
          })
          setTimeout(() => {
            toast({
              title: 'Second Toast',
              description: 'This is the second toast.',
            })
          }, 500)
        }}
      >
        Show Multiple Toasts
      </Button>
      <Toaster />
    </div>
  )
}

export const Multiple: Story = {
  render: () => <MultipleToastsDemo />,
}

export const AllPositions: Story = {
  render: () => <ToasterDemo />,
}

function WithLimitDemo() {
  const { toast } = useToast()

  return (
    <div>
      <Button
        onClick={() => {
          toast({
            title: 'Toast with Limit',
            description: 'Only one toast can be shown at a time.',
          })
        }}
      >
        Show Toast
      </Button>
      <Toaster />
    </div>
  )
}

export const WithLimit: Story = {
  render: () => <WithLimitDemo />,
}
