import type { Meta, StoryObj } from '@storybook/react'
import { useIsMobile } from './UseMobile'

// Create a demo component
function UseMobileDemo() {
  const isMobile = useIsMobile()

  return (
    <div className="rounded border p-6">
      <h3 className="mb-4 text-lg font-semibold">useIsMobile Hook Demo</h3>
      <div className="space-y-2">
        <p className="text-sm">
          Current viewport: <strong>{isMobile ? 'Mobile' : 'Desktop'}</strong>
        </p>
        <p className="text-xs text-muted-foreground">Resize your browser window to see the change</p>
        <p className="text-xs text-muted-foreground">Breakpoint: 768px</p>
      </div>
    </div>
  )
}

const meta: Meta = {
  title: 'Atoms/UseMobile',
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj

export const Default: Story = {
  render: () => <UseMobileDemo />,
}

function WithComponentDemo() {
  const isMobile = useIsMobile()

  return (
    <div className="w-[400px] rounded border p-6">
      {isMobile ? (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Mobile View</h3>
          <p className="text-sm">This content is optimized for mobile devices.</p>
          <button className="w-full rounded-md bg-primary px-4 py-2 text-primary-foreground">Mobile Action</button>
        </div>
      ) : (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Desktop View</h3>
          <p className="text-sm">This content is optimized for desktop devices.</p>
          <div className="flex gap-2">
            <button className="rounded-md bg-primary px-4 py-2 text-primary-foreground">Action 1</button>
            <button className="rounded-md border px-4 py-2">Action 2</button>
          </div>
        </div>
      )}
    </div>
  )
}

export const WithComponent: Story = {
  render: () => <WithComponentDemo />,
}

export const Responsive: Story = {
  render: () => {
    const isMobile = useIsMobile()
    return (
      <div className="w-full max-w-2xl space-y-4 rounded border p-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Responsive Layout</h3>
          <span className="rounded-md bg-muted px-2 py-1 text-xs">{isMobile ? 'Mobile' : 'Desktop'}</span>
        </div>
        <div className={isMobile ? 'flex flex-col gap-4' : 'grid grid-cols-3 gap-4'}>
          <div className="rounded-md bg-muted p-4">Card 1</div>
          <div className="rounded-md bg-muted p-4">Card 2</div>
          <div className="rounded-md bg-muted p-4">Card 3</div>
        </div>
      </div>
    )
  },
}

export const WithBreakpoint: Story = {
  render: () => <UseMobileDemo />,
}
