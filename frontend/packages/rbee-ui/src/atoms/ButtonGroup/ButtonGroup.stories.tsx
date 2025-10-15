// Created by: TEAM-011

import { Button } from '@rbee/ui/atoms/Button'
import type { Meta, StoryObj } from '@storybook/react'
import { AlignCenter, AlignLeft, AlignRight, Bold, Italic, Underline } from 'lucide-react'

const meta: Meta = {
  title: 'Atoms/ButtonGroup',
  parameters: { layout: 'centered' },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj

export const Default: Story = {
  render: () => (
    <div className="inline-flex rounded-md shadow-sm">
      <Button variant="outline" className="rounded-r-none">
        Left
      </Button>
      <Button variant="outline" className="rounded-none border-l-0">
        Middle
      </Button>
      <Button variant="outline" className="rounded-l-none border-l-0">
        Right
      </Button>
    </div>
  ),
}

export const Vertical: Story = {
  render: () => (
    <div className="inline-flex flex-col rounded-md shadow-sm">
      <Button variant="outline" className="rounded-b-none">
        Top
      </Button>
      <Button variant="outline" className="rounded-none border-t-0">
        Middle
      </Button>
      <Button variant="outline" className="rounded-t-none border-t-0">
        Bottom
      </Button>
    </div>
  ),
}

export const WithIcons: Story = {
  render: () => (
    <div className="inline-flex rounded-md shadow-sm">
      <Button variant="outline" size="icon" className="rounded-r-none">
        <Bold className="size-4" />
      </Button>
      <Button variant="outline" size="icon" className="rounded-none border-l-0">
        <Italic className="size-4" />
      </Button>
      <Button variant="outline" size="icon" className="rounded-l-none border-l-0">
        <Underline className="size-4" />
      </Button>
    </div>
  ),
}

export const Mixed: Story = {
  render: () => (
    <div className="inline-flex rounded-md shadow-sm">
      <Button className="rounded-r-none">Save</Button>
      <Button variant="outline" className="rounded-l-none border-l-0">
        Cancel
      </Button>
    </div>
  ),
}
