import type { Meta, StoryObj } from '@storybook/react'
import { Kbd, KbdGroup } from './Kbd'

const meta: Meta<typeof Kbd> = {
  title: 'Atoms/Kbd',
  component: Kbd,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Kbd>

export const Default: Story = {
  render: () => <Kbd>K</Kbd>,
}

export const Combination: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <span className="text-sm">Search:</span>
        <KbdGroup>
          <Kbd>⌘</Kbd>
          <span className="text-muted-foreground">+</span>
          <Kbd>K</Kbd>
        </KbdGroup>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm">Save:</span>
        <KbdGroup>
          <Kbd>Ctrl</Kbd>
          <span className="text-muted-foreground">+</span>
          <Kbd>S</Kbd>
        </KbdGroup>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm">Copy:</span>
        <KbdGroup>
          <Kbd>⌘</Kbd>
          <span className="text-muted-foreground">+</span>
          <Kbd>C</Kbd>
        </KbdGroup>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm">Paste:</span>
        <KbdGroup>
          <Kbd>⌘</Kbd>
          <span className="text-muted-foreground">+</span>
          <Kbd>V</Kbd>
        </KbdGroup>
      </div>
    </div>
  ),
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex items-center gap-4">
      <Kbd className="h-4 min-w-4 text-[10px]">K</Kbd>
      <Kbd>K</Kbd>
      <Kbd className="h-6 min-w-6 text-sm">K</Kbd>
      <Kbd className="h-7 min-w-7 text-base">K</Kbd>
    </div>
  ),
}

export const AllKeys: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        <Kbd>⌘</Kbd>
        <Kbd>Ctrl</Kbd>
        <Kbd>Alt</Kbd>
        <Kbd>Shift</Kbd>
        <Kbd>Tab</Kbd>
        <Kbd>Enter</Kbd>
        <Kbd>Esc</Kbd>
        <Kbd>Space</Kbd>
      </div>

      <div className="flex flex-wrap gap-2">
        <Kbd>A</Kbd>
        <Kbd>B</Kbd>
        <Kbd>C</Kbd>
        <Kbd>D</Kbd>
        <Kbd>E</Kbd>
        <Kbd>F</Kbd>
        <Kbd>G</Kbd>
        <Kbd>H</Kbd>
      </div>

      <div className="flex flex-wrap gap-2">
        <Kbd>1</Kbd>
        <Kbd>2</Kbd>
        <Kbd>3</Kbd>
        <Kbd>4</Kbd>
        <Kbd>5</Kbd>
        <Kbd>6</Kbd>
        <Kbd>7</Kbd>
        <Kbd>8</Kbd>
        <Kbd>9</Kbd>
        <Kbd>0</Kbd>
      </div>

      <div className="flex flex-wrap gap-2">
        <Kbd>↑</Kbd>
        <Kbd>↓</Kbd>
        <Kbd>←</Kbd>
        <Kbd>→</Kbd>
      </div>

      <div className="flex flex-wrap gap-2">
        <Kbd>F1</Kbd>
        <Kbd>F2</Kbd>
        <Kbd>F3</Kbd>
        <Kbd>F4</Kbd>
        <Kbd>F5</Kbd>
        <Kbd>F6</Kbd>
      </div>
    </div>
  ),
}
