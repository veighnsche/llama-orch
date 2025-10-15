import { Card, CardContent } from '@rbee/ui/atoms/Card'
import type { Meta, StoryObj } from '@storybook/react'
import { ButtonCardFooter } from './ButtonCardFooter'

const meta = {
  title: 'Molecules/ButtonCardFooter',
  component: ButtonCardFooter,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ButtonCardFooter>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    buttonText: 'Action Button',
    href: '#',
  },
  render: (args) => (
    <Card className="w-[400px]">
      <CardContent className="p-6">
        <h3 className="text-lg font-semibold mb-2">Card Title</h3>
        <p className="text-muted-foreground">This is some card content that demonstrates the sticky footer.</p>
      </CardContent>
      <ButtonCardFooter {...args} />
    </Card>
  ),
}

export const WithBadge: Story = {
  args: {
    badgeSlot: (
      <span className="inline-flex items-center rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
        Special Offer
      </span>
    ),
    buttonText: 'Get Started',
    href: '#',
  },
  render: (args) => (
    <Card className="w-[400px]">
      <CardContent className="p-6">
        <h3 className="text-lg font-semibold mb-2">Card Title</h3>
        <p className="text-muted-foreground">This card has a badge above the button.</p>
      </CardContent>
      <ButtonCardFooter {...args} />
    </Card>
  ),
}

export const Elevated: Story = {
  args: {
    variant: 'elevated',
    badgeSlot: (
      <span className="inline-flex items-center rounded-full bg-destructive/10 px-3 py-1 text-xs font-medium text-destructive">
        Limited Time
      </span>
    ),
    buttonText: 'Claim Now',
    href: '#',
  },
  render: (args) => (
    <Card className="w-[400px]">
      <CardContent className="p-6">
        <h3 className="text-lg font-semibold mb-2">Card Title</h3>
        <p className="text-muted-foreground">This footer has elevated shadow variant.</p>
      </CardContent>
      <ButtonCardFooter {...args} />
    </Card>
  ),
}

export const ColoredButtons: Story = {
  args: {
    buttonText: 'Action',
    href: '#',
  },
  render: () => (
    <div className="flex flex-col gap-4">
      {(['primary', 'chart-1', 'chart-2', 'chart-3'] as const).map((color) => (
        <Card key={color} className="w-[400px]">
          <CardContent className="p-6">
            <h3 className="text-lg font-semibold mb-2">Card with {color}</h3>
            <p className="text-muted-foreground">Different button colors.</p>
          </CardContent>
          <ButtonCardFooter buttonText="Action" buttonColor={color} href="#" />
        </Card>
      ))}
    </div>
  ),
}
