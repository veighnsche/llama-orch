// Dark Mode Polish Showcase for Input/Select/Textarea
import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '../Button/Button'
import { Input } from '../Input/Input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../Select/Select'
import { Textarea } from '../Textarea/Textarea'

const meta: Meta<typeof Input> = {
  title: 'Atoms/Forms/Dark Mode Polish',
  component: Input,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#0b1220' }],
    },
  },
  decorators: [
    (Story) => (
      <div className="dark">
        <Story />
      </div>
    ),
  ],
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Input>

/**
 * ## Dark Mode Polish Features
 *
 * Demonstrates:
 * - No drift: bg-[color:var(--background)] tracks canvas hue
 * - Selection: Amber-700 wash (rgba(185,83,9,0.32))
 * - Caret: Accent color (#f59e0b)
 * - Disabled: Clear legibility (#1a2435 bg, #6c7a90 text)
 */

export const InputStates: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded space-y-6 max-w-md">
      <div className="space-y-2">
        <label className="text-sm font-medium">Default Input</label>
        <Input placeholder="Type to see accent caret..." />
        <p className="text-xs text-muted-foreground">Caret color: #f59e0b (accent). Select text to see amber wash.</p>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Disabled Input</label>
        <Input placeholder="Disabled state" disabled />
        <p className="text-xs text-muted-foreground">bg-[#1a2435], text-[#6c7a90], border-[#223047] for legibility.</p>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">With Value (Select Text)</label>
        <Input defaultValue="Select this text to see amber-700 wash (rgba(185,83,9,0.32))" />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Input states showing caret color, selection wash, and disabled legibility.',
      },
    },
  },
}

export const SelectStates: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded space-y-6 max-w-md">
      <div className="space-y-2">
        <label className="text-sm font-medium">Select Dropdown</label>
        <Select>
          <SelectTrigger>
            <SelectValue placeholder="Choose an option..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">Option 1</SelectItem>
            <SelectItem value="2">Option 2</SelectItem>
            <SelectItem value="3">Option 3</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">Placeholder: #8b9bb0 for legibility. Background tracks canvas.</p>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Disabled Select</label>
        <Select disabled>
          <SelectTrigger>
            <SelectValue placeholder="Disabled..." />
          </SelectTrigger>
        </Select>
        <p className="text-xs text-muted-foreground">Same disabled colors as Input for consistency.</p>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Select states with improved placeholder legibility and disabled clarity.',
      },
    },
  },
}

export const TextareaStates: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded space-y-6 max-w-md">
      <div className="space-y-2">
        <label className="text-sm font-medium">Textarea</label>
        <Textarea placeholder="Type multiple lines..." rows={4} />
        <p className="text-xs text-muted-foreground">Same caret, selection, and inset shadow as Input.</p>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Disabled Textarea</label>
        <Textarea placeholder="Disabled..." rows={3} disabled />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Textarea with consistent styling across all form inputs.',
      },
    },
  },
}

export const FormComplete: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded">
      <div className="bg-card border border-border rounded-md p-6 max-w-md">
        <h3 className="text-lg font-semibold mb-4">Contact Form</h3>
        <div className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="name" className="text-sm font-medium">
              Name
            </label>
            <Input id="name" placeholder="Your name" />
          </div>

          <div className="space-y-2">
            <label htmlFor="email" className="text-sm font-medium">
              Email
            </label>
            <Input id="email" type="email" placeholder="you@example.com" />
          </div>

          <div className="space-y-2">
            <label htmlFor="topic" className="text-sm font-medium">
              Topic
            </label>
            <Select>
              <SelectTrigger id="topic">
                <SelectValue placeholder="Select a topic..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="support">Support</SelectItem>
                <SelectItem value="sales">Sales</SelectItem>
                <SelectItem value="feedback">Feedback</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label htmlFor="message" className="text-sm font-medium">
              Message
            </label>
            <Textarea id="message" placeholder="Your message..." rows={4} />
          </div>

          <div className="flex gap-2 pt-2">
            <Button variant="outline" className="flex-1">
              Cancel
            </Button>
            <Button className="flex-1">Send Message</Button>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Complete form showing all input types with consistent dark mode polish.',
      },
    },
  },
}
