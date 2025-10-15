// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { Info } from 'lucide-react'
import { Label } from './Label'

const meta: Meta<typeof Label> = {
  title: 'Atoms/Label',
  component: Label,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A form label component built on Radix UI Label primitive.

## Features
- Accessible form labeling
- Supports disabled state via group data attribute
- Peer-based styling for connected inputs
- Gap support for icons and help text
- Proper cursor and opacity handling

## Used In
- Form fields
- Input groups
- Checkbox and radio groups
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    children: {
      control: 'text',
      description: 'Label text content',
    },
    htmlFor: {
      control: 'text',
      description: 'ID of the associated form control',
    },
  },
}

export default meta
type Story = StoryObj<typeof Label>

export const Default: Story = {
  args: {
    children: 'Email address',
    htmlFor: 'email',
  },
}

export const Required: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div>
        <Label htmlFor="email-required">
          Email address <span className="text-destructive">*</span>
        </Label>
      </div>
      <div>
        <Label htmlFor="name-required">
          Full name <span className="text-destructive">*</span>
        </Label>
      </div>
      <div>
        <Label htmlFor="company">Company (optional)</Label>
      </div>
    </div>
  ),
}

export const WithHelp: Story = {
  render: () => (
    <div className="flex flex-col gap-6 max-w-md">
      <div className="space-y-2">
        <Label htmlFor="api-key" className="flex items-center gap-2">
          API Key
          <Info className="size-3.5 text-muted-foreground" />
        </Label>
        <input
          id="api-key"
          type="text"
          className="w-full px-3 py-2 border border-input rounded-md bg-transparent text-sm"
          placeholder="rbee_..."
        />
        <p className="text-xs text-muted-foreground">Your API key is used to authenticate requests.</p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="model" className="flex items-center gap-2">
          Model Selection
          <Info className="size-3.5 text-muted-foreground" />
        </Label>
        <select id="model" className="w-full px-3 py-2 border border-input rounded-md bg-transparent text-sm">
          <option>llama-3.1-8b</option>
          <option>llama-3.1-70b</option>
          <option>mistral-7b</option>
        </select>
        <p className="text-xs text-muted-foreground">Choose the model that best fits your use case.</p>
      </div>
    </div>
  ),
}

export const InForm: Story = {
  render: () => (
    <form className="w-full max-w-md space-y-6 p-6 border rounded-lg">
      <div>
        <h3 className="text-lg font-semibold mb-4">Create Account</h3>
      </div>

      <div className="space-y-2">
        <Label htmlFor="form-email">
          Email <span className="text-destructive">*</span>
        </Label>
        <input
          id="form-email"
          type="email"
          required
          className="w-full px-3 py-2 border border-input rounded-md bg-transparent text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          placeholder="you@example.com"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="form-password">
          Password <span className="text-destructive">*</span>
        </Label>
        <input
          id="form-password"
          type="password"
          required
          className="w-full px-3 py-2 border border-input rounded-md bg-transparent text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          placeholder="••••••••"
        />
        <p className="text-xs text-muted-foreground">Must be at least 8 characters.</p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="form-company">Company (optional)</Label>
        <input
          id="form-company"
          type="text"
          className="w-full px-3 py-2 border border-input rounded-md bg-transparent text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          placeholder="Acme Inc."
        />
      </div>

      <div className="flex items-center gap-2">
        <input id="form-terms" type="checkbox" className="size-4" />
        <Label htmlFor="form-terms" className="font-normal cursor-pointer">
          I agree to the terms and conditions
        </Label>
      </div>

      <button
        type="submit"
        className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        Create Account
      </button>
    </form>
  ),
}

export const DisabledState: Story = {
  render: () => (
    <div className="flex flex-col gap-6 max-w-md">
      <div className="space-y-2" data-disabled="true">
        <Label htmlFor="disabled-input">Disabled Field</Label>
        <input
          id="disabled-input"
          type="text"
          disabled
          className="w-full px-3 py-2 border border-input rounded-md bg-transparent text-sm opacity-50 cursor-not-allowed"
          placeholder="Cannot edit"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="enabled-input">Enabled Field</Label>
        <input
          id="enabled-input"
          type="text"
          className="w-full px-3 py-2 border border-input rounded-md bg-transparent text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          placeholder="Can edit"
        />
      </div>
    </div>
  ),
}
