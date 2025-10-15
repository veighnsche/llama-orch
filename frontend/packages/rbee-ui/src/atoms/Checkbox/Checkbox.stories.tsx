// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { Checkbox } from './Checkbox'

const meta: Meta<typeof Checkbox> = {
  title: 'Atoms/Checkbox',
  component: Checkbox,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A checkbox component built on Radix UI Checkbox primitive.

## Features
- Three states: unchecked, checked, indeterminate
- Accessible with proper ARIA attributes
- Keyboard navigation support
- Focus visible states
- Disabled state support
- Dark mode compatible

## Used In
- Forms
- Settings panels
- Multi-select lists
- Terms and conditions
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    checked: {
      control: 'select',
      options: [true, false, 'indeterminate'],
      description: 'Checkbox state',
    },
    disabled: {
      control: 'boolean',
      description: 'Disable the checkbox',
    },
  },
}

export default meta
type Story = StoryObj<typeof Checkbox>

export const Default: Story = {
  args: {
    checked: false,
  },
}

export const Checked: Story = {
  args: {
    checked: true,
  },
}

export const Indeterminate: Story = {
  args: {
    checked: 'indeterminate',
  },
}

export const Disabled: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <Checkbox disabled checked={false} id="disabled-unchecked" />
        <label htmlFor="disabled-unchecked" className="text-sm text-muted-foreground">
          Disabled unchecked
        </label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox disabled checked={true} id="disabled-checked" />
        <label htmlFor="disabled-checked" className="text-sm text-muted-foreground">
          Disabled checked
        </label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox disabled checked="indeterminate" id="disabled-indeterminate" />
        <label htmlFor="disabled-indeterminate" className="text-sm text-muted-foreground">
          Disabled indeterminate
        </label>
      </div>
    </div>
  ),
}

export const WithLabels: Story = {
  render: () => (
    <div className="flex flex-col gap-4 max-w-md">
      <div className="flex items-start gap-3">
        <Checkbox id="terms" />
        <div className="grid gap-1.5 leading-none">
          <label
            htmlFor="terms"
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
          >
            Accept terms and conditions
          </label>
          <p className="text-sm text-muted-foreground">You agree to our Terms of Service and Privacy Policy.</p>
        </div>
      </div>

      <div className="flex items-start gap-3">
        <Checkbox id="marketing" />
        <div className="grid gap-1.5 leading-none">
          <label
            htmlFor="marketing"
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
          >
            Marketing emails
          </label>
          <p className="text-sm text-muted-foreground">Receive emails about new features and product updates.</p>
        </div>
      </div>

      <div className="flex items-start gap-3">
        <Checkbox id="security" defaultChecked />
        <div className="grid gap-1.5 leading-none">
          <label
            htmlFor="security"
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
          >
            Security alerts
          </label>
          <p className="text-sm text-muted-foreground">Get notified about important security updates. (Recommended)</p>
        </div>
      </div>
    </div>
  ),
}

export const InForm: Story = {
  render: () => (
    <form className="w-full max-w-md space-y-6 p-6 border rounded-lg">
      <div>
        <h3 className="text-lg font-semibold mb-2">Notification Preferences</h3>
        <p className="text-sm text-muted-foreground">Choose what updates you want to receive.</p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Checkbox id="all-notifications" checked="indeterminate" />
          <label htmlFor="all-notifications" className="text-sm font-medium cursor-pointer">
            All notifications
          </label>
        </div>

        <div className="ml-6 space-y-3">
          <div className="flex items-center gap-3">
            <Checkbox id="email-notif" defaultChecked />
            <label htmlFor="email-notif" className="text-sm cursor-pointer">
              Email notifications
            </label>
          </div>

          <div className="flex items-center gap-3">
            <Checkbox id="push-notif" />
            <label htmlFor="push-notif" className="text-sm cursor-pointer">
              Push notifications
            </label>
          </div>

          <div className="flex items-center gap-3">
            <Checkbox id="sms-notif" defaultChecked />
            <label htmlFor="sms-notif" className="text-sm cursor-pointer">
              SMS notifications
            </label>
          </div>
        </div>
      </div>

      <div className="space-y-4 pt-4 border-t border-border">
        <div className="flex items-center gap-3">
          <Checkbox id="deployment-alerts" defaultChecked />
          <label htmlFor="deployment-alerts" className="text-sm cursor-pointer">
            Deployment alerts
          </label>
        </div>

        <div className="flex items-center gap-3">
          <Checkbox id="usage-reports" />
          <label htmlFor="usage-reports" className="text-sm cursor-pointer">
            Weekly usage reports
          </label>
        </div>

        <div className="flex items-center gap-3">
          <Checkbox id="billing-updates" defaultChecked />
          <label htmlFor="billing-updates" className="text-sm cursor-pointer">
            Billing updates
          </label>
        </div>
      </div>

      <button
        type="submit"
        className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        Save Preferences
      </button>
    </form>
  ),
}

export const MultiSelect: Story = {
  render: () => (
    <div className="w-full max-w-md p-6 border rounded-lg">
      <div className="mb-4">
        <h3 className="text-sm font-semibold mb-2">Select Models</h3>
        <p className="text-xs text-muted-foreground">Choose which models to deploy</p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/40 transition-colors">
          <div className="flex items-center gap-3">
            <Checkbox id="llama-8b" defaultChecked />
            <div>
              <label htmlFor="llama-8b" className="text-sm font-medium cursor-pointer block">
                Llama 3.1 8B
              </label>
              <p className="text-xs text-muted-foreground">Fast, efficient for most tasks</p>
            </div>
          </div>
          <span className="text-xs text-muted-foreground">8GB VRAM</span>
        </div>

        <div className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/40 transition-colors">
          <div className="flex items-center gap-3">
            <Checkbox id="llama-70b" />
            <div>
              <label htmlFor="llama-70b" className="text-sm font-medium cursor-pointer block">
                Llama 3.1 70B
              </label>
              <p className="text-xs text-muted-foreground">High quality, complex reasoning</p>
            </div>
          </div>
          <span className="text-xs text-muted-foreground">40GB VRAM</span>
        </div>

        <div className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/40 transition-colors">
          <div className="flex items-center gap-3">
            <Checkbox id="mistral-7b" defaultChecked />
            <div>
              <label htmlFor="mistral-7b" className="text-sm font-medium cursor-pointer block">
                Mistral 7B
              </label>
              <p className="text-xs text-muted-foreground">Balanced performance</p>
            </div>
          </div>
          <span className="text-xs text-muted-foreground">7GB VRAM</span>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-border">
        <p className="text-xs text-muted-foreground">
          <strong>2 models selected</strong> • Total VRAM: 15GB
        </p>
      </div>
    </div>
  ),
}
