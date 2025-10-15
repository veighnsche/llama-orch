// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { ConsoleOutput } from './ConsoleOutput'

const meta: Meta<typeof ConsoleOutput> = {
  title: 'Atoms/ConsoleOutput',
  component: ConsoleOutput,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    showChrome: {
      control: 'boolean',
      description: 'Show terminal window chrome',
    },
    variant: {
      control: 'select',
      options: ['terminal', 'code', 'output'],
    },
    background: {
      control: 'select',
      options: ['dark', 'light', 'card'],
    },
    copyable: {
      control: 'boolean',
    },
  },
}

export default meta
type Story = StoryObj<typeof ConsoleOutput>

/**
 * ## Overview
 * ConsoleOutput displays terminal/console output with monospace font and optional chrome.
 * Features copy-to-clipboard and multiple visual variants.
 *
 * ## When to Use
 * - Show command examples
 * - Display API responses
 * - Code snippets
 * - Installation instructions
 *
 * ## Used In
 * - HowItWorksSection
 * - DeveloperGuide
 */

export const Default: Story = {
  args: {
    children: '$ npm install @orchyra/sdk\n✓ Installed successfully',
  },
}

export const WithCommand: Story = {
  render: () => (
    <ConsoleOutput showChrome title="bash" copyable copyText="curl -X POST https://api.orchyra.ai/v1/chat/completions">
      $ curl -X POST https://api.orchyra.ai/v1/chat/completions
    </ConsoleOutput>
  ),
}

export const Streaming: Story = {
  render: () => (
    <div className="w-[600px] space-y-4">
      <ConsoleOutput showChrome title="terminal" background="dark">
        {`$ orchyra deploy llama-3-8b
Initializing deployment...
✓ Model downloaded
✓ GPU allocated
✓ Container started
→ Deployment ready at https://llama-3-8b.orchyra.ai`}
      </ConsoleOutput>
    </div>
  ),
}

export const InHowItWorks: Story = {
  render: () => (
    <div className="max-w-4xl space-y-8">
      <div>
        <h3 className="text-xl font-semibold mb-4">1. Install the SDK</h3>
        <ConsoleOutput showChrome title="bash" copyable copyText="npm install @orchyra/sdk">
          $ npm install @orchyra/sdk
        </ConsoleOutput>
      </div>
      <div>
        <h3 className="text-xl font-semibold mb-4">2. Initialize your client</h3>
        <ConsoleOutput showChrome title="index.ts" background="card" copyable>
          {`import { Orchyra } from '@orchyra/sdk'

const client = new Orchyra({
  apiKey: process.env.ORCHYRA_API_KEY
})`}
        </ConsoleOutput>
      </div>
      <div>
        <h3 className="text-xl font-semibold mb-4">3. Make your first request</h3>
        <ConsoleOutput showChrome title="index.ts" background="card" copyable>
          {`const response = await client.chat.completions.create({
  model: 'llama-3-8b',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
})

console.log(response.choices[0].message.content)`}
        </ConsoleOutput>
      </div>
    </div>
  ),
}
