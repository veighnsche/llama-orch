// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { CodeSnippet } from './CodeSnippet'

const meta: Meta<typeof CodeSnippet> = {
  title: 'Atoms/CodeSnippet',
  component: CodeSnippet,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A component for displaying inline or small block code snippets with proper monospace font and styling.

## Features
- Uses Geist Mono font for proper code display
- Inline variant for use within text
- Block variant for standalone snippets
- Proper text selection and copy support

## Used In
- StepsSection organism
- Documentation pages
- Technical content
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['inline', 'block'],
      description: 'Visual variant of the code snippet',
    },
    children: {
      control: 'text',
      description: 'Code content to display',
    },
    className: {
      control: 'text',
      description: 'Additional CSS classes',
    },
  },
}

export default meta
type Story = StoryObj<typeof CodeSnippet>

export const Default: Story = {
  args: {
    children: 'npm install @rbee/sdk',
  },
}

export const WithLanguage: Story = {
  render: () => (
    <div className="flex flex-col gap-4 max-w-2xl">
      <div>
        <p className="text-sm text-muted-foreground mb-2">Bash command:</p>
        <CodeSnippet variant="block">curl -sSL rbee.dev/install.sh | sh</CodeSnippet>
      </div>
      <div>
        <p className="text-sm text-muted-foreground mb-2">Python import:</p>
        <CodeSnippet variant="block">from rbee import Client</CodeSnippet>
      </div>
      <div>
        <p className="text-sm text-muted-foreground mb-2">TypeScript usage:</p>
        <CodeSnippet variant="block">const client = new RbeeClient()</CodeSnippet>
      </div>
    </div>
  ),
}

export const WithCopy: Story = {
  render: () => (
    <div className="flex flex-col gap-4 max-w-2xl">
      <div>
        <p className="text-sm mb-2">Inline code in text:</p>
        <p className="text-sm">
          Run <CodeSnippet>npm install</CodeSnippet> to get started with the SDK.
        </p>
      </div>
      <div>
        <p className="text-sm mb-2">Block code snippet:</p>
        <CodeSnippet variant="block">
          export RBEE_API_KEY=your_api_key_here{'\n'}
          rbee deploy --model llama-3.1-8b
        </CodeSnippet>
      </div>
    </div>
  ),
}

export const InStepsContext: Story = {
  render: () => (
    <div className="max-w-3xl space-y-6">
      <h3 className="text-lg font-semibold">Quick Start Guide</h3>
      <div className="space-y-4">
        <div>
          <h4 className="text-sm font-medium mb-2">Step 1: Install the SDK</h4>
          <CodeSnippet variant="block">npm install @rbee/sdk</CodeSnippet>
        </div>
        <div>
          <h4 className="text-sm font-medium mb-2">Step 2: Initialize the client</h4>
          <CodeSnippet variant="block">
            import {'{ RbeeClient }'} from '@rbee/sdk'{'\n'}
            {'\n'}
            const client = new RbeeClient({'{'}
            {'\n'} apiKey: process.env.RBEE_API_KEY{'\n'}
            {'}'})
          </CodeSnippet>
        </div>
        <div>
          <h4 className="text-sm font-medium mb-2">Step 3: Make your first request</h4>
          <CodeSnippet variant="block">
            const response = await client.chat.completions.create({'{'}
            {'\n'} model: 'llama-3.1-8b',{'\n'} messages: [{'{ role: "user", content: "Hello!" }'}]{'\n'}
            {'}'})
          </CodeSnippet>
        </div>
      </div>
    </div>
  ),
}

export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-col gap-6 max-w-2xl">
      <div>
        <h4 className="text-sm font-semibold mb-3">Inline Variant</h4>
        <p className="text-sm">
          Use <CodeSnippet>inline</CodeSnippet> variant for code within text, like{' '}
          <CodeSnippet>npm install</CodeSnippet> or <CodeSnippet>const x = 42</CodeSnippet>.
        </p>
      </div>
      <div>
        <h4 className="text-sm font-semibold mb-3">Block Variant</h4>
        <CodeSnippet variant="block">
          # This is a block code snippet{'\n'}# It supports multiple lines{'\n'}
          curl -X POST https://api.rbee.dev/v1/chat/completions \{'\n'}
          -H "Authorization: Bearer $RBEE_API_KEY" \{'\n'}
          -d '{`{"model": "llama-3.1-8b"}`}'
        </CodeSnippet>
      </div>
    </div>
  ),
}
