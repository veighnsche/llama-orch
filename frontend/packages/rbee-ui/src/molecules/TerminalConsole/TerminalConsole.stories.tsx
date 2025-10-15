import type { Meta, StoryObj } from '@storybook/react'
import { TerminalConsole } from './TerminalConsole'

const meta: Meta<typeof TerminalConsole> = {
  title: 'Molecules/TerminalConsole',
  component: TerminalConsole,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The TerminalConsole molecule displays a terminal-style console with a macOS-style top bar (traffic lights), title, content area, and optional footer.

## Composition
This molecule is composed of:
- **Top Bar**: macOS-style traffic lights (red, yellow, green) + title
- **Content**: Monospace font area for logs/commands
- **Footer**: Optional footer content

## When to Use
- Displaying command-line examples
- Showing log output
- Error messages in terminal format
- Code execution results
- System diagnostics

## Used In
- **ErrorHandling**: Displays error logs and diagnostic output
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    title: {
      control: 'text',
      description: 'Terminal window title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    children: {
      control: false,
      description: 'Terminal content (logs, commands, etc.)',
      table: {
        type: { summary: 'ReactNode' },
        category: 'Content',
      },
    },
    footer: {
      control: false,
      description: 'Optional footer content',
      table: {
        type: { summary: 'ReactNode' },
        category: 'Content',
      },
    },
    ariaLabel: {
      control: 'text',
      description: 'Accessible label for screen readers',
      table: {
        type: { summary: 'string' },
        category: 'Accessibility',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof TerminalConsole>

export const Default: Story = {
  args: {
    title: 'bash',
    children: (
      <div>
        <div className="text-chart-3">$ curl -X POST https://api.rbee.nl/v1/inference</div>
        <div className="text-muted-foreground mt-2">
          {'{'} "model": "llama-3.1-8b", "prompt": "Hello" {'}'}
        </div>
      </div>
    ),
  },
}

export const WithLogs: Story = {
  args: {
    title: 'worker-logs.txt',
    children: (
      <div className="space-y-1">
        <div className="text-chart-3">[2025-10-15 08:30:12] INFO: Worker started</div>
        <div className="text-chart-2">[2025-10-15 08:30:15] INFO: Model loaded: llama-3.1-8b</div>
        <div className="text-muted-foreground">[2025-10-15 08:30:18] INFO: Ready to accept requests</div>
        <div className="text-chart-3">[2025-10-15 08:30:22] INFO: Request received</div>
        <div className="text-chart-2">[2025-10-15 08:30:24] INFO: Inference complete (2.1s)</div>
      </div>
    ),
    ariaLabel: 'Worker logs console',
  },
}

export const WithErrors: Story = {
  args: {
    title: 'error-trace.log',
    children: (
      <div className="space-y-1">
        <div className="text-chart-3">[2025-10-15 08:45:10] INFO: Starting inference</div>
        <div className="text-destructive">[2025-10-15 08:45:12] ERROR: CUDA out of memory</div>
        <div className="text-muted-foreground ml-4">at model.forward(input_ids)</div>
        <div className="text-muted-foreground ml-4">RuntimeError: CUDA OOM</div>
        <div className="text-chart-4 mt-2">[2025-10-15 08:45:15] WARN: Retrying with smaller batch</div>
        <div className="text-chart-3">[2025-10-15 08:45:18] INFO: Inference complete</div>
      </div>
    ),
    footer: (
      <div className="text-xs text-muted-foreground">
        <span className="text-destructive">1 error</span>, <span className="text-chart-4">1 warning</span>
      </div>
    ),
    ariaLabel: 'Error trace console',
  },
}

export const InErrorHandlingContext: Story = {
  render: () => (
    <div className="w-full max-w-5xl">
      <div className="mb-4 text-sm text-muted-foreground">Example: TerminalConsole in ErrorHandling organism</div>
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold text-foreground mb-3">Live Error Logs</h3>
          <TerminalConsole
            title="worker-01.log"
            ariaLabel="Worker error logs"
            footer={
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Last updated: 2 seconds ago</span>
                <button className="text-primary hover:underline">Clear logs</button>
              </div>
            }
          >
            <div className="space-y-1">
              <div className="text-chart-3">[08:45:10] INFO: Request received</div>
              <div className="text-destructive">[08:45:12] ERROR: Model timeout (30s)</div>
              <div className="text-muted-foreground ml-4">Worker: gpu-worker-01</div>
              <div className="text-muted-foreground ml-4">Model: llama-3.1-70b</div>
              <div className="text-chart-4">[08:45:15] WARN: Falling back to smaller model</div>
              <div className="text-chart-3">[08:45:18] INFO: Retry successful</div>
            </div>
          </TerminalConsole>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'TerminalConsole as used in the ErrorHandling organism, showing live error logs with footer controls.',
      },
    },
  },
}
