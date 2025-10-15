import type { Meta, StoryObj } from '@storybook/react'
import { AlertTriangle, Shield, Zap } from 'lucide-react'
import { PlaybookHeader, PlaybookItem } from './PlaybookAccordion'

const meta: Meta<typeof PlaybookItem> = {
  title: 'Molecules/PlaybookAccordion',
  component: PlaybookItem,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
## Overview
The PlaybookAccordion molecule (PlaybookItem + PlaybookHeader) displays error handling playbooks with expandable/collapsible sections showing checks and severity indicators.

## Composition
This molecule is composed of:
- **IconBox**: Category icon
- **Title**: Playbook category name
- **Check Count**: Number of checks
- **Severity Dots**: Visual severity indicators
- **Description**: Category description
- **Checks List**: Expandable list of checks with severity and details
- **Footer**: Optional footer content

## When to Use
- Error handling documentation
- Operational playbooks
- Incident response guides
- System health checks

## Used In
- **ErrorHandling**: Displays error handling playbooks with categories
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    icon: {
      control: false,
      description: 'Lucide icon component',
      table: {
        type: { summary: 'LucideIcon' },
        category: 'Content',
      },
    },
    color: {
      control: 'select',
      options: ['primary', 'chart-2', 'chart-3', 'chart-4'],
      description: 'Icon color',
      table: {
        type: { summary: 'IconBoxProps["color"]' },
        category: 'Appearance',
      },
    },
    title: {
      control: 'text',
      description: 'Playbook title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    checkCount: {
      control: 'number',
      description: 'Number of checks',
      table: {
        type: { summary: 'number' },
        category: 'Content',
      },
    },
    severityDots: {
      control: 'object',
      description: 'Array of severity indicators',
      table: {
        type: { summary: "Array<'destructive' | 'primary' | 'chart-2' | 'chart-3'>" },
        category: 'Content',
      },
    },
    description: {
      control: 'text',
      description: 'Playbook description',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    checks: {
      control: 'object',
      description: 'Array of checks with severity, text, and detail',
      table: {
        type: { summary: 'Array<{severity, text, detail}>' },
        category: 'Content',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof PlaybookItem>

export const Default: Story = {
  args: {
    icon: AlertTriangle,
    color: 'chart-4',
    title: 'Model Loading Failures',
    checkCount: 4,
    severityDots: ['destructive', 'primary', 'chart-2', 'chart-3'],
    description: 'Diagnostics for model loading and initialization issues',
    checks: [
      { severity: 'destructive', text: 'Model file not found', detail: 'Check path' },
      { severity: 'primary', text: 'Insufficient VRAM', detail: 'Check GPU memory' },
      { severity: 'chart-2', text: 'Corrupted weights', detail: 'Verify checksum' },
      { severity: 'chart-3', text: 'Version mismatch', detail: 'Update runtime' },
    ],
    footer: (
      <div className="mt-4 text-xs text-muted-foreground">
        <strong>Resolution:</strong> Verify model path and GPU availability
      </div>
    ),
  },
}

export const WithSteps: Story = {
  args: {
    icon: Shield,
    color: 'primary',
    title: 'Authentication Errors',
    checkCount: 6,
    severityDots: ['destructive', 'destructive', 'primary', 'primary', 'chart-2', 'chart-3'],
    description: 'Token validation and authentication failure diagnostics',
    checks: [
      { severity: 'destructive', text: 'Token expired', detail: 'Refresh required' },
      { severity: 'destructive', text: 'Invalid signature', detail: 'Check secret' },
      { severity: 'primary', text: 'Missing claims', detail: 'Verify payload' },
      { severity: 'primary', text: 'Audience mismatch', detail: 'Check aud claim' },
      { severity: 'chart-2', text: 'Clock skew', detail: 'Sync time' },
      { severity: 'chart-3', text: 'Issuer unknown', detail: 'Verify iss claim' },
    ],
    footer: (
      <div className="mt-4 rounded-lg bg-muted/50 p-3 text-xs">
        <strong>Quick Fix:</strong> Ensure system time is synchronized and refresh tokens are rotated properly.
      </div>
    ),
  },
}

export const Expanded: Story = {
  render: () => (
    <div className="w-full max-w-4xl">
      <details open className="group border-b border-border">
        <summary className="flex items-center justify-between gap-3 cursor-pointer px-5 py-4 hover:bg-muted/50 transition-colors">
          <span className="flex items-center gap-3">
            <div className="rounded-lg bg-chart-3/10 p-2">
              <Zap className="h-4 w-4 text-chart-3" />
            </div>
            <span className="font-semibold text-foreground">Performance Issues</span>
            <span className="ml-2 text-xs text-muted-foreground">5 checks</span>
          </span>
          <span className="hidden sm:flex items-center gap-2 mr-1">
            <span className="size-1.5 rounded-full bg-primary/80" />
            <span className="size-1.5 rounded-full bg-chart-2/80" />
            <span className="size-1.5 rounded-full bg-chart-3/80" />
            <span className="text-muted-foreground text-xs">severity</span>
            <span className="text-muted-foreground text-xs group-open:rotate-180 transition-transform">▾</span>
          </span>
        </summary>
        <div className="px-6 pb-5 border-t border-border/80">
          <p className="text-sm text-muted-foreground/90 mb-3 mt-3">
            Diagnostics for slow inference and throughput issues
          </p>
          <ul className="grid sm:grid-cols-2 gap-x-6 gap-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="mt-2 size-1.5 rounded-full bg-primary/80 shrink-0" />
              <span>
                High latency <span className="text-muted-foreground">(Check network)</span>
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-2 size-1.5 rounded-full bg-chart-2/80 shrink-0" />
              <span>
                GPU throttling <span className="text-muted-foreground">(Check temperature)</span>
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-2 size-1.5 rounded-full bg-chart-3/80 shrink-0" />
              <span>
                Memory pressure <span className="text-muted-foreground">(Check VRAM)</span>
              </span>
            </li>
          </ul>
        </div>
      </details>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'PlaybookItem in expanded state showing all checks.',
      },
    },
  },
}

export const InErrorHandlingContext: Story = {
  render: () => (
    <div className="w-full max-w-5xl">
      <div className="mb-4 text-sm text-muted-foreground">Example: PlaybookAccordion in ErrorHandling organism</div>
      <div className="rounded-2xl border bg-card overflow-hidden">
        <PlaybookHeader
          title="Error Playbooks"
          description="24 automated checks"
          filterCategories={['Critical', 'Performance', 'Auth']}
          onExpandAll={() => {}}
          onCollapseAll={() => {}}
        />
        <PlaybookItem
          icon={AlertTriangle}
          color="chart-4"
          title="Model Loading Failures"
          checkCount={4}
          severityDots={['destructive', 'primary', 'chart-2', 'chart-3']}
          description="Diagnostics for model loading and initialization issues"
          checks={[
            { severity: 'destructive', text: 'Model file not found', detail: 'Check path' },
            { severity: 'primary', text: 'Insufficient VRAM', detail: 'Check GPU memory' },
          ]}
          footer={<div className="mt-4 text-xs text-muted-foreground">Auto-retry enabled</div>}
        />
        <PlaybookItem
          icon={Shield}
          color="primary"
          title="Authentication Errors"
          checkCount={3}
          severityDots={['destructive', 'primary', 'chart-2']}
          description="Token validation and authentication failure diagnostics"
          checks={[
            { severity: 'destructive', text: 'Token expired', detail: 'Refresh required' },
            { severity: 'primary', text: 'Invalid signature', detail: 'Check secret' },
          ]}
          footer={<div className="mt-4 text-xs text-muted-foreground">Token refresh available</div>}
        />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'PlaybookAccordion as used in the ErrorHandling organism with header and multiple items.',
      },
    },
  },
}
