import type { Meta, StoryObj } from '@storybook/react'
import { Activity, AlertTriangle, Network, Server } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
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
The PlaybookAccordion molecule displays error handling playbooks in a scannable Check → Meaning → Action format with single-open accordion behavior.

## Composition
- **SeverityDot**: Color-coded severity indicator atom
- **CheckRow**: 3-column grid (Check | Meaning | Action) molecule
- **SeverityLegend**: Always-visible legend explaining severity dots
- **PlaybookHeader**: Badge, description, filter toggles, expand/collapse controls
- **PlaybookItem**: Single category with checks in structured rows

## Key Features
- **Instant scannability**: 3-column layout eliminates paragraph soup
- **Single-open behavior**: Auto-closes other categories to prevent wall-of-text
- **Filter toggles**: Show/hide categories via toggle buttons with aria-pressed
- **Severity legend**: Persistent legend makes dots meaningful at a glance
- **Accessibility**: Keyboard navigation, live region announcements, focus management

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
        type: {
          summary: "Array<'destructive' | 'primary' | 'chart-2' | 'chart-3'>",
        },
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
    icon: Network,
    color: 'primary',
    title: 'Network & Connectivity',
    checkCount: 4,
    severityDots: ['destructive', 'primary', 'primary', 'chart-2'],
    description: 'Detects timeouts, auth failures, HTTP errors; retries with exponential backoff.',
    checks: [
      {
        severity: 'destructive',
        title: 'SSH connection timeout',
        meaning: "Remote host didn't respond within threshold; likely egress or firewall.",
        actionLabel: 'Retry with jitter',
        href: '#retry',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'primary',
        title: 'SSH authentication failure',
        meaning: 'Keys or agents rejected by server.',
        actionLabel: 'Open fix steps',
        href: '#fix',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'primary',
        title: 'HTTP connection failures',
        meaning: 'Auto-retry on transient TCP resets.',
        actionLabel: 'Enable auto-retry',
        href: '#enable',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'chart-2',
        title: 'Connection loss during inference',
        meaning: 'Stream dropped; partial results were saved.',
        actionLabel: 'Resume stream',
        href: '#resume',
        guideLabel: 'Timeline',
        guideHref: '#timeline',
      },
    ],
    footer: (
      <div className="text-xs text-muted-foreground">
        <strong>Auto-recovery:</strong> Exponential backoff with jitter enabled
      </div>
    ),
  },
}

export const ResourceErrors: Story = {
  args: {
    icon: Server,
    color: 'chart-2',
    title: 'Resource Errors',
    checkCount: 4,
    severityDots: ['destructive', 'destructive', 'primary', 'primary'],
    description: 'Fail fast on RAM/VRAM limits with safe aborts and pre-download disk checks.',
    checks: [
      {
        severity: 'destructive',
        title: 'Insufficient RAM',
        meaning: 'Process exceeds available memory; swap thrashing.',
        actionLabel: 'Lower batch / increase memory',
        href: '#memory',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'destructive',
        title: 'VRAM exhausted',
        meaning: 'GPU out of memory; no CPU fallback configured.',
        actionLabel: 'Use smaller precision',
        href: '#precision',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'primary',
        title: 'Disk space pre-check',
        meaning: 'Download blocked to prevent low-disk failures.',
        actionLabel: 'Choose cache path',
        href: '#cache',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'primary',
        title: 'OOM during model load',
        meaning: 'Abort safely before corrupted state.',
        actionLabel: 'Stream weights',
        href: '#stream',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
    ],
  },
}

export const ModelBackend: Story = {
  args: {
    icon: AlertTriangle,
    color: 'chart-4',
    title: 'Model & Backend',
    checkCount: 4,
    severityDots: ['destructive', 'primary', 'primary', 'chart-2'],
    description: 'Validate model presence, credentials, and ready inference endpoints.',
    checks: [
      {
        severity: 'destructive',
        title: 'Model 404 (Hugging Face link)',
        meaning: 'Requested model not found or renamed.',
        actionLabel: 'Select available model',
        href: '#select',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'primary',
        title: 'Download failures (resume support)',
        meaning: 'Interrupted download with resumable chunks.',
        actionLabel: 'Resume now',
        href: '#resume',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'primary',
        title: 'Private model 403',
        meaning: 'Token lacks permission for repository.',
        actionLabel: 'Fix token scope',
        href: '#token',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'chart-2',
        title: 'Backend not available',
        meaning: 'Health probe failing; show alternatives.',
        actionLabel: 'Switch endpoint',
        href: '#switch',
        guideLabel: 'Alternatives',
        guideHref: '#alternatives',
      },
    ],
  },
}

export const ProcessLifecycle: Story = {
  args: {
    icon: Activity,
    color: 'chart-3',
    title: 'Process Lifecycle',
    checkCount: 4,
    severityDots: ['destructive', 'primary', 'chart-2', 'chart-3'],
    description: 'Watch workers from startup to shutdown with safe termination.',
    checks: [
      {
        severity: 'destructive',
        title: 'Worker binary missing',
        meaning: 'Install steps incomplete; worker cannot spawn.',
        actionLabel: 'Run installer',
        href: '#install',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'primary',
        title: 'Crash during startup',
        meaning: 'Read early log pointers for root cause.',
        actionLabel: 'Open logs',
        href: '#logs',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'chart-2',
        title: 'Graceful shutdown',
        meaning: 'Drain active requests before exit.',
        actionLabel: 'Send SIGTERM',
        href: '#sigterm',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
      {
        severity: 'chart-3',
        title: 'Force-kill after 30s',
        meaning: 'Timeout guard to prevent hung exits.',
        actionLabel: 'Adjust timeout',
        href: '#timeout',
        guideLabel: 'Guide',
        guideHref: '#guide',
      },
    ],
  },
}

export const FullPlaybook: Story = {
  render: () => {
    const [selectedCategories, setSelectedCategories] = useState<string[]>([])
    const [allExpanded, setAllExpanded] = useState(false)
    const [openCategory, setOpenCategory] = useState<string | null>(null)
    const detailsRefs = useRef<Record<string, HTMLDetailsElement | null>>({})

    const categories = [
      {
        id: 'network',
        icon: Network,
        color: 'primary' as const,
        title: 'Network & Connectivity',
        checkCount: 4,
        severityDots: ['destructive', 'primary', 'primary', 'chart-2'] as const,
        description: 'Detects timeouts, auth failures, HTTP errors; retries with exponential backoff.',
        checks: [
          {
            severity: 'destructive' as const,
            title: 'SSH connection timeout',
            meaning: "Remote host didn't respond within threshold; likely egress or firewall.",
            actionLabel: 'Retry with jitter',
            href: '#retry',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'primary' as const,
            title: 'SSH authentication failure',
            meaning: 'Keys or agents rejected by server.',
            actionLabel: 'Open fix steps',
            href: '#fix',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'primary' as const,
            title: 'HTTP connection failures',
            meaning: 'Auto-retry on transient TCP resets.',
            actionLabel: 'Enable auto-retry',
            href: '#enable',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'chart-2' as const,
            title: 'Connection loss during inference',
            meaning: 'Stream dropped; partial results were saved.',
            actionLabel: 'Resume stream',
            href: '#resume',
            guideLabel: 'Timeline',
            guideHref: '#timeline',
          },
        ],
      },
      {
        id: 'resource',
        icon: Server,
        color: 'chart-2' as const,
        title: 'Resource Errors',
        checkCount: 4,
        severityDots: ['destructive', 'destructive', 'primary', 'primary'] as const,
        description: 'Fail fast on RAM/VRAM limits with safe aborts and pre-download disk checks.',
        checks: [
          {
            severity: 'destructive' as const,
            title: 'Insufficient RAM',
            meaning: 'Process exceeds available memory; swap thrashing.',
            actionLabel: 'Lower batch / increase memory',
            href: '#memory',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'destructive' as const,
            title: 'VRAM exhausted',
            meaning: 'GPU out of memory; no CPU fallback configured.',
            actionLabel: 'Use smaller precision',
            href: '#precision',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'primary' as const,
            title: 'Disk space pre-check',
            meaning: 'Download blocked to prevent low-disk failures.',
            actionLabel: 'Choose cache path',
            href: '#cache',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'primary' as const,
            title: 'OOM during model load',
            meaning: 'Abort safely before corrupted state.',
            actionLabel: 'Stream weights',
            href: '#stream',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
        ],
      },
      {
        id: 'model',
        icon: AlertTriangle,
        color: 'chart-4' as const,
        title: 'Model & Backend',
        checkCount: 4,
        severityDots: ['destructive', 'primary', 'primary', 'chart-2'] as const,
        description: 'Validate model presence, credentials, and ready inference endpoints.',
        checks: [
          {
            severity: 'destructive' as const,
            title: 'Model 404 (Hugging Face link)',
            meaning: 'Requested model not found or renamed.',
            actionLabel: 'Select available model',
            href: '#select',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'primary' as const,
            title: 'Download failures (resume support)',
            meaning: 'Interrupted download with resumable chunks.',
            actionLabel: 'Resume now',
            href: '#resume',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'primary' as const,
            title: 'Private model 403',
            meaning: 'Token lacks permission for repository.',
            actionLabel: 'Fix token scope',
            href: '#token',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'chart-2' as const,
            title: 'Backend not available',
            meaning: 'Health probe failing; show alternatives.',
            actionLabel: 'Switch endpoint',
            href: '#switch',
            guideLabel: 'Alternatives',
            guideHref: '#alternatives',
          },
        ],
      },
      {
        id: 'process',
        icon: Activity,
        color: 'chart-3' as const,
        title: 'Process Lifecycle',
        checkCount: 4,
        severityDots: ['destructive', 'primary', 'chart-2', 'chart-3'] as const,
        description: 'Watch workers from startup to shutdown with safe termination.',
        checks: [
          {
            severity: 'destructive' as const,
            title: 'Worker binary missing',
            meaning: 'Install steps incomplete; worker cannot spawn.',
            actionLabel: 'Run installer',
            href: '#install',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'primary' as const,
            title: 'Crash during startup',
            meaning: 'Read early log pointers for root cause.',
            actionLabel: 'Open logs',
            href: '#logs',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'chart-2' as const,
            title: 'Graceful shutdown',
            meaning: 'Drain active requests before exit.',
            actionLabel: 'Send SIGTERM',
            href: '#sigterm',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
          {
            severity: 'chart-3' as const,
            title: 'Force-kill after 30s',
            meaning: 'Timeout guard to prevent hung exits.',
            actionLabel: 'Adjust timeout',
            href: '#timeout',
            guideLabel: 'Guide',
            guideHref: '#guide',
          },
        ],
      },
    ]

    const handleFilterToggle = (category: string) => {
      setSelectedCategories((prev) => {
        if (prev.includes(category)) {
          return prev.filter((c) => c !== category)
        }
        return [...prev, category]
      })
    }

    const handleExpandAll = () => {
      Object.values(detailsRefs.current).forEach((ref) => {
        if (ref) ref.open = true
      })
      setAllExpanded(true)
    }

    const handleCollapseAll = () => {
      Object.values(detailsRefs.current).forEach((ref) => {
        if (ref) ref.open = false
      })
      setAllExpanded(false)
      setOpenCategory(null)
    }

    const handleToggle = (categoryId: string, isOpen: boolean) => {
      if (isOpen) {
        Object.entries(detailsRefs.current).forEach(([id, ref]) => {
          if (id !== categoryId && ref) {
            ref.open = false
          }
        })
        setOpenCategory(categoryId)
      } else {
        setOpenCategory(null)
      }
    }

    useEffect(() => {
      const allOpen = Object.values(detailsRefs.current).every((ref) => ref?.open)
      setAllExpanded(allOpen)
    }, [openCategory])

    const visibleCategories =
      selectedCategories.length === 0
        ? categories
        : categories.filter((cat) => selectedCategories.includes(cat.title.split(' ')[0]))

    return (
      <div className="w-full max-w-5xl">
        <div className="rounded-2xl border border-border bg-card overflow-hidden">
          <PlaybookHeader
            title="Playbook"
            description="19+ scenarios · 4 categories"
            filterCategories={['Network', 'Resource', 'Model', 'Process']}
            selectedCategories={selectedCategories}
            onFilterToggle={handleFilterToggle}
            onExpandAll={handleExpandAll}
            onCollapseAll={handleCollapseAll}
            allExpanded={allExpanded}
          />
          {visibleCategories.map((category) => (
            <PlaybookItem key={category.id} {...category} onToggle={(isOpen) => handleToggle(category.id, isOpen)} />
          ))}
        </div>
      </div>
    )
  },
  parameters: {
    docs: {
      description: {
        story:
          'Complete playbook with all 4 categories, filter toggles, single-open behavior, and severity legend. Demonstrates the full Check → Meaning → Action layout.',
      },
    },
  },
}
