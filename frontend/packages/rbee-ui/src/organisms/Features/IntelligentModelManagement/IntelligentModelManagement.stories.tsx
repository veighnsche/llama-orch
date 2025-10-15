import type { Meta, StoryObj } from '@storybook/react'
import { IntelligentModelManagement } from './IntelligentModelManagement'

const meta = {
  title: 'Organisms/Features/IntelligentModelManagement',
  component: IntelligentModelManagement,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The IntelligentModelManagement section explains rbee's automatic model provisioning, caching, and validation capabilities. It shows how rbee downloads models from Hugging Face, verifies checksums, caches locally, and validates resources before loading to prevent failures.

## Composition
This organism contains:
- **Section Title**: "Intelligent Model Management"
- **Section Subtitle**: "Automatic model provisioning, caching, and validation. Download once; use everywhere."
- **Badge**: "Provision • Cache • Validate"
- **Two Cards**:
  1. **Automatic Model Catalog**: Terminal showing download progress, checksum verification
  2. **Resource Preflight Checks**: Checklist of validation steps (RAM, VRAM, disk, backend)

## Marketing Strategy

### Target Sub-Audience
**Primary**: Developers and technical users concerned about reliability
**Secondary**: Teams evaluating operational complexity

### Page-Specific Messaging
- **Features page**: Technical deep dive into model management
- **Technical level**: Intermediate to Advanced
- **Focus**: Reliability, automation, fail-fast design

### Copy Analysis
- **Technical level**: Intermediate to Advanced
- **Card 1 - Automatic Model Catalog**:
  - Benefit: "Download once; use everywhere"
  - Proof: Terminal showing download progress (20% → 100%), checksum verification
  - Features: Checksum validation (SHA256), resume support, SQLite catalog
- **Card 2 - Resource Preflight Checks**:
  - Benefit: "Fail fast with clear errors—no mystery crashes"
  - Proof: Checklist (RAM check, VRAM check, disk space, backend availability)
  - Features: Validates resources before loading

### Conversion Elements
- **Reliability**: "No mystery crashes" (reduces fear of operational issues)
- **Automation**: "Automatic provisioning" (reduces operational burden)
- **Intelligence**: "Preflight checks" (shows thoughtful design)

## When to Use
- On the Features page after CrossNodeOrchestration
- To explain model management capabilities
- To demonstrate reliability and automation
- To show fail-fast design philosophy

## Content Requirements
- **Section Title**: Clear heading
- **Section Subtitle**: Brief overview
- **Badge**: Feature category
- **Terminal Example**: Download progress with animations
- **Checklist**: Validation steps with icons
- **Feature Strips**: Key benefits (3 per card)

## Variants
- **Default**: Full section with both cards
- **Auto-Download Focus**: Emphasis on automatic provisioning
- **Caching Focus**: Emphasis on caching benefits

## Examples
\`\`\`tsx
import { IntelligentModelManagement } from '@rbee/ui/organisms/Features/IntelligentModelManagement'

<IntelligentModelManagement />
\`\`\`

## Used In
- Features page (\`/features\`)

## Technical Implementation
- Uses SectionContainer for consistent layout
- IconBox for feature icons
- Animated progress bars (animate-in grow-in)
- Staggered animations (delay-75, delay-150, etc.)

## Related Components
- SectionContainer
- IconBox
- Badge
- CheckCircle2 (Lucide)

## Accessibility
- **Keyboard Navigation**: All interactive elements keyboard accessible
- **ARIA Labels**: Terminal has aria-label="Model download and validation log" and aria-live="polite"
- **Semantic HTML**: Uses proper heading hierarchy
- **Progress Bars**: Visual progress bars for download status
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof IntelligentModelManagement>

export default meta
type Story = StoryObj<typeof meta>

export const FeaturesPageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default intelligent model management section for the Features page. Shows automatic model catalog (download progress, checksum verification) and resource preflight checks (RAM, VRAM, disk, backend validation). Demonstrates reliability and automation.',
      },
    },
  },
}

export const AutoDownloadFocus: Story = {
  render: () => (
    <div className="space-y-8">
      <div className="bg-primary/10 p-6 text-center">
        <h3 className="text-xl font-bold">Automatic Download Focus</h3>
        <p className="text-muted-foreground">
          Request any model from Hugging Face. rbee downloads, verifies, and caches automatically.
        </p>
      </div>
      <IntelligentModelManagement />
      <div className="bg-muted p-8">
        <h3 className="text-xl font-bold mb-4 text-center">Pain Point Addressed</h3>
        <div className="max-w-2xl mx-auto space-y-4 text-sm">
          <div className="bg-background p-4 rounded-lg">
            <strong className="block mb-2">Without rbee:</strong>
            <ul className="space-y-1 text-muted-foreground">
              <li>• Manually download models from Hugging Face</li>
              <li>• Verify checksums manually (or skip and risk corruption)</li>
              <li>• Manage local cache manually</li>
              <li>• Re-download if interrupted</li>
            </ul>
          </div>
          <div className="bg-background p-4 rounded-lg">
            <strong className="block mb-2">With rbee:</strong>
            <ul className="space-y-1 text-muted-foreground">
              <li>• Request model by name</li>
              <li>• rbee handles download, verification, caching</li>
              <li>• Resume support for interrupted downloads</li>
              <li>• Never download the same model twice</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'Focus on automatic download capabilities. Compares manual workflow (without rbee) vs. automated workflow (with rbee).',
      },
    },
  },
}

export const CachingFocus: Story = {
  render: () => (
    <div className="space-y-8">
      <IntelligentModelManagement />
      <div className="bg-muted p-8">
        <h3 className="text-xl font-bold mb-4 text-center">Caching Benefits</h3>
        <div className="max-w-3xl mx-auto space-y-4 text-sm">
          <p className="text-muted-foreground">The SQLite catalog and local caching provide significant benefits:</p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-background p-4 rounded-lg">
              <strong className="block mb-2">Speed</strong>
              <p className="text-muted-foreground">Fast lookups. No need to re-download models you already have.</p>
            </div>
            <div className="bg-background p-4 rounded-lg">
              <strong className="block mb-2">Bandwidth</strong>
              <p className="text-muted-foreground">
                Save bandwidth. Download once, use everywhere across your network.
              </p>
            </div>
            <div className="bg-background p-4 rounded-lg">
              <strong className="block mb-2">Reliability</strong>
              <p className="text-muted-foreground">No duplicates. Checksum verification ensures integrity.</p>
            </div>
          </div>
          <div className="bg-background p-4 rounded-lg">
            <strong className="block mb-2">Key Differentiator:</strong>
            <p className="text-muted-foreground">
              "Download once; use everywhere" means you can share models across all machines in your pool without
              re-downloading. The SQLite catalog tracks what's available where.
            </p>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'Focus on caching benefits: speed, bandwidth savings, reliability. Emphasizes "download once; use everywhere" value proposition.',
      },
    },
  },
}
