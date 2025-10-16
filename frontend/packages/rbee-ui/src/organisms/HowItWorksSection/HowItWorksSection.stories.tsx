import type { Meta, StoryObj } from '@storybook/react'
import { developersHowItWorksProps } from '@rbee/ui/pages/DevelopersPage'
import { HowItWorksSection } from './HowItWorksSection'

const meta = {
  title: 'Organisms/HowItWorksSection',
  component: HowItWorksSection,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The HowItWorksSection breaks down the setup process into digestible steps with code examples. Each step includes a terminal or code block showing actual commands, making the process tangible and achievable.

## Marketing Strategy

### Target Audience
Technical visitors evaluating feasibility. They need:
- Proof that setup is achievable (not complex)
- Real commands/code (not hand-waving)
- Time estimate (15 minutes)
- Confidence that it works with their tools

### Primary Message
**"From zero to AI infrastructure in 15 minutes"** — Emphasizes speed and simplicity.

### Copy Analysis
- **Headline tone**: Confident, time-specific
- **Emotional appeal**: Reduces fear of complexity, builds confidence
- **Power words**: "15 minutes", "zero", "OpenAI-compatible", "just works"
- **Social proof**: Real commands build credibility

### Conversion Elements
- **Four steps**: Install → Add machines → Configure IDE → Build agents
- **Code examples**: Real bash/TypeScript showing actual usage
- **Copyable blocks**: Users can copy-paste commands
- **Progressive complexity**: Starts simple (install), ends powerful (build agents)

### Objection Handling
- **"Is it hard to set up?"** → Four simple steps with real commands
- **"How long does it take?"** → "15 minutes" in headline
- **"Will it work with my IDE?"** → Step 3 shows Zed & Cursor compatibility
- **"Can I build real things?"** → Step 4 shows TypeScript agent example

### Variations to Test
- Alternative headline: "Four Commands to Your Own AI Infrastructure"
- Alternative step order: Lead with IDE config (most familiar) vs. install
- Alternative final step: Show production deployment vs. agent building

## Composition
This organism contains:
- **Header**: Title and optional subtitle (via SectionContainer)
- **Steps**: Array of step objects with labels and code blocks
- **Step Badges**: Numbered circles for visual hierarchy
- **TerminalWindow**: Terminal/code blocks with syntax highlighting and copy buttons
- **Staggered Animations**: Steps animate in with delays

## When to Use
- Home page: After solution section, before features
- Getting Started page: As primary content
- Documentation: Quick start guide
- Landing pages: To reduce perceived complexity

## Content Requirements
- **Title**: Clear, time-specific if possible (required)
- **Steps**: 3-5 steps with labels and code blocks (required)
- **Code blocks**: Real, copyable commands
- **Step labels**: Action-oriented (Install, Add, Configure, Build)

## Usage in Commercial Site

### Home Page (/)
\`\`\`tsx
<HowItWorksSection
  title="From zero to AI infrastructure in 15 minutes"
  steps={[...]}
/>
\`\`\`

**Context**: Appears after HomeSolutionSection, before FeaturesSection  
**Purpose**: Reduce perceived complexity, show achievable path  
**Metrics**: Engagement indicator (do visitors read all steps?)

## Examples
\`\`\`tsx
import { HowItWorksSection } from '@rbee/ui/organisms'

// Custom title and steps
<HowItWorksSection
  title="Get started in minutes"
  subtitle="Four simple steps to your own AI infrastructure"
  steps={[
    {
      label: 'Install rbee',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: <div>curl -sSL https://rbee.dev/install.sh | sh</div>,
        copyText: 'curl -sSL https://rbee.dev/install.sh | sh',
      },
    },
    // ... more steps
  ]}
/>
\`\`\`

## Related Components
- TerminalWindow
- SectionContainer

## Accessibility
- **Keyboard Navigation**: All interactive elements are keyboard accessible
- **Focus States**: Visible focus indicators on copy buttons
- **Semantic HTML**: Proper heading hierarchy via SectionContainer
- **Motion**: Respects prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
- **Code Blocks**: Copyable for assistive technology users
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    title: {
      control: 'text',
      description: 'Section title (required)',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    subtitle: {
      control: 'text',
      description: 'Optional subtitle',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
  },
} satisfies Meta<typeof HowItWorksSection>

export default meta
type Story = StoryObj<typeof meta>

export const HomePageDefault: Story = {
  args: {
    title: 'From zero to AI infrastructure in 15 minutes',
    steps: [
      {
        label: 'Install rbee',
        block: {
          kind: 'terminal',
          title: 'terminal',
          lines: (
            <>
              <div>curl -sSL https://rbee.dev/install.sh | sh</div>
              <div className="text-[var(--syntax-comment)]">rbee-keeper daemon start</div>
            </>
          ),
          copyText: 'curl -sSL https://rbee.dev/install.sh | sh\nrbee-keeper daemon start',
        },
      },
      {
        label: 'Add your machines',
        block: {
          kind: 'terminal',
          title: 'terminal',
          lines: (
            <>
              <div>rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10</div>
              <div className="text-[var(--syntax-comment)]">
                rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20
              </div>
            </>
          ),
          copyText:
            'rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10\nrbee-keeper setup add-node --name mac --ssh-host 192.168.1.20',
        },
      },
      {
        label: 'Configure your IDE',
        block: {
          kind: 'terminal',
          title: 'terminal',
          lines: (
            <>
              <div>
                <span className="text-[var(--syntax-keyword)]">export</span> OPENAI_API_BASE=http://localhost:8080/v1
              </div>
              <div className="text-[var(--syntax-comment)]"># OpenAI-compatible endpoint — works with Zed & Cursor</div>
            </>
          ),
          copyText: 'export OPENAI_API_BASE=http://localhost:8080/v1',
        },
      },
      {
        label: 'Build AI agents',
        block: {
          kind: 'code',
          title: 'TypeScript',
          language: 'ts',
          code: `import { invoke } from '@rbee/utils';

const code = await invoke({
  prompt: 'Generate API from schema',
  model: 'llama-3.1-70b'
});`,
        },
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Headline**: "From zero to AI infrastructure in 15 minutes" — Time-specific reduces fear
- **Four steps**: Progressive complexity from install to agent building
  1. **Install rbee**: Single curl command, shows simplicity
  2. **Add your machines**: SSH-based discovery, shows multi-machine capability
  3. **Configure your IDE**: OpenAI-compatible endpoint, shows tool compatibility
  4. **Build AI agents**: TypeScript example, shows end-to-end capability

**Code Examples:**
- All blocks are copyable (reduces friction)
- Real commands (not pseudocode)
- Syntax highlighting for readability
- Comments explain context

**Conversion Strategy:**
- Positioned after solution section to show "how"
- Four steps feel achievable (not overwhelming)
- Final step shows powerful outcome (agents)
- No CTA (continues to features section)

**Tone**: Confident, technical, step-by-step guidance`,
      },
    },
  },
}

export const AlternativeSteps: Story = {
  args: {
    title: 'Four commands to your own AI infrastructure',
    subtitle: 'No complex configuration. No cloud accounts. Just your hardware.',
    steps: [
      {
        label: 'Install rbee-keeper',
        block: {
          kind: 'terminal',
          title: 'terminal',
          lines: (
            <>
              <div>curl -sSL https://rbee.dev/install.sh | sh</div>
              <div className="text-slate-400"># Installs orchestrator + worker daemon</div>
            </>
          ),
          copyText: 'curl -sSL https://rbee.dev/install.sh | sh',
        },
      },
      {
        label: 'Start the daemon',
        block: {
          kind: 'terminal',
          title: 'terminal',
          lines: (
            <>
              <div>rbee-keeper daemon start</div>
              <div className="text-slate-400"># Starts orchestrator on localhost:8080</div>
            </>
          ),
          copyText: 'rbee-keeper daemon start',
        },
      },
      {
        label: 'Test with curl',
        block: {
          kind: 'terminal',
          title: 'terminal',
          lines: (
            <>
              <div>curl http://localhost:8080/v1/models</div>
              <div className="text-slate-400"># Lists available models</div>
            </>
          ),
          copyText: 'curl http://localhost:8080/v1/models',
        },
      },
      {
        label: 'Use in your code',
        block: {
          kind: 'code',
          title: 'TypeScript',
          language: 'ts',
          code: `import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1'
});`,
        },
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story: `Alternative step sequence for A/B testing. This variant:
- Emphasizes "Four commands" (concrete, countable)
- Focuses on single-machine setup first (simpler)
- Includes explicit "Test with curl" step (validation)
- Shows OpenAI SDK integration (familiar pattern)
- Removes multi-machine complexity from initial flow

**Use case**: Test with developers who want fastest path to validation, not full multi-GPU setup.`,
      },
    },
  },
}

export const WithoutVisuals: Story = {
  args: {
    title: 'Three steps to private AI',
    steps: [
      {
        label: 'Install rbee-keeper',
        block: {
          kind: 'note',
          content: (
            <>
              <p>
                Run the install script on your primary machine. This installs the orchestrator daemon and worker
                process.
              </p>
              <p className="mt-2">
                <strong>Time:</strong> ~2 minutes
              </p>
            </>
          ),
        },
      },
      {
        label: 'Add your GPUs',
        block: {
          kind: 'note',
          content: (
            <>
              <p>Point rbee to other machines on your network via SSH or local discovery.</p>
              <p className="mt-2">
                <strong>Time:</strong> ~5 minutes
              </p>
            </>
          ),
        },
      },
      {
        label: 'Start building',
        block: {
          kind: 'note',
          content: (
            <>
              <p>
                Use the OpenAI-compatible API at <code>localhost:8080/v1</code>. Works with Zed, Cursor, Continue, and
                any OpenAI SDK.
              </p>
              <p className="mt-2">
                <strong>Time:</strong> Immediate
              </p>
            </>
          ),
        },
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story:
          'Text-only variant without code blocks. Useful for high-level overview or when code examples might overwhelm.',
      },
    },
  },
}

export const OnDevelopersPage: Story = {
  args: developersHowItWorksProps,
  parameters: {
    docs: {
      description: {
        story: '**Developers page context** — 15-minute setup guide with terminal commands and code examples.',
      },
    },
  },
}
