import type { Meta, StoryObj } from '@storybook/react'
import { DevelopersHowItWorks } from './DevelopersHowItWorks'

const meta = {
  title: 'Organisms/Developers/DevelopersHowItWorks',
  component: DevelopersHowItWorks,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The DevelopersHowItWorks section explains the technical workflow for developers using rbee. It shows the step-by-step process from installation to inference, with developer-specific technical details.

## Composition
This organism contains:
- **Section Title**: "How It Works"
- **Step-by-step Process**: Visual workflow showing the technical steps
- **Technical Details**: Specific commands, APIs, and integration points
- **Visual Aids**: Diagrams or illustrations showing the architecture

## Marketing Strategy

### Target Sub-Audience
**Primary**: Developers evaluating rbee (need to understand the technical workflow)
**Secondary**: Technical decision makers (CTOs, lead engineers)

### Page-Specific Messaging
- **Developers page**: More technical depth than home page version
- **Technical level**: Intermediate to Advanced
- **Focus**: Actual implementation details, not just high-level concepts

### Copy Analysis
- **Technical level**: Intermediate to Advanced
- **Details**: Specific commands, API endpoints, configuration
- **Proof points**: Shows it's technically feasible and well-designed

### Conversion Elements
- **Educational**: Reduces uncertainty about implementation
- **Technical credibility**: Shows the system is well-architected
- **Reduces friction**: "This is how it actually works"

## When to Use
- On the Developers page after the solution section
- To provide technical depth for evaluating developers
- To show the system architecture and workflow

## Content Requirements
- **Title**: Clear section heading
- **Steps**: Detailed technical workflow
- **Visual aids**: Architecture diagrams or illustrations
- **Technical details**: Commands, APIs, configuration

## Examples
\`\`\`tsx
import { DevelopersHowItWorks } from '@rbee/ui/organisms/Developers/DevelopersHowItWorks'

<DevelopersHowItWorks />
\`\`\`

## Used In
- Developers page (\`/developers\`)

## Related Components
- Home page HowItWorksSection
- StepsSection

## Accessibility
- **Keyboard Navigation**: All interactive elements keyboard accessible
- **ARIA Labels**: Proper labels on diagrams and steps
- **Semantic HTML**: Uses proper heading hierarchy
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof DevelopersHowItWorks>

export default meta
type Story = StoryObj<typeof meta>

export const DevelopersPageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default "How It Works" section for the Developers page. Shows the technical workflow with developer-specific details.',
      },
    },
  },
}

export const SimplifiedFlow: Story = {
  render: () => (
    <div className="space-y-8">
      <DevelopersHowItWorks />
      <div className="bg-muted p-8">
        <h3 className="text-xl font-bold mb-4 text-center">Alternative: Simplified 3-Step Flow</h3>
        <div className="max-w-2xl mx-auto space-y-4">
          <p className="text-sm text-muted-foreground text-center">
            For landing pages or quick overviews, consider a simplified 3-step flow that focuses on the essential
            actions.
          </p>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Alternative simplified workflow with fewer steps for landing pages or quick overviews.',
      },
    },
  },
}

export const TechnicalDepth: Story = {
  render: () => (
    <div className="space-y-8">
      <DevelopersHowItWorks />
      <div className="bg-muted p-8">
        <h3 className="text-xl font-bold mb-4 text-center">Technical Depth vs. Home Page</h3>
        <div className="max-w-3xl mx-auto">
          <p className="text-sm text-muted-foreground mb-4">
            The Developers page version should include more technical details than the home page version:
          </p>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="bg-background p-4 rounded-lg">
              <strong>Home Page:</strong>
              <ul className="mt-2 space-y-1 text-muted-foreground">
                <li>• High-level concepts</li>
                <li>• Accessible to non-technical</li>
                <li>• Focus on benefits</li>
              </ul>
            </div>
            <div className="bg-background p-4 rounded-lg">
              <strong>Developers Page:</strong>
              <ul className="mt-2 space-y-1 text-muted-foreground">
                <li>• Specific commands and APIs</li>
                <li>• Technical architecture details</li>
                <li>• Focus on implementation</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Comparison of technical depth: Developers page version should be more detailed than home page version.',
      },
    },
  },
}
