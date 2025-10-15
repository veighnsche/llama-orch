import type { Meta, StoryObj } from '@storybook/react'
import { UseCasesIndustry } from './UseCasesIndustry'

const meta = {
  title: 'Organisms/UseCases/UseCasesIndustry',
  component: UseCasesIndustry,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The UseCasesIndustry section showcases 6 industry-specific use cases (Financial Services, Healthcare, Legal, Government, Education, Manufacturing) in a 3-column grid layout. Each industry card highlights compliance requirements (GDPR, HIPAA, ITAR, FERPA) and specific use cases for that sector. Includes filter navigation and hero visual.

## Use Case Storytelling

### Industry-Specific Narrative Structure
1. **Industry:** Which sector? (e.g., "Financial Services")
2. **Compliance:** What regulations? (e.g., "GDPR-ready with audit trails")
3. **Use Cases:** What AI applications? (e.g., "AI code review and risk analysis")
4. **Data Residency:** How is data protected? (e.g., "Without sending financial data to external APIs")

### Emotional Journey (Per Industry)
- **Before:** Compliance concerns, data residency requirements, vendor lock-in, external API risks
- **After:** Compliant AI infrastructure, data stays on-premises, full audit trails, zero external dependencies

### Compliance Focus (Per Industry)
- **Financial Services:** GDPR, audit trails, data residency
- **Healthcare:** HIPAA, patient data protection
- **Legal:** Attorney-client privilege, document confidentiality
- **Government:** ITAR, sovereign infrastructure, no foreign cloud
- **Education:** FERPA, student information protection
- **Manufacturing:** IP protection, trade secret safeguarding

## Composition
This organism contains:
- **SectionContainer**: "Industry-Specific Solutions" with subtitle
- **Header Block**:
  - Eyebrow: "Regulated sectors · Private-by-design"
  - Hero Visual: IndustriesHero illustration
  - Filter Pills: All, Finance, Healthcare, Legal, Public Sector, Education, Manufacturing
- **Industries Grid** (3-column responsive):
  - 6 IndustryCard components
  - Each with icon, color, title, compliance badge, copy, anchor

## When to Use
- On the Use Cases page after primary use cases section
- To address industry-specific compliance and security requirements
- To target regulated sectors with specific needs

## Content Requirements
- **Industries:** Must represent regulated sectors
- **Compliance Badges:** Specific regulations (GDPR, HIPAA, ITAR, FERPA)
- **Use Cases:** Industry-specific AI applications
- **Data Residency:** Clear explanation of data protection

## Marketing Strategy
- **Target Audience:** Organizations in regulated industries
- **Primary Message:** "rbee meets your industry's compliance requirements"
- **Emotional Appeal:** Confidence (compliant) + Security (data stays on-premises)
- **Copy Tone:** Professional, compliance-focused, security-oriented

## Variants
- **Default**: All 6 industries with filter navigation
- **SoftwareDevFocus**: Deep dive into software development use cases
- **ResearchFocus**: Focus on research and education sectors

## Examples
\`\`\`tsx
import { UseCasesIndustry } from '@rbee/ui/organisms/UseCases/UseCasesIndustry'

// Simple usage - no props needed
<UseCasesIndustry />
\`\`\`

## Used In
- Use Cases page (/use-cases)

## Related Components
- SectionContainer (layout wrapper)
- IndustryCard (individual industry display)
- UseCasesPrimary (primary use cases)
- UseCasesHero (page hero)

## Accessibility
- **Semantic HTML**: Proper heading hierarchy and card structure
- **Keyboard Navigation**: Filter pills and cards are keyboard accessible
- **ARIA Labels**: Filter navigation has aria-label
- **Screen Readers**: Industry details and compliance badges are properly announced
- **Focus States**: Visible focus indicators on interactive elements
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof UseCasesIndustry>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default UseCasesIndustry as used on /use-cases page.
 * Shows all 6 industries with filter navigation.
 */
export const UseCasesPageDefault: Story = {}

/**
 * Variant focusing on software development industry.
 * Deep dive into developers and tech companies.
 */
export const SoftwareDevFocus: Story = {}

/**
 * Variant focusing on research and education sectors.
 * Highlights academic and research use cases.
 */
export const ResearchFocus: Story = {}
