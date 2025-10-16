import type { Meta, StoryObj } from '@storybook/react'
import { IndustriesHero } from '@rbee/ui/icons'
import { Banknote, Factory, GraduationCap, Heart, Landmark, Scale } from 'lucide-react'
import { UseCasesIndustryTemplate } from './UseCasesIndustryTemplate'

const meta = {
  title: 'Templates/UseCasesIndustryTemplate',
  component: UseCasesIndustryTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof UseCasesIndustryTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnUseCasesPage: Story = {
  args: {
    eyebrow: 'Regulated sectors · Private-by-design',
    heroImage: (
      <IndustriesHero
        size="100%"
        className="w-full h-auto"
        aria-label="Visual representation of various industry sectors including healthcare, government, finance, education, manufacturing, and research with AI integration"
      />
    ),
    heroImageAriaLabel:
      'Visual representation of various industry sectors including healthcare, government, finance, education, manufacturing, and research with AI integration',
    filters: [
      { label: 'All', anchor: '#architecture' },
      { label: 'Finance', anchor: '#finance' },
      { label: 'Healthcare', anchor: '#healthcare' },
      { label: 'Legal', anchor: '#legal' },
      { label: 'Public Sector', anchor: '#government' },
      { label: 'Education', anchor: '#education' },
      { label: 'Manufacturing', anchor: '#manufacturing' },
    ],
    industries: [
      {
        title: 'Financial Services',
        icon: Banknote,
        color: 'primary',
        badge: 'GDPR',
        copy: 'GDPR-ready with audit trails and data residency. Run AI code review and risk analysis without sending financial data to external APIs.',
        anchor: 'finance',
      },
      {
        title: 'Healthcare',
        icon: Heart,
        color: 'chart-2',
        badge: 'HIPAA',
        copy: 'HIPAA-compliant by design. Patient data stays on your network while AI assists with medical coding, documentation, and research.',
        anchor: 'healthcare',
      },
      {
        title: 'Legal',
        icon: Scale,
        color: 'chart-3',
        copy: 'Preserve attorney–client privilege. Perform document/contract analysis and legal research with AI—without client data leaving your environment.',
        anchor: 'legal',
      },
      {
        title: 'Government',
        icon: Landmark,
        color: 'chart-4',
        badge: 'ITAR',
        copy: 'Sovereign, no foreign cloud dependency. Full auditability and policy-enforced routing to meet government security standards.',
        anchor: 'government',
      },
      {
        title: 'Education',
        icon: GraduationCap,
        color: 'chart-2',
        badge: 'FERPA',
        copy: 'Protect student information (FERPA-friendly). AI tutoring, grading assistance, and research tools with zero third-party data sharing.',
        anchor: 'education',
      },
      {
        title: 'Manufacturing',
        icon: Factory,
        color: 'primary',
        copy: 'Safeguard IP and trade secrets. AI-assisted CAD review, quality control, and process optimization—no exposure of proprietary designs.',
        anchor: 'manufacturing',
      },
    ],
  },
}
