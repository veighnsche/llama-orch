import { solutionTemplateProps as developersSolutionProps } from '@rbee/ui/pages/DevelopersPage'
import { solutionTemplateProps } from '@rbee/ui/pages/HomePage'
import { providersSolutionProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { CommissionStructureCard } from '@rbee/ui/molecules'
import { TrendingUp, Users, Globe, Shield } from 'lucide-react'
import { SolutionTemplate } from './SolutionTemplate'

const meta = {
  title: 'Templates/SolutionTemplate',
  component: SolutionTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof SolutionTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * SolutionTemplate as used on the Home page
 * - Four benefits with topology visualization
 * - Multi-host BeeArchitecture diagram
 * - Shows CUDA, Metal, and CPU orchestration
 */
export const OnHomePage: Story = {
  args: solutionTemplateProps,
}

/**
 * SolutionTemplate as used on the Developers page
 * - Four benefits focused on developer needs
 * - How It Works steps
 * - OpenAI-compatible API code example in aside
 * - Primary and secondary CTAs
 */
export const OnDevelopersPage: Story = {
  args: developersSolutionProps,
}

/**
 * SolutionTemplate as used on the Providers page
 * - Four benefits focused on GPU providers
 * - How It Works steps for marketplace
 * - Earnings card with GPU pricing estimates
 * - Primary and secondary CTAs
 */
export const OnProvidersPage: Story = {
  args: providersSolutionProps,
}

/**
 * SolutionTemplate for Providers Marketplace
 * - Four marketplace feature tiles
 * - Marketplace features list
 * - Commission structure card as aside
 */
export const ProvidersMarketplace: Story = {
  args: {} as any,
  render: () => (
    <SolutionTemplate
        features={[
          {
            icon: <TrendingUp className="size-6" />,
            title: "Dynamic Pricing",
            body: "Set your own rate or use auto-pricing.",
          },
          {
            icon: <Users className="size-6" />,
            title: "Growing Demand",
            body: "Thousands of AI jobs posted monthly.",
          },
          {
            icon: <Globe className="size-6" />,
            title: "Global Reach",
            body: "Your GPUs are discoverable worldwide.",
          },
          {
            icon: <Shield className="size-6" />,
            title: "Fair Commission",
            body: "Keep 85% of every payout.",
          },
        ]}
        steps={[
          {
            title: "Automatic Matching",
            body: "Jobs match your GPUs based on specs and your pricing.",
          },
          {
            title: "Rating System",
            body: "Higher ratings unlock more jobs and better rates.",
          },
          {
            title: "Guaranteed Payments",
            body: "Customers pre-pay. Every completed job is paid.",
          },
          {
            title: "Dispute Resolution",
            body: "A fair process protects both providers and customers.",
          },
        ]}
        aside={
          <div className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
            <CommissionStructureCard
              title="Commission Structure"
              standardCommissionLabel="Standard Commission"
              standardCommissionValue="15%"
              standardCommissionDescription="Covers marketplace operations, payouts, and support."
              youKeepLabel="You Keep"
              youKeepValue="85%"
              youKeepDescription="No hidden fees or surprise deductions."
              exampleItems={[
                { label: "Example job", value: "€100.00" },
                { label: "rbee commission (15%)", value: "−€15.00" },
              ]}
              exampleTotalLabel="Your earnings"
              exampleTotalValue="€85.00"
              exampleBadgeText="Effective take-home: 85%"
            />
          </div>
        }
      />
  ),
}
