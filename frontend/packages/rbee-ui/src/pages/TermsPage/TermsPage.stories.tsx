import type { Meta, StoryObj } from '@storybook/react'
import TermsPage from './TermsPage'

const meta = {
  title: 'Pages/TermsPage',
  component: TermsPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Terms of Service page with comprehensive legal content in FAQ format. Uses HeroTemplate, FAQTemplate, and CTATemplate.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof TermsPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Terms of Service page
 *
 * **Structure:**
 * 1. HeroTemplate - Title, last updated, legal icon
 * 2. FAQTemplate - 17 terms sections as searchable Q&A across 8 categories
 * 3. CTATemplate - Contact legal team
 *
 * **Features:**
 * - Searchable terms content
 * - 8 category filters (Agreement, License, Use Policy, IP, Liability, Termination, Dispute, General)
 * - Expand/collapse all functionality
 * - Support card with quick links
 * - JSON-LD schema for SEO
 * - Mobile-responsive
 * - Light/dark mode support
 *
 * **Legal Coverage:**
 * - Acceptance of terms
 * - GPL-3.0 license explanation
 * - Acceptable use policy
 * - Intellectual property
 * - Warranties & liability
 * - Termination conditions
 * - Dispute resolution
 * - Contact information
 *
 * **Status:** ✅ Complete - Pending legal review
 */
export const Default: Story = {}

/**
 * Mobile view
 *
 * Tests responsive behavior:
 * - Support card hidden on mobile
 * - Accordion stacks properly
 * - Search and filters work on small screens
 */
export const Mobile: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
  },
}

/**
 * Tablet view
 *
 * Tests medium screen layout:
 * - Support card still hidden
 * - Content width adjusts
 * - Filters wrap properly
 */
export const Tablet: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'tablet',
    },
  },
}

/**
 * Dark mode
 *
 * Tests dark theme:
 * - Proper contrast
 * - Readable text
 * - Card backgrounds
 */
export const DarkMode: Story = {
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
}
