import type { Meta, StoryObj } from '@storybook/react'
import { FooterColumn } from './FooterColumn'

const meta: Meta<typeof FooterColumn> = {
  title: 'Molecules/FooterColumn',
  component: FooterColumn,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
## Overview
FooterColumn is a structured navigation column molecule for footer sections. It displays a title and a list of links with proper accessibility attributes.

## Composition
This molecule is composed of:
- **Heading**: Section title with semantic h3
- **Link list**: Unordered list of navigation links
- **Link items**: Internal (Next.js Link) or external (anchor) links
- **Accessibility**: aria-labelledby for list association

## When to Use
- Footer navigation (site map columns)
- Sitemap sections (organized link groups)
- Footer menus (categorized links)
- Resource lists (documentation, support)

## Variants
- **Internal links**: Use Next.js Link for same-site navigation
- **External links**: Use anchor tags with target="_blank"

## Used In Commercial Site
Used in:
- Footer (main footer navigation)
- SiteMap (footer sitemap section)
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    title: {
      control: 'text',
      description: 'Column title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    links: {
      control: 'object',
      description: 'Array of footer links',
      table: {
        type: { summary: 'FooterLink[]' },
        category: 'Content',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof FooterColumn>

export const Default: Story = {
  args: {
    title: 'Product',
    links: [
      { href: '/features', text: 'Features' },
      { href: '/pricing', text: 'Pricing' },
      { href: '/docs', text: 'Documentation' },
      { href: '/changelog', text: 'Changelog' },
    ],
  },
}

export const WithTitle: Story = {
  render: () => (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-8 max-w-4xl">
      <FooterColumn
        title="Product"
        links={[
          { href: '/features', text: 'Features' },
          { href: '/pricing', text: 'Pricing' },
          { href: '/enterprise', text: 'Enterprise' },
          { href: '/changelog', text: 'Changelog' },
        ]}
      />
      <FooterColumn
        title="Developers"
        links={[
          { href: '/docs', text: 'Documentation' },
          { href: '/api', text: 'API Reference' },
          { href: '/sdk', text: 'SDK' },
          { href: '/examples', text: 'Examples' },
        ]}
      />
      <FooterColumn
        title="Resources"
        links={[
          { href: '/blog', text: 'Blog' },
          { href: '/guides', text: 'Guides' },
          { href: '/support', text: 'Support' },
          { href: '/status', text: 'Status' },
        ]}
      />
      <FooterColumn
        title="Company"
        links={[
          { href: '/about', text: 'About' },
          { href: '/careers', text: 'Careers' },
          { href: '/contact', text: 'Contact' },
          { href: '/legal', text: 'Legal' },
        ]}
      />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Multiple footer columns showing typical footer navigation structure.',
      },
    },
  },
}

export const WithIcons: Story = {
  render: () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-3xl">
      <FooterColumn
        title="Social"
        links={[
          { href: 'https://github.com/rbee', text: 'GitHub', external: true },
          { href: 'https://twitter.com/rbee', text: 'Twitter', external: true },
          { href: 'https://discord.gg/rbee', text: 'Discord', external: true },
          { href: 'https://linkedin.com/company/rbee', text: 'LinkedIn', external: true },
        ]}
      />
      <FooterColumn
        title="Legal"
        links={[
          { href: '/privacy', text: 'Privacy Policy' },
          { href: '/terms', text: 'Terms of Service' },
          { href: '/gdpr', text: 'GDPR Compliance' },
          { href: '/cookies', text: 'Cookie Policy' },
        ]}
      />
      <FooterColumn
        title="Support"
        links={[
          { href: '/help', text: 'Help Center' },
          { href: '/contact', text: 'Contact Us' },
          { href: 'https://status.rbee.nl', text: 'System Status', external: true },
          { href: '/feedback', text: 'Feedback' },
        ]}
      />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Footer columns with mix of internal and external links.',
      },
    },
  },
}

export const InFooterContext: Story = {
  render: () => (
    <div className="w-full">
      <div className="mb-4 text-sm text-muted-foreground">Example: FooterColumn in Footer component</div>
      <div className="w-full bg-card rounded-lg border p-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          <div>
            <div className="font-bold text-lg mb-4">rbee</div>
            <p className="text-sm text-muted-foreground">
              Private LLM hosting in the Netherlands. GDPR-compliant AI infrastructure.
            </p>
          </div>
          <FooterColumn
            title="Product"
            links={[
              { href: '/features', text: 'Features' },
              { href: '/pricing', text: 'Pricing' },
              { href: '/enterprise', text: 'Enterprise' },
            ]}
          />
          <FooterColumn
            title="Developers"
            links={[
              { href: '/docs', text: 'Documentation' },
              { href: '/api', text: 'API Reference' },
              { href: '/sdk', text: 'SDK' },
            ]}
          />
          <FooterColumn
            title="Company"
            links={[
              { href: '/about', text: 'About' },
              { href: '/contact', text: 'Contact' },
              { href: '/legal', text: 'Legal' },
            ]}
          />
        </div>
        <div className="pt-8 border-t border-border text-center text-sm text-muted-foreground">
          © 2025 rbee. All rights reserved.
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'FooterColumn as used in the Footer component with brand section and copyright.',
      },
    },
  },
}
