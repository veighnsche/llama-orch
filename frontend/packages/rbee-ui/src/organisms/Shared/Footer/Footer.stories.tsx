import type { Meta, StoryObj } from '@storybook/react'
import { Footer } from './Footer'

const meta = {
	title: 'Organisms/Shared/Footer',
	component: Footer,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The Footer component provides comprehensive site navigation, newsletter signup, social media links, and legal information. It's designed with a three-tier structure: utility bar, sitemap grid, and bottom bar with branding.

## Composition
This organism contains:
- **Utility Bar**: Newsletter signup form and quick action buttons
- **Sitemap Grid**: Four columns of organized links (Product, Community, Company, Legal)
- **Bottom Bar**: Brand logo, copyright, and social media icons
- **FooterColumn**: Reusable column component for link groups
- **Social Icons**: GitHub, Discord, Twitter/X, GitHub Discussions

## When to Use
- At the bottom of every page
- Provides secondary navigation and important links
- Newsletter subscription and community engagement

## Content Requirements
- **Newsletter Form**: Email input with subscribe button
- **Link Columns**: Organized by category (4 columns on desktop, stacked on mobile)
- **Social Links**: All active social media channels
- **Legal Links**: Privacy, Terms, License, Security
- **Branding**: Logo and copyright notice

## Variants
- **Default**: Full desktop layout with 4-column grid
- **Mobile**: Stacked single-column layout
- **With Newsletter Success**: State after successful subscription (future enhancement)

## Examples
\`\`\`tsx
import { Footer } from '@rbee/ui/organisms/Footer'

// Simple usage - no props needed
<Footer />
\`\`\`

## Used In
- All pages (layout.tsx)
- Bottom of every page in the application

## Related Components
- FooterColumn
- BrandLogo
- GitHubIcon, DiscordIcon, XTwitterIcon
- Button, Input

## Accessibility
- **Keyboard Navigation**: Tab through all links and form elements
- **Form Labels**: Proper aria-label on email input
- **External Links**: Proper rel="noreferrer" and title attributes
- **Semantic HTML**: Uses <footer> and <nav> elements
- **Focus States**: Visible focus indicators on all interactive elements
- **Screen Readers**: All icons have proper aria-labels
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof Footer>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	parameters: {
		docs: {
			description: {
				story: 'Default footer with all sections: newsletter signup, sitemap grid, and bottom bar with social links. Use the viewport toolbar to test responsive behavior.',
			},
		},
	},
}

export const WithPageContent: Story = {
	render: () => (
		<div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
			<div
				style={{ flex: 1, padding: '2rem', background: 'linear-gradient(to bottom, transparent, rgba(0,0,0,0.02))' }}
			>
				<h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '1rem' }}>Page Content</h1>
				<p style={{ marginBottom: '1rem' }}>
					This demonstrates how the footer appears at the bottom of a page with content.
				</p>
				<p>The footer has a subtle top border and proper spacing from the main content.</p>
			</div>
			<Footer />
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Footer with page content showing typical page layout and spacing.',
			},
		},
	},
}

export const NewsletterFormFocus: Story = {
	render: () => (
		<div>
			<Footer />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', marginTop: '2rem' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Newsletter Form Test</h2>
				<p style={{ marginBottom: '1rem' }}>Test the newsletter signup form:</p>
				<ol style={{ listStyle: 'decimal', paddingLeft: '2rem', lineHeight: '1.8' }}>
					<li>Enter an email address in the input field</li>
					<li>Click Subscribe button</li>
					<li>Check browser console for form submission log</li>
				</ol>
				<p style={{ marginTop: '1rem', color: '#666' }}>Form validation requires a valid email format.</p>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Test the newsletter signup form functionality. Submissions are logged to console.',
			},
		},
	},
}

export const SocialLinksHighlight: Story = {
	render: () => (
		<div>
			<Footer />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', marginTop: '2rem' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Social Links</h2>
				<p style={{ marginBottom: '1rem' }}>The footer includes social media links in two locations:</p>
				<ul style={{ listStyle: 'disc', paddingLeft: '2rem', lineHeight: '1.8' }}>
					<li>
						<strong>Utility Bar</strong>: Quick action buttons (View Docs, Star on GitHub, Join Discord)
					</li>
					<li>
						<strong>Bottom Bar</strong>: Icon-only links (GitHub, Discussions, Twitter/X, Discord)
					</li>
				</ul>
				<p style={{ marginTop: '1rem', color: '#666' }}>Hover over icons to see the color transition effect.</p>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Highlights the social media links and their hover states.',
			},
		},
	},
}

export const LinkOrganization: Story = {
	render: () => (
		<div>
			<Footer />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', marginTop: '2rem' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Link Organization</h2>
				<p style={{ marginBottom: '1rem' }}>The footer organizes links into four categories:</p>
				<div
					style={{
						display: 'grid',
						gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
						gap: '1rem',
						marginTop: '1rem',
					}}
				>
					<div>
						<h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Product</h3>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>Documentation, Quickstart, GitHub, Roadmap, Changelog</p>
					</div>
					<div>
						<h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Community</h3>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>Discord, GitHub Discussions, Twitter, Blog</p>
					</div>
					<div>
						<h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Company</h3>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>About, Pricing, Contact, Support</p>
					</div>
					<div>
						<h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Legal</h3>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>Privacy, Terms, License, Security</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Overview of how links are organized into logical categories.',
			},
		},
	},
}
