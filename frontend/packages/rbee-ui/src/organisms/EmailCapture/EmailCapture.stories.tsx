import type { Meta, StoryObj } from '@storybook/react'
import { EmailCapture } from './EmailCapture'

const meta = {
	title: 'Organisms/EmailCapture',
	component: EmailCapture,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The EmailCapture component is a dedicated section for collecting email addresses for the waitlist. It features an engaging headline, visual illustration, email input form with validation, and success state feedback. The section includes decorative bee glyphs and a progress badge showing development status.

## Composition
This organism contains:
- **PulseBadge**: Development status indicator (M0 milestone progress)
- **Headline**: Large, compelling headline
- **Subheadline**: Clear value proposition for joining waitlist
- **Illustration**: Homelab-themed SVG illustration
- **Email Form**: Input field with icon and submit button
- **Trust Microcopy**: Privacy assurance ("No spam. Unsubscribe anytime")
- **Success State**: Confirmation message with checkmark
- **Community Footer**: GitHub repository link and progress info
- **BeeGlyph**: Decorative background elements

## When to Use
- As a dedicated waitlist signup section
- On landing pages to capture early interest
- In marketing campaigns for product launches
- To build an email list before product launch

## Content Requirements
- **Headline**: Compelling, benefit-focused (max 2 lines)
- **Subheadline**: Clear explanation of what users get (2-3 sentences)
- **Illustration**: Visual representation of the product/concept
- **Form**: Email input with clear placeholder
- **Trust Elements**: Privacy assurance and unsubscribe option
- **Success Message**: Clear confirmation of submission

## Variants
- **Default**: Form ready for input
- **Success**: After successful submission (auto-resets after 3s)
- **Loading**: During form submission (future enhancement)
- **Error**: Form validation error state (future enhancement)

## Examples
\`\`\`tsx
import { EmailCapture } from '@rbee/ui/organisms/EmailCapture'

// Simple usage - no props needed
<EmailCapture />
\`\`\`

## Used In
- Home page (/)
- Landing pages
- Marketing campaign pages

## Related Components
- PulseBadge
- BeeGlyph
- Input
- Button

## Accessibility
- **Keyboard Navigation**: Full keyboard support for form
- **Form Labels**: Proper label (sr-only) for email input
- **ARIA Live**: Success message uses aria-live="polite"
- **Focus Management**: Focus returns to form after success state
- **Validation**: HTML5 email validation with required attribute
- **Icons**: All decorative icons marked with aria-hidden
- **Color Contrast**: Meets WCAG AA standards in both themes
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof EmailCapture>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default email capture form ready for input. Use the theme toggle in the toolbar to switch between light and dark modes.',
			},
		},
	},
}

export const MobileView: Story = {
	parameters: {
		viewport: {
			defaultViewport: 'mobile1',
		},
		docs: {
			description: {
				story: 'Mobile view with stacked form layout. Input and button stack vertically on small screens.',
			},
		},
	},
}

export const TabletView: Story = {
	parameters: {
		viewport: {
			defaultViewport: 'tablet',
		},
		docs: {
			description: {
				story: 'Tablet view showing responsive breakpoint behavior.',
			},
		},
	},
}

export const InteractiveDemo: Story = {
	render: () => (
		<div>
			<EmailCapture />
			<div style={{ padding: '2rem', textAlign: 'center', background: 'rgba(0,0,0,0.02)' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Try the Form</h2>
				<ol
					style={{
						listStyle: 'decimal',
						paddingLeft: '2rem',
						textAlign: 'left',
						maxWidth: '600px',
						margin: '0 auto',
						lineHeight: '1.8',
					}}
				>
					<li>Enter an email address in the form above</li>
					<li>Click "Join Waitlist" button</li>
					<li>See the success message appear</li>
					<li>Form automatically resets after 3 seconds</li>
					<li>Check browser console for submission log</li>
				</ol>
				<p style={{ marginTop: '1rem', color: '#666' }}>Form includes HTML5 email validation.</p>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Interactive demo showing the complete form flow: input → submit → success → reset.',
			},
		},
	},
}

export const FormStates: Story = {
	render: () => (
		<div>
			<EmailCapture />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>
					Form States
				</h2>
				<div style={{ maxWidth: '600px', margin: '0 auto' }}>
					<h3 style={{ fontWeight: 'bold', marginTop: '1rem', marginBottom: '0.5rem' }}>Default State</h3>
					<p style={{ marginBottom: '1rem', color: '#666' }}>
						Form is ready for input with email icon and placeholder text.
					</p>

					<h3 style={{ fontWeight: 'bold', marginTop: '1rem', marginBottom: '0.5rem' }}>Success State</h3>
					<p style={{ marginBottom: '1rem', color: '#666' }}>
						After submission, shows green checkmark with confirmation message. Auto-resets after 3 seconds.
					</p>

					<h3 style={{ fontWeight: 'bold', marginTop: '1rem', marginBottom: '0.5rem' }}>Validation</h3>
					<p style={{ marginBottom: '1rem', color: '#666' }}>
						HTML5 email validation prevents submission of invalid emails. Try submitting without "@" symbol.
					</p>

					<h3 style={{ fontWeight: 'bold', marginTop: '1rem', marginBottom: '0.5rem' }}>Trust Elements</h3>
					<p style={{ color: '#666' }}>
						Lock icon and "No spam" message build trust. GitHub link provides transparency.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Overview of all form states and trust-building elements.',
			},
		},
	},
}

export const WithPageContext: Story = {
	render: () => (
		<div>
			<div
				style={{
					padding: '4rem 2rem',
					textAlign: 'center',
					background: 'linear-gradient(to bottom, transparent, rgba(0,0,0,0.02))',
				}}
			>
				<h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '1rem' }}>Previous Section</h1>
				<p>Content above the email capture section.</p>
			</div>
			<EmailCapture />
			<div style={{ padding: '4rem 2rem', textAlign: 'center', background: 'rgba(0,0,0,0.02)' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Next Section</h2>
				<p>Content below the email capture section.</p>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Email capture section in page context showing spacing and visual hierarchy.',
			},
		},
	},
}
