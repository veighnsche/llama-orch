import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface SectionContainerProps {
	/** Section title (null to skip rendering) */
	title: string | ReactNode | null
	/** Optional subtitle (deprecated, use description) */
	subtitle?: string | ReactNode
	/** Optional description (alias of subtitle, preferred) */
	description?: string | ReactNode
	/** Small badge/label above title */
	eyebrow?: string | ReactNode
	/** Short lead-in sentence between eyebrow and title */
	kicker?: string | ReactNode
	/** Kicker color variant */
	kickerVariant?: 'default' | 'destructive'
	/** Right-aligned controls near the title (e.g., buttons) */
	actions?: ReactNode
	/** Background variant */
	bgVariant?: 'background' | 'secondary' | 'card' | 'default' | 'muted' | 'subtle' | 'destructive-gradient'
	/** Center the content (deprecated, use align) */
	centered?: boolean
	/** Content alignment */
	align?: 'start' | 'center'
	/** Header layout: stack or split (two-column on md+) */
	layout?: 'stack' | 'split'
	/** Allow full-width background while constraining inner content */
	bleed?: boolean
	/** Vertical padding size */
	paddingY?: 'lg' | 'xl' | '2xl'
	/** Maximum width of content */
	maxWidth?: 'xl' | '2xl' | '3xl' | '4xl' | '5xl' | '6xl' | '7xl'
	/** Section content */
	children: ReactNode
	/** Additional CSS classes */
	className?: string
	/** Optional ID for the heading (for aria-labelledby) */
	headingId?: string
	/** Semantic heading level (1, 2, or 3) */
	headlineLevel?: 1 | 2 | 3
	/** Show a subtle separator under the header block */
	divider?: boolean
}

const bgClasses = {
	background: 'bg-background',
	secondary: 'bg-secondary',
	card: 'bg-card',
	default: 'bg-background',
	muted: 'bg-muted',
	subtle: 'bg-background relative before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-border/60',
	'destructive-gradient': 'bg-gradient-to-b from-background via-destructive/8 to-background border-b border-border',
} as const

const padY = {
	lg: 'py-16',
	xl: 'py-20',
	'2xl': 'py-24',
} as const

const maxWidthClasses = {
	xl: 'max-w-xl',
	'2xl': 'max-w-2xl',
	'3xl': 'max-w-3xl',
	'4xl': 'max-w-4xl',
	'5xl': 'max-w-5xl',
	'6xl': 'max-w-6xl',
	'7xl': 'max-w-7xl',
} as const

/** Semantic heading component that renders h1, h2, or h3 with same visual classes */
function HTag({ as = 2, ...props }: { as?: 1 | 2 | 3 } & React.HTMLAttributes<HTMLHeadingElement> & { id?: string }) {
	const Tag = (as === 1 ? 'h1' : as === 2 ? 'h2' : 'h3') as 'h1' | 'h2' | 'h3'
	return <Tag {...props} />
}

/** Slugify a string for use as an ID */
function slugify(value?: string): string | undefined {
	if (!value) return undefined
	return value
		.toLowerCase()
		.replace(/\s+/g, '-')
		.replace(/[^\w-]/g, '')
		.slice(0, 64)
}

export function SectionContainer({
	title,
	subtitle,
	description,
	eyebrow,
	kicker,
	kickerVariant = 'default',
	actions,
	bgVariant = 'background',
	centered = true,
	align,
	layout = 'stack',
	bleed = false,
	paddingY = '2xl',
	maxWidth = '4xl',
	children,
	className,
	headingId,
	headlineLevel = 2,
	divider = false,
}: SectionContainerProps) {
	// Resolve alignment: prefer align prop, fallback to centered
	const resolvedAlign = align ?? (centered ? 'center' : 'start')

	// Prefer description over subtitle
	const resolvedDescription = description ?? subtitle

	// Generate heading ID if not provided
	const generatedId = headingId ?? (typeof title === 'string' ? slugify(title) : undefined)

	return (
		<section
			className={cn(padY[paddingY], bgClasses[bgVariant], bleed && 'px-0', className)}
			aria-labelledby={title ? generatedId : undefined}
		>
			<div className={cn('container mx-auto', bleed ? 'px-4' : 'px-4')}>
				{title && (
					<div
						className={cn(
							maxWidthClasses[maxWidth],
							'mx-auto mb-14 md:mb-16',
							resolvedAlign === 'center' ? 'text-center' : 'text-left md:text-left',
							layout === 'split' && actions ? 'md:grid md:grid-cols-12 md:items-end md:gap-6' : '',
						)}
					>
						<div className={cn(layout === 'split' && actions ? 'md:col-span-8 space-y-3' : 'space-y-3')}>
							{eyebrow && (
								<div className="text-xs font-medium text-primary uppercase tracking-wide animate-fade-in">
									{eyebrow}
								</div>
							)}

							{kicker && (
								<div
									className={cn(
										'text-sm font-medium animate-fade-in',
										kickerVariant === 'destructive' ? 'text-destructive/80' : 'text-muted-foreground',
									)}
								>
									{kicker}
								</div>
							)}

							<HTag
								as={headlineLevel}
								id={generatedId}
								className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground mb-2 text-balance leading-tight animate-fade-in-up"
							>
								{title}
							</HTag>

							{resolvedDescription && (
								<div
									className={cn(
										'text-lg md:text-xl text-muted-foreground max-w-3xl leading-relaxed animate-fade-in',
										resolvedAlign === 'center' ? 'mx-auto' : 'mx-auto md:mx-0',
									)}
								>
									{resolvedDescription}
								</div>
							)}

							{divider && <div className="h-px bg-border/60 mt-6" />}
						</div>

						{actions && (
							<div
								className={cn(
									'mt-6 md:mt-0',
									layout === 'split'
										? 'md:col-span-4 md:flex md:justify-end'
										: resolvedAlign === 'center'
											? 'flex justify-center'
											: '',
								)}
							>
								{actions}
							</div>
						)}
					</div>
				)}

				{children}
			</div>
		</section>
	)
}
