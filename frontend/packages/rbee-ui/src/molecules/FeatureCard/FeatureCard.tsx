import { CheckItem } from '@rbee/ui/atoms/CheckItem'
import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

export interface FeatureCardProps {
	/** Lucide icon component */
	icon: LucideIcon | ReactNode
	/** Card title */
	title: string
	/** Card intro/description */
	intro: string
	/** Bullet points list */
	bullets: string[]
	/** Optional footer link */
	href?: string
	/** Icon background color (Tailwind class) */
	iconColor?: string
	/** Enable hover effect */
	hover?: boolean
	/** Card size variant */
	size?: 'sm' | 'md' | 'lg'
	/** Additional CSS classes */
	className?: string
	/** Optional footer content (e.g., micro-metrics) */
	children?: React.ReactNode
	/** Optional mini-stat for header */
	stat?: { label: string; value: string }
	/** Optional ID for anchor linking */
	id?: string
	// Legacy props for backward compatibility
	/** @deprecated Use intro instead */
	description?: string
}

export function FeatureCard({
	icon,
	title,
	intro,
	bullets,
	href,
	iconColor = 'primary',
	hover = false,
	size = 'md',
	className,
	children,
	stat,
	id,
	description, // Legacy support
}: FeatureCardProps) {
	const IconComponent = typeof icon === 'function' ? (icon as LucideIcon) : null
	const displayIntro = intro || description || ''
	const sizeClasses = {
		sm: 'p-4 space-y-2',
		md: 'p-6 space-y-3',
		lg: 'p-8 space-y-4',
	}

	const iconSizeClasses = {
		sm: 'h-8 w-8',
		md: 'h-10 w-10',
		lg: 'h-12 w-12',
	}

	const iconInnerSizeClasses = {
		sm: 'h-4 w-4',
		md: 'h-5 w-5',
		lg: 'h-6 w-6',
	}

	const titleSizeClasses = {
		sm: 'text-sm',
		md: 'text-base',
		lg: 'text-lg',
	}

	const descriptionSizeClasses = {
		sm: 'text-xs',
		md: 'text-sm',
		lg: 'text-base',
	}

	const colorClasses = {
		primary: { bg: 'bg-primary/10', text: 'text-primary' },
		'chart-1': { bg: 'bg-chart-1/10', text: 'text-chart-1' },
		'chart-2': { bg: 'bg-chart-2/10', text: 'text-chart-2' },
		'chart-3': { bg: 'bg-chart-3/10', text: 'text-chart-3' },
		'chart-4': { bg: 'bg-chart-4/10', text: 'text-chart-4' },
		'chart-5': { bg: 'bg-chart-5/10', text: 'text-chart-5' },
	}

	const colors = colorClasses[iconColor as keyof typeof colorClasses] || colorClasses.primary

	// Map iconColor to border accent color
	const borderAccentClasses = {
		primary: 'border-t-primary',
		'chart-1': 'border-t-chart-1',
		'chart-2': 'border-t-chart-2',
		'chart-3': 'border-t-chart-3',
		'chart-4': 'border-t-chart-4',
		'chart-5': 'border-t-chart-5',
	}
	const borderAccent = borderAccentClasses[iconColor as keyof typeof borderAccentClasses] || borderAccentClasses.primary

	const titleId = id ? `${id}-title` : undefined

	return (
		<div
			id={id}
			role="group"
			aria-labelledby={titleId}
			className={cn(
				'h-full flex flex-col rounded-2xl border border-border bg-card/60 p-6 md:p-8',
				hover && 'transition-all hover:border-primary/50 hover:bg-card/80',
				className,
			)}
		>
			{/* Header Row */}
			<div className="mb-4 flex items-center gap-3">
				<div className={cn('rounded-xl flex items-center justify-center shrink-0 p-3', colors.bg)} aria-hidden="true">
					{IconComponent ? (
						<IconComponent aria-hidden="true" focusable="false" className={cn('h-6 w-6', colors.text)} />
					) : (
						icon as ReactNode
					)}
				</div>
				<h3 id={titleId} className="text-xl font-semibold text-foreground">
					{title}
				</h3>
			</div>

			{/* Intro Paragraph */}
			<p className="text-sm text-muted-foreground mb-3">{displayIntro}</p>

			{/* Bullet List */}
			{bullets && bullets.length > 0 && (
				<ul className="mt-3 space-y-2">
					{bullets.map((bullet, index) => (
						<CheckItem key={index}>{bullet}</CheckItem>
					))}
				</ul>
			)}

			{/* Optional Footer Link */}
			{href && (
				<a
					href={href}
					className="mt-4 inline-flex items-center gap-1 text-sm text-primary hover:underline focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none rounded"
				>
					Learn more →
				</a>
			)}

			{/* Optional Children */}
			{children && <div className="mt-auto pt-2">{children}</div>}
		</div>
	)
}
