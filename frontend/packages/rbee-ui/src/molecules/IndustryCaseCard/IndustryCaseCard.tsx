import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'
import Link from 'next/link'

export interface IndustryCaseCardProps {
	/** Lucide icon component (e.g., Building2, Heart, Scale, Shield) */
	icon: LucideIcon
	/** Industry name (e.g., "Financial Services") */
	industry: string
	/** Segment description (e.g., "Banks, Insurance, FinTech") */
	segments: string
	/** Brief summary of the use case */
	summary: string
	/** List of challenges */
	challenges: string[]
	/** List of solutions */
	solutions: string[]
	/** Optional compliance badges */
	badges?: string[]
	/** Optional link to industry page */
	href?: string
	/** Additional CSS classes */
	className?: string
}

/**
 * IndustryCaseCard molecule for regulated industry use cases
 * with Challenge → Solution contrast and compliance badges
 */
export function IndustryCaseCard({
	icon,
	industry,
	segments,
	summary,
	challenges,
	solutions,
	badges,
	href,
	className,
}: IndustryCaseCardProps) {
	const industryId = `industry-${industry.toLowerCase().replace(/\s+/g, '-')}`

	return (
		<div
			className={cn(
				'flex h-full flex-col rounded-2xl border border-border bg-card/60 p-6 md:p-7',
				'transition-shadow hover:shadow-lg',
				className,
			)}
			role="group"
			aria-labelledby={industryId}
		>
			{/* Header */}
			<div className="mb-4 flex items-center gap-3">
				<IconPlate icon={icon} size="lg" tone="primary" className="shrink-0" />
				<div className="flex-1">
					<h3 id={industryId} className="text-xl font-bold text-foreground">
						{industry}
					</h3>
					<p className="text-sm text-muted-foreground">{segments}</p>
				</div>
			</div>

			{/* Compliance badges */}
			{badges && badges.length > 0 && (
				<div className="mb-4 flex flex-wrap gap-2">
					{badges.map((badge) => (
						<span
							key={badge}
							className="rounded-full border border-border bg-background px-2.5 py-0.5 text-xs text-muted-foreground"
						>
							{badge}
						</span>
					))}
				</div>
			)}

			{/* Summary */}
			<p className="mb-4 text-sm text-muted-foreground">{summary}</p>

			{/* Challenge panel */}
			<div className="mb-4 rounded-xl border border-border bg-background p-4">
				<div className="mb-2 font-semibold text-foreground">Challenge</div>
				<ul className="space-y-1.5 text-sm text-muted-foreground" role="list">
					{challenges.map((challenge) => (
						<li key={challenge} className="flex gap-2">
							<span className="sr-only">Challenge:</span>
							<span className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true">
								•
							</span>
							<span>{challenge}</span>
						</li>
					))}
				</ul>
			</div>

			{/* Solution panel */}
			<div className="rounded-xl border border-chart-3/50 bg-chart-3/10 p-4">
				<div className="mb-2 font-semibold text-chart-3">Solution with rbee</div>
				<ul className="space-y-1.5 text-sm text-foreground/85" role="list">
					{solutions.map((solution) => (
						<li key={solution} className="flex gap-2">
							<span className="sr-only">Solution:</span>
							<span className="mt-0.5 h-4 w-4 shrink-0 text-chart-3" aria-hidden="true">
								✓
							</span>
							<span>{solution}</span>
						</li>
					))}
				</ul>
			</div>

			{/* Optional footer link */}
			{href && (
				<div className="mt-auto pt-4">
					<Link
						href={href}
						className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
						aria-label={`Learn more about ${industry}`}
					>
						Learn more →
					</Link>
				</div>
			)}
		</div>
	)
}
