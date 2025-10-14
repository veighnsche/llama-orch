import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface StepCardProps {
	/** Step number (1-4) */
	index: number
	/** Icon element (e.g., Shield, Server, CheckCircle, Rocket) */
	icon: ReactNode
	/** Step title */
	title: string
	/** Introduction paragraph */
	intro: string
	/** List of deliverables/items */
	items: string[]
	/** Optional footnote */
	footnote?: string
	/** Whether this is the last step (hides connector line) */
	isLast?: boolean
	/** Additional CSS classes */
	className?: string
}

/**
 * StepCard molecule for deployment process steps
 * with numbered badge, icon, intro, and deliverables list
 */
export function StepCard({ index, icon, title, intro, items, footnote, isLast, className }: StepCardProps) {
	return (
		<li
			className={cn('relative flex gap-6', className)}
			role="group"
			aria-label={`Step ${index}: ${title}`}
			style={{ ['--i' as any]: index - 1 }}
		>
			{/* Badge with connector line */}
			<div className="relative flex shrink-0 flex-col items-center">
				<div
					className="grid h-10 w-10 place-content-center rounded-full bg-primary text-lg font-bold text-primary-foreground"
					aria-hidden="true"
				>
					{index}
				</div>
				{/* Connector line (hidden on last step) */}
				{!isLast && <div className="absolute left-5 top-12 h-[calc(100%+2rem)] w-px bg-border" aria-hidden="true" />}
			</div>

			{/* Card content */}
			<div className="flex-1 rounded-2xl border border-border bg-card/50 p-6 md:p-7">
				{/* Header */}
				<div className="mb-4 flex items-center gap-3">
					<div className="text-primary" aria-hidden="true">
						{icon}
					</div>
					<h3 className="text-2xl font-semibold text-foreground">{title}</h3>
				</div>

				{/* Intro */}
				<p className="mb-4 text-sm leading-relaxed text-foreground/85">{intro}</p>

				{/* Deliverables list */}
				<ul className="space-y-1.5 text-sm text-muted-foreground" role="list">
					{items.map((item) => (
						<li key={item} className="flex gap-2" aria-label={`Deliverable: ${item}`}>
							<span className="mt-0.5 h-4 w-4 shrink-0 text-chart-3" aria-hidden="true">
								✓
							</span>
							<span>{item}</span>
						</li>
					))}
				</ul>

				{/* Optional footnote */}
				{footnote && <p className="mt-4 text-xs text-muted-foreground">{footnote}</p>}
			</div>
		</li>
	)
}
