import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface CTAOptionCardProps {
	icon: ReactNode
	title: string
	body: string
	action: ReactNode
	tone?: 'primary' | 'outline'
	note?: string
	className?: string
}

export function CTAOptionCard({ icon, title, body, action, tone = 'outline', note, className }: CTAOptionCardProps) {
	const titleId = `cta-option-${title.toLowerCase().replace(/\s+/g, '-')}`

	return (
		<article
			role="group"
			aria-labelledby={titleId}
			className={cn(
				'h-full flex flex-col rounded-2xl border border-border bg-card/60 p-6',
				'hover:border-primary/30 transition-colors',
				tone === 'primary' && 'border-primary/40 bg-primary/5',
				className,
			)}
		>
			{/* Icon Chip */}
			<div className="mb-4 flex justify-center">
				<div className="rounded-xl bg-primary/10 text-primary p-3" aria-hidden="true">
					{icon}
				</div>
			</div>

			{/* Title */}
			<h3 id={titleId} className="mb-3 text-center text-xl font-semibold text-foreground">
				{title}
			</h3>

			{/* Body */}
			<p className="mb-4 text-center text-sm text-muted-foreground">{body}</p>

			{/* Action (pushed to bottom) */}
			<div className="mt-auto">
				{action}
				{note && <p className="mt-2 text-center text-[11px] text-muted-foreground">{note}</p>}
			</div>
		</article>
	)
}
