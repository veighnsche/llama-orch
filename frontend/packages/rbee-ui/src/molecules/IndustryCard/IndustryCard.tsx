import { Badge } from '@rbee/ui/atoms/Badge'
import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'

export interface IndustryCardProps {
	title: string
	copy: string
	icon: LucideIcon
	color: 'primary' | 'chart-2' | 'chart-3' | 'chart-4'
	badge?: string
	anchor?: string
	className?: string
	style?: React.CSSProperties
}

export function IndustryCard({ title, copy, icon, color, badge, anchor, className, style }: IndustryCardProps) {
	return (
		<article
			id={anchor}
			role="article"
			aria-labelledby={anchor ? `${anchor}-title` : undefined}
			tabIndex={-1}
			className={cn(
				'bg-card border border-border/80 rounded-xl p-6 md:p-7 shadow-sm hover:shadow-md transition-all scroll-mt-28',
				'hover:-translate-y-0.5',
				'focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary/40',
				'animate-in fade-in-50 slide-in-from-bottom-4',
				className,
			)}
			style={style}
		>
			{/* Top row: Icon + Badge */}
			<div className="flex items-start justify-between mb-4">
				<IconPlate icon={icon} size="lg" tone={color} shape="rounded" />
				{badge && (
					<Badge variant="outline" className="text-xs font-semibold">
						{badge}
					</Badge>
				)}
			</div>

			{/* Title */}
			<h3
				id={anchor ? `${anchor}-title` : undefined}
				className="text-xl md:text-2xl font-semibold tracking-tight text-foreground mb-3"
			>
				{title}
			</h3>

			{/* Body copy */}
			<p className="text-sm leading-relaxed text-muted-foreground">{copy}</p>
		</article>
	)
}
