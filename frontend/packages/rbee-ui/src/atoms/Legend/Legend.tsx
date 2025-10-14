import { cn } from '@rbee/ui/utils'
import { Check, X } from 'lucide-react'
import type { CSSProperties } from 'react'

export interface LegendProps {
	className?: string
	style?: CSSProperties
}

export function Legend({ className, style }: LegendProps) {
	return (
		<div
			className={cn('flex flex-wrap items-center justify-center gap-3 text-xs text-muted-foreground', className)}
			style={style}
		>
			<span className="flex items-center gap-1.5">
				<Check className="h-3.5 w-3.5 text-chart-3" aria-hidden="true" />
				<span>Included</span>
			</span>
			<span className="text-muted-foreground/40">·</span>
			<span className="flex items-center gap-1.5">
				<X className="h-3.5 w-3.5 text-destructive" aria-hidden="true" />
				<span>Not available</span>
			</span>
			<span className="text-muted-foreground/40">·</span>
			<span className="flex items-center gap-1.5">
				<span
					className="inline-flex rounded-full border border-border/60 bg-background px-1.5 py-0.5 text-[10px] text-foreground/80"
					aria-hidden="true"
				>
					Partial
				</span>
				<span>Available with constraints (region, SKU, or config)</span>
			</span>
		</div>
	)
}
