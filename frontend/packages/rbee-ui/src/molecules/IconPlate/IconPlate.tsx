import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'

export interface IconPlateProps {
	/** Lucide icon component to display */
	icon: LucideIcon
	/** Size variant */
	size?: 'sm' | 'md' | 'lg' | 'xl'
	/** Color tone - supports both tone names and chart colors */
	tone?: 'primary' | 'muted' | 'success' | 'warning' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'
	/** Shape variant */
	shape?: 'square' | 'rounded' | 'circle'
	/** Additional CSS classes */
	className?: string
}

/**
 * IconPlate molecule - reusable icon container
 * Consolidates IconBox and 15+ instances of icon wrapper patterns
 * Used across features, stats, cards, and list items
 * 
 * Only accepts LucideIcon components for consistency and type safety.
 */
export function IconPlate({ icon: Icon, size = 'md', tone = 'primary', shape = 'square', className }: IconPlateProps) {
	const sizeClasses = {
		sm: 'h-8 w-8',
		md: 'h-10 w-10',
		lg: 'h-12 w-12',
		xl: 'h-14 w-14',
	}

	const iconSizeClasses = {
		sm: '[&>svg]:h-4 [&>svg]:w-4',
		md: '[&>svg]:h-5 [&>svg]:w-5',
		lg: '[&>svg]:h-6 [&>svg]:w-6',
		xl: '[&>svg]:h-7 [&>svg]:w-7',
	}

	const toneClasses = {
		primary: 'bg-primary/10 text-primary',
		muted: 'bg-muted text-muted-foreground',
		success: 'bg-emerald-500/10 text-emerald-500',
		warning: 'bg-amber-500/10 text-amber-500',
		'chart-1': 'bg-chart-1/10 text-chart-1',
		'chart-2': 'bg-chart-2/10 text-chart-2',
		'chart-3': 'bg-chart-3/10 text-chart-3',
		'chart-4': 'bg-chart-4/10 text-chart-4',
		'chart-5': 'bg-chart-5/10 text-chart-5',
	}

	const shapeClasses = {
		square: 'rounded-none',
		rounded: 'rounded-lg',
		circle: 'rounded-full',
	}

	return (
		<div
			className={cn(
				'flex items-center justify-center',
				sizeClasses[size],
				iconSizeClasses[size],
				toneClasses[tone],
				shapeClasses[shape],
				className,
			)}
		>
			<Icon aria-hidden="true" />
		</div>
	)
}
