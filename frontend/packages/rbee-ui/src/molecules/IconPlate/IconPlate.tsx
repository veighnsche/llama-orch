import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface IconPlateProps {
	/** Icon element to display */
	icon: ReactNode
	/** Size variant */
	size?: 'sm' | 'md' | 'lg'
	/** Color tone */
	tone?: 'primary' | 'muted' | 'success' | 'warning'
	/** Shape variant */
	shape?: 'square' | 'circle'
	/** Additional CSS classes */
	className?: string
}

/**
 * IconPlate molecule - reusable icon container
 * Consolidates 15+ instances of icon wrapper patterns
 * Used across features, stats, cards, and list items
 */
export function IconPlate({ icon, size = 'md', tone = 'primary', shape = 'square', className }: IconPlateProps) {
	const sizeClasses = {
		sm: 'h-8 w-8',
		md: 'h-9 w-9',
		lg: 'h-12 w-12',
	}

	const iconSizeClasses = {
		sm: '[&>svg]:h-3.5 [&>svg]:w-3.5',
		md: '[&>svg]:h-4 [&>svg]:w-4',
		lg: '[&>svg]:h-5 [&>svg]:w-5',
	}

	const toneClasses = {
		primary: 'bg-primary/10 text-primary',
		muted: 'bg-muted text-muted-foreground',
		success: 'bg-emerald-500/10 text-emerald-500',
		warning: 'bg-amber-500/10 text-amber-500',
	}

	const shapeClasses = {
		square: 'rounded-lg',
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
			{icon}
		</div>
	)
}
