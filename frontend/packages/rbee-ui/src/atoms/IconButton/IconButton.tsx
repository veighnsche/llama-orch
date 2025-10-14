import { Slot } from '@radix-ui/react-slot'
import { cn } from '@rbee/ui/utils'
import * as React from 'react'

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
	children: React.ReactNode
	asChild?: boolean
}

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
	({ className, children, asChild = false, ...props }, ref) => {
		const Comp = asChild ? Slot : 'button'

		return (
			<Comp
				ref={ref}
				className={cn(
					'inline-flex items-center justify-center size-9 rounded-lg',
					'text-muted-foreground hover:text-foreground hover:bg-muted/40',
					'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40',
					'transition-colors',
					'disabled:pointer-events-none disabled:opacity-50',
					className,
				)}
				{...props}
			>
				{children}
			</Comp>
		)
	},
)

IconButton.displayName = 'IconButton'
