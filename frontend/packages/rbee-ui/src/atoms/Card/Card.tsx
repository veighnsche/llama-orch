import { cn } from '@rbee/ui/utils'
import * as React from 'react'

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

const Card = React.forwardRef<HTMLDivElement, CardProps>(({ className, ...props }, ref) => {
	return (
		<div
			ref={ref}
			data-slot="card"
			className={cn('bg-card text-card-foreground flex flex-col rounded-xl border/40 shadow-sm', className)}
			{...props}
		/>
	)
})
Card.displayName = 'Card'

const CardHeader = React.forwardRef<HTMLDivElement, CardProps>(({ className, ...props }, ref) => {
	return (
		<div
			ref={ref}
			data-slot="card-header"
			className={cn(
				'@container/card-header grid auto-rows-min grid-rows-[auto_auto] items-start gap-2 has-data-[slot=card-action]:grid-cols-[1fr_auto] p-6 [.border-b]:pb-6',
				className,
			)}
			{...props}
		/>
	)
})
CardHeader.displayName = 'CardHeader'

const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
	({ className, ...props }, ref) => {
		return <h3 ref={ref} data-slot="card-title" className={cn('leading-none font-semibold', className)} {...props} />
	},
)
CardTitle.displayName = 'CardTitle'

const CardDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
	({ className, ...props }, ref) => {
		return <p ref={ref} data-slot="card-description" className={cn('text-muted-foreground text-sm', className)} {...props} />
	},
)
CardDescription.displayName = 'CardDescription'

const CardAction = React.forwardRef<HTMLDivElement, CardProps>(({ className, ...props }, ref) => {
	return (
		<div
			ref={ref}
			data-slot="card-action"
			className={cn('col-start-2 row-span-2 row-start-1 self-start justify-self-end', className)}
			{...props}
		/>
	)
})
CardAction.displayName = 'CardAction'

const CardContent = React.forwardRef<HTMLDivElement, CardProps>(({ className, ...props }, ref) => {
	return <div ref={ref} data-slot="card-content" className={cn('p-6 pt-0', className)} {...props} />
})
CardContent.displayName = 'CardContent'

const CardFooter = React.forwardRef<HTMLDivElement, CardProps>(({ className, ...props }, ref) => {
	return <div ref={ref} data-slot="card-footer" className={cn('flex items-center p-6 pt-0 [.border-t]:pt-6', className)} {...props} />
})
CardFooter.displayName = 'CardFooter'

export { Card, CardHeader, CardFooter, CardTitle, CardAction, CardDescription, CardContent }
