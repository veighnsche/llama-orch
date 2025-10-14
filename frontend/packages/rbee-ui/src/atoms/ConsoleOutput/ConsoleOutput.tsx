// Created by: TEAM-AI-ASSISTANT
'use client'

import { cn } from '@rbee/ui/utils'
import { Check, Copy } from 'lucide-react'
import type { ReactNode } from 'react'
import { useState } from 'react'

export interface ConsoleOutputProps {
	/** Console content - can be string or ReactNode for syntax highlighting */
	children: ReactNode
	/** Show terminal window chrome (traffic lights, title bar) */
	showChrome?: boolean
	/** Terminal title */
	title?: string
	/** Terminal variant */
	variant?: 'terminal' | 'code' | 'output'
	/** Additional CSS classes */
	className?: string
	/** Background style */
	background?: 'dark' | 'light' | 'card'
	/** Show copy button */
	copyable?: boolean
	/** Raw text to copy (if different from rendered children) */
	copyText?: string
}

/**
 * ConsoleOutput - A component for displaying terminal/console output with proper monospace font
 *
 * Features:
 * - Uses Geist Mono font for authentic console appearance
 * - Optional terminal window chrome (macOS-style traffic lights)
 * - Multiple variants for different use cases
 * - Dark/light background options
 * - Proper text selection and overflow handling
 *
 * @example
 * ```tsx
 * <ConsoleOutput showChrome title="bash">
 *   $ npm install rbee
 * </ConsoleOutput>
 * ```
 */
export function ConsoleOutput({
	children,
	showChrome = false,
	title,
	variant = 'terminal',
	className,
	background = 'dark',
	copyable = false,
	copyText,
}: ConsoleOutputProps) {
	const [copied, setCopied] = useState(false)

	const bgStyles = {
		dark: 'bg-slate-950 text-slate-50',
		light: 'bg-background text-foreground',
		card: 'bg-card text-card-foreground',
	}

	const handleCopy = async () => {
		const textToCopy = copyText || (typeof children === 'string' ? children : '')
		if (textToCopy) {
			await navigator.clipboard.writeText(textToCopy)
			setCopied(true)
			setTimeout(() => setCopied(false), 2000)
		}
	}

	return (
		<div className={cn('overflow-hidden rounded-lg border border-border shadow-sm', className)}>
			{showChrome && (
				<div className="flex items-center justify-between border-b border-border bg-muted px-4 py-3">
					<div className="flex items-center gap-2">
						<div className="flex gap-2">
							<div className="h-3 w-3 rounded-full bg-terminal-red" />
							<div className="h-3 w-3 rounded-full bg-terminal-amber" />
							<div className="h-3 w-3 rounded-full bg-terminal-green" />
						</div>
						{title && <span className="ml-2 font-mono text-sm text-muted-foreground">{title}</span>}
					</div>
					{copyable && (
						<button
							onClick={handleCopy}
							className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors rounded-md hover:bg-accent/50"
							aria-label="Copy to clipboard"
						>
							{copied ? (
								<>
									<Check className="h-3.5 w-3.5" />
									<span>Copied</span>
								</>
							) : (
								<>
									<Copy className="h-3.5 w-3.5" />
									<span>Copy</span>
								</>
							)}
						</button>
					)}
				</div>
			)}
			<div className={cn('overflow-x-auto p-4 text-sm leading-relaxed font-mono', bgStyles[background])}>
				{children}
			</div>
		</div>
	)
}
