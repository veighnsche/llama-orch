'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { Input } from '@rbee/ui/atoms/Input'
import { PulseBadge } from '@rbee/ui/molecules'
import { BeeGlyph } from '@rbee/ui/patterns/BeeGlyph'
import { CheckCircle2, GitBranch, Lock, Mail } from 'lucide-react'
import Image from 'next/image'
import type React from 'react'
import { useState } from 'react'

export function EmailCapture() {
	const [email, setEmail] = useState('')
	const [submitted, setSubmitted] = useState(false)

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault()
		// TODO: Wire up to actual email service
		console.log('Email submitted:', email)
		setSubmitted(true)
		setTimeout(() => {
			setSubmitted(false)
			setEmail('')
		}, 3000)
	}

	return (
		<section className="relative isolate py-28 bg-background">
			{/* Decorative bee glyphs */}
			<BeeGlyph className="absolute top-16 left-[8%] opacity-5 pointer-events-none" />
			<BeeGlyph className="absolute bottom-20 right-[10%] opacity-5 pointer-events-none" />

			<div className="relative max-w-3xl mx-auto px-6 text-center">
				{/* Status badge */}
				<div
					className="mb-4 inline-flex animate-in fade-in slide-in-from-bottom-2 duration-500"
					style={{ animationDelay: '100ms' }}
				>
					<PulseBadge text="In Development · M0 · 68%" />
				</div>

				{/* Headline */}
				<h2
					className="text-5xl md:text-6xl font-bold tracking-tight text-foreground mb-5 animate-in fade-in slide-in-from-bottom-2 duration-500"
					style={{ animationDelay: '300ms' }}
				>
					Get Updates. Own Your AI.
				</h2>

				{/* Subhead */}
				<p
					className="text-lg md:text-xl text-muted-foreground mb-8 leading-relaxed max-w-2xl mx-auto animate-in fade-in slide-in-from-bottom-2 duration-500"
					style={{ animationDelay: '450ms' }}
				>
					Join the rbee waitlist to get early access, build notes, and launch perks for running AI on your own hardware.
				</p>

				{/* Supportive visual - homelab illustration */}
				<Image
					src="/illustrations/homelab-bee.svg"
					width={960}
					height={140}
					priority
					className="mx-auto mb-6 opacity-90"
					alt="isometric homelab rack connected across small PCs; friendly bee icon hovering above; warm midnight palette; minimal linework; evokes 'run AI on your own hardware'"
				/>

				{/* Form or success state */}
				{!submitted ? (
					<form onSubmit={handleSubmit} className="mx-auto max-w-xl">
						<div className="flex flex-col sm:flex-row items-stretch gap-3">
							<div className="relative flex-1">
								<label htmlFor="waitlist-email" className="sr-only">
									Email address
								</label>
								<Mail
									className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground/80"
									aria-hidden="true"
									focusable="false"
								/>
								<Input
									id="waitlist-email"
									type="email"
									placeholder="you@company.com"
									value={email}
									onChange={(e) => setEmail(e.target.value)}
									required
									className="h-12 pl-10 bg-card/80 border-border/70 text-foreground placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-primary/40 transition-shadow data-[invalid=true]:border-destructive/60 data-[invalid=true]:bg-destructive/5"
								/>
							</div>
							<Button
								type="submit"
								className="h-12 px-7 bg-primary text-primary-foreground font-semibold rounded-xl shadow-sm hover:translate-y-[-1px] hover:shadow-md transition-transform"
							>
								Join Waitlist
							</Button>
						</div>

						{/* Trust microcopy */}
						<div className="mt-3 text-sm text-muted-foreground flex items-center justify-center gap-2">
							<Lock className="w-3.5 h-3.5 text-muted-foreground/70" aria-hidden="true" focusable="false" />
							<span>No spam. Unsubscribe anytime.</span>
						</div>
					</form>
				) : (
					<div
						className="inline-flex items-center gap-2 text-chart-3 text-base md:text-lg font-medium bg-card/60 border border-border/60 rounded-xl px-4 py-3 shadow-xs"
						role="status"
						aria-live="polite"
					>
						<CheckCircle2 className="w-5 h-5" aria-hidden="true" focusable="false" />
						<span>Thanks! You're on the list — we'll keep you posted.</span>
					</div>
				)}

				{/* Community footer band */}
				<div className="mt-14 pt-8">
					{/* Gradient divider */}
					<div className="h-px w-full mx-auto bg-gradient-to-r from-transparent via-border to-transparent" />

					<p className="text-sm text-muted-foreground mt-6">Follow progress & contribute on GitHub</p>

					<a
						href="https://github.com/veighnsche/llama-orch"
						target="_blank"
						rel="noopener noreferrer"
						className="inline-flex items-center gap-2 mt-3 text-primary font-medium hover:text-primary/90 transition-colors"
					>
						<GitBranch className="w-5 h-5" aria-hidden="true" focusable="false" />
						<span>View Repository</span>
					</a>

					<p className="text-xs text-muted-foreground/80 mt-2">Weekly dev notes. Roadmap issues tagged M0–M2.</p>
				</div>
			</div>
		</section>
	)
}
