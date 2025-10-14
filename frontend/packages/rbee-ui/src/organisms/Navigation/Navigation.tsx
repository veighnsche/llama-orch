'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { GitHubIcon } from '@rbee/ui/atoms/GitHubIcon'
import { IconButton } from '@rbee/ui/atoms/IconButton'
import { Separator } from '@rbee/ui/atoms/Separator'
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from '@rbee/ui/atoms/Sheet'
import { NavLink } from '@rbee/ui/molecules'
import { BrandLogo } from '@rbee/ui/molecules/BrandLogo'
import { ThemeToggle } from '@rbee/ui/molecules/ThemeToggle'
import { Menu, X } from 'lucide-react'
import { useState } from 'react'

export function Navigation() {
	const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

	return (
		<>
			{/* Skip to content link */}
			<a
				href="#main"
				className="ui:sr-only ui:focus:not-sr-only ui:focus:fixed ui:focus:top-2 ui:focus:left-2 ui:focus:z-[60] ui:rounded-md ui:bg-primary ui:px-3 ui:py-2 ui:text-primary-foreground ui:shadow"
			>
				Skip to content
			</a>

			<nav
				role="navigation"
				aria-label="Primary"
				className="ui:fixed ui:top-0 ui:inset-x-0 ui:z-50 ui:bg-background/95 ui:backdrop-blur-sm ui:border-b ui:border-border/60"
			>
				<div className="ui:relative ui:before:absolute ui:before:inset-x-0 ui:before:top-0 ui:before:h-px ui:before:bg-gradient-to-r ui:before:from-transparent ui:before:via-primary/20 ui:before:to-transparent">
					<div className="ui:px-4 ui:sm:px-6 ui:lg:px-8">
						<div className="ui:grid ui:grid-cols-[auto_1fr_auto] ui:items-center ui:h-14">
							{/* Zone A: Logo + Brand */}
							<BrandLogo priority />

							{/* Zone B: Primary Links (Desktop) */}
							<div className="ui:hidden ui:md:flex ui:items-center ui:justify-center ui:gap-6 ui:xl:gap-8">
								<NavLink href="/features">Features</NavLink>
								<NavLink href="/use-cases">Use Cases</NavLink>
								<NavLink href="/pricing">Pricing</NavLink>
								<NavLink href="/developers">Developers</NavLink>
								<NavLink href="/gpu-providers">Providers</NavLink>
								<NavLink href="/enterprise">Enterprise</NavLink>
								<NavLink href="https://github.com/veighnsche/llama-orch/tree/main/docs" target="_blank" rel="noopener">
									Docs
								</NavLink>
							</div>

							{/* Zone C: Actions (Desktop) */}
							<div className="ui:hidden ui:md:flex ui:items-center ui:gap-2 ui:justify-self-end">
								<div className="ui:flex ui:items-center ui:gap-1 ui:rounded-xl ui:p-0.5 ui:bg-muted/30 ui:ring-1 ui:ring-border/60">
									<IconButton asChild aria-label="Open rbee on GitHub" title="GitHub">
										<a href="https://github.com/veighnsche/llama-orch" target="_blank" rel="noopener noreferrer">
											<GitHubIcon className="ui:size-5" />
										</a>
									</IconButton>

									<ThemeToggle />
								</div>

								<Button
									className="ui:bg-primary ui:hover:bg-primary/85 ui:text-primary-foreground ui:h-9"
									data-umami-event="cta:join-waitlist"
									aria-label="Join the rbee waitlist"
								>
									Join Waitlist
								</Button>
							</div>

							{/* Mobile Menu Toggle */}
							<Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
								<SheetTrigger asChild>
									<IconButton className="ui:md:hidden" aria-label="Toggle menu">
										{mobileMenuOpen ? <X className="ui:size-6" aria-hidden /> : <Menu className="ui:size-6" aria-hidden />}
									</IconButton>
								</SheetTrigger>
								<SheetContent side="top" className="ui:top-14 ui:border-t-0 ui:pt-4 ui:pb-[calc(env(safe-area-inset-bottom)+1rem)]">
									<SheetTitle className="ui:sr-only">Navigation Menu</SheetTitle>
									<div className="ui:space-y-3">
										<NavLink href="/features" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
											Features
										</NavLink>
										<NavLink href="/use-cases" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
											Use Cases
										</NavLink>
										<NavLink href="/pricing" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
											Pricing
										</NavLink>
										<NavLink href="/developers" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
											Developers
										</NavLink>
										<NavLink href="/gpu-providers" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
											Providers
										</NavLink>
										<NavLink href="/enterprise" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
											Enterprise
										</NavLink>
										<NavLink
											href="https://github.com/veighnsche/llama-orch/tree/main/docs"
											variant="mobile"
											onClick={() => setMobileMenuOpen(false)}
											target="_blank"
											rel="noopener"
										>
											Docs
										</NavLink>

										<Separator className="ui:my-2 ui:opacity-60" />

										<a
											href="https://github.com/veighnsche/llama-orch"
											target="_blank"
											rel="noopener noreferrer"
											className="ui:flex ui:items-center ui:gap-2 ui:py-2 ui:text-muted-foreground ui:hover:text-foreground ui:transition-colors"
										>
											<GitHubIcon className="ui:size-5" />
											<span>GitHub</span>
										</a>

										<Button
											className="ui:w-full ui:bg-primary ui:hover:bg-primary/85 ui:text-primary-foreground ui:h-9 ui:mt-2"
											data-umami-event="cta:join-waitlist"
											aria-label="Join the rbee waitlist"
										>
											Join Waitlist
										</Button>
									</div>
								</SheetContent>
							</Sheet>
						</div>
					</div>
				</div>
			</nav>
		</>
	)
}
