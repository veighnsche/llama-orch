'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { GitHubIcon } from '@rbee/ui/atoms/GitHubIcon'
import { IconButton } from '@rbee/ui/atoms/IconButton'
import { Separator } from '@rbee/ui/atoms/Separator'
import { Sheet, SheetContent, SheetTrigger } from '@rbee/ui/atoms/Sheet'
import { Menu, X } from 'lucide-react'
import { useState } from 'react'
import { BrandLogo } from '@rbee/ui/molecules/BrandLogo'
import { ThemeToggle } from '@rbee/ui/molecules/ThemeToggle'
import { NavLink } from '@rbee/ui/molecules'

export function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <>
      {/* Skip to content link */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[60] rounded-md bg-primary px-3 py-2 text-primary-foreground shadow"
      >
        Skip to content
      </a>

      <nav
        role="navigation"
        aria-label="Primary"
        className="fixed top-0 inset-x-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border/60"
      >
        <div className="relative before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-primary/20 before:to-transparent">
          <div className="px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-[auto_1fr_auto] items-center h-14">
              {/* Zone A: Logo + Brand */}
              <BrandLogo priority />

              {/* Zone B: Primary Links (Desktop) */}
              <div className="hidden md:flex items-center justify-center gap-6 xl:gap-8">
                <NavLink href="/features">Features</NavLink>
                <NavLink href="/use-cases">Use Cases</NavLink>
                <NavLink href="/pricing">Pricing</NavLink>
                <NavLink href="/developers">Developers</NavLink>
                <NavLink href="/gpu-providers">Providers</NavLink>
                <NavLink href="/enterprise">Enterprise</NavLink>
                <NavLink
                  href="https://github.com/veighnsche/llama-orch/tree/main/docs"
                  target="_blank"
                  rel="noopener"
                >
                  Docs
                </NavLink>
              </div>

              {/* Zone C: Actions (Desktop) */}
              <div className="hidden md:flex items-center gap-2 justify-self-end">
                <div className="flex items-center gap-1 rounded-xl p-0.5 bg-muted/30 ring-1 ring-border/60">
                  <IconButton
                    asChild
                    aria-label="Open rbee on GitHub"
                    title="GitHub"
                  >
                    <a
                      href="https://github.com/veighnsche/llama-orch"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <GitHubIcon className="size-5" />
                    </a>
                  </IconButton>

                  <ThemeToggle />
                </div>

                <Button
                  className="bg-primary hover:bg-primary/85 text-primary-foreground h-9"
                  data-umami-event="cta:join-waitlist"
                  aria-label="Join the rbee waitlist"
                >
                  Join Waitlist
                </Button>
              </div>

              {/* Mobile Menu Toggle */}
              <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
                <SheetTrigger asChild>
                  <IconButton
                    className="md:hidden"
                    aria-label="Toggle menu"
                  >
                    {mobileMenuOpen ? <X className="size-6" aria-hidden /> : <Menu className="size-6" aria-hidden />}
                  </IconButton>
                </SheetTrigger>
                <SheetContent
                  side="top"
                  className="top-14 border-t-0 pt-4 pb-[calc(env(safe-area-inset-bottom)+1rem)]"
                >
                  <div className="space-y-3">
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

                    <Separator className="my-2 opacity-60" />

                    <a
                      href="https://github.com/veighnsche/llama-orch"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 py-2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <GitHubIcon className="size-5" />
                      <span>GitHub</span>
                    </a>

                    <Button
                      className="w-full bg-primary hover:bg-primary/85 text-primary-foreground h-9 mt-2"
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
