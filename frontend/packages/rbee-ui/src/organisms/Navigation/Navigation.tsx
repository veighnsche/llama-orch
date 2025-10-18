'use client'

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@rbee/ui/atoms/Accordion'
import { Button } from '@rbee/ui/atoms/Button'
import { IconButton } from '@rbee/ui/atoms/IconButton'
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
} from '@rbee/ui/atoms/NavigationMenu'
import { Separator } from '@rbee/ui/atoms/Separator'
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from '@rbee/ui/atoms/Sheet'
import { GitHubIcon } from '@rbee/ui/icons'
import { BrandLogo, NavLink, ThemeToggle } from '@rbee/ui/molecules'
import {
  BookOpen,
  Building,
  Code,
  FlaskConical,
  GraduationCap,
  Home,
  Lock,
  Menu,
  Rocket,
  Scale,
  Server,
  Settings,
  Shield,
  Users,
  X,
} from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'

export function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const pathname = usePathname()

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
        aria-label="Primary"
        className="fixed top-0 inset-x-0 z-50 bg-background/95 supports-[backdrop-filter]:bg-background/70 backdrop-blur-sm border-b border-border/60"
      >
        <div className="relative before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-primary/15 before:to-transparent">
          <div className="px-4 sm:px-6 lg:px-8 mx-auto max-w-7xl">
            <div className="grid grid-cols-[auto_1fr_auto] items-center h-16 md:h-14">
              {/* Zone A: Logo + Brand */}
              <BrandLogo priority />

              {/* Zone B: Dropdown Menus (Desktop) */}
              <div className="hidden md:flex items-center justify-center font-sans">
                <NavigationMenu viewport={false}>
                  <NavigationMenuList className="gap-2">
                    {/* Platform Dropdown */}
                    <NavigationMenuItem>
                      <NavigationMenuTrigger className="px-2 text-sm font-medium text-foreground/80 hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2">
                        Platform
                      </NavigationMenuTrigger>
                      <NavigationMenuContent className="animate-fade-in md:motion-safe:animate-slide-in-down border border-border">
                        <div className="grid gap-1 p-3 w-[280px]">
                          <ul className="grid gap-1">
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/features"
                                  className="block select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/features' ? 'page' : undefined}
                                >
                                  <div className="text-sm font-medium leading-none">Features</div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/pricing"
                                  className="block select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/pricing' ? 'page' : undefined}
                                >
                                  <div className="text-sm font-medium leading-none">Pricing</div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/use-cases"
                                  className="block select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/use-cases' ? 'page' : undefined}
                                >
                                  <div className="text-sm font-medium leading-none">Use Cases</div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                          </ul>
                          {/* Quick Start Rail */}
                          <div className="mt-2 flex items-center justify-between rounded-lg bg-muted/30 p-2 ring-1 ring-border/50">
                            <Link
                              href="https://github.com/veighnsche/llama-orch/tree/main/docs"
                              target="_blank"
                              rel="noopener"
                              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                            >
                              <BookOpen className="size-3.5" />
                              Docs
                            </Link>
                            <Button size="sm" className="h-7 text-xs" data-umami-event="cta:join-waitlist-platform">
                              Join Waitlist
                            </Button>
                          </div>
                        </div>
                      </NavigationMenuContent>
                    </NavigationMenuItem>

                    {/* Solutions Dropdown */}
                    <NavigationMenuItem>
                      <NavigationMenuTrigger className="px-2 text-sm font-medium text-foreground/80 hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2">
                        Solutions
                      </NavigationMenuTrigger>
                      <NavigationMenuContent className="animate-fade-in md:motion-safe:animate-slide-in-down border border-border">
                        <div className="grid gap-1 p-3 w-[280px]">
                          <ul className="grid gap-1">
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/developers"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/developers' ? 'page' : undefined}
                                >
                                  <Code className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">Developers</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      Build agents on your own hardware. OpenAI-compatible, drop-in.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/enterprise"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/enterprise' ? 'page' : undefined}
                                >
                                  <Building className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">Enterprise</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      GDPR-native orchestration with audit trails and controls.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/gpu-providers"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/gpu-providers' ? 'page' : undefined}
                                >
                                  <Server className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">Providers</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      Monetize idle GPUs. Task-based payouts.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                          </ul>
                          {/* Quick Start Rail */}
                          <div className="mt-2 flex items-center justify-between rounded-lg bg-muted/30 p-2 ring-1 ring-border/50">
                            <Link
                              href="https://github.com/veighnsche/llama-orch/tree/main/docs"
                              target="_blank"
                              rel="noopener"
                              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                            >
                              <BookOpen className="size-3.5" />
                              Docs
                            </Link>
                            <Button size="sm" className="h-7 text-xs" data-umami-event="cta:join-waitlist-solutions">
                              Join Waitlist
                            </Button>
                          </div>
                        </div>
                      </NavigationMenuContent>
                    </NavigationMenuItem>

                    {/* Industries Dropdown */}
                    <NavigationMenuItem>
                      <NavigationMenuTrigger className="px-2 text-sm font-medium text-foreground/80 hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2">
                        Industries
                      </NavigationMenuTrigger>
                      <NavigationMenuContent className="animate-fade-in md:motion-safe:animate-slide-in-down border border-border">
                        <div className="grid gap-1 p-3 w-[560px]">
                          <ul className="grid grid-cols-2 gap-1">
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/industries/startups"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/industries/startups' ? 'page' : undefined}
                                >
                                  <Rocket className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">Startups</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      Prototype fast. Own your stack from day one.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/industries/homelab"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/industries/homelab' ? 'page' : undefined}
                                >
                                  <Home className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">Homelab</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      Self-hosted LLMs across all your machines.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/industries/research"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/industries/research' ? 'page' : undefined}
                                >
                                  <FlaskConical className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">Research</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      Reproducible runs with deterministic seeds.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/industries/legal"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/industries/legal' ? 'page' : undefined}
                                >
                                  <Scale className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">Legal</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      AI for law firms. Document review at scale.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/industries/education"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/industries/education' ? 'page' : undefined}
                                >
                                  <GraduationCap className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">Education</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      Teach distributed AI with real infra.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/industries/devops"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/industries/devops' ? 'page' : undefined}
                                >
                                  <Settings className="size-5 mt-0.5 shrink-0" />
                                  <div>
                                    <div className="text-sm font-medium leading-none mb-1">DevOps</div>
                                    <p className="text-[13px] leading-[1.2] text-muted-foreground">
                                      SSH-first lifecycle. No orphaned workers.
                                    </p>
                                  </div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                          </ul>
                          {/* Quick Start Rail */}
                          <div className="mt-2 flex items-center justify-between rounded-lg bg-muted/30 p-2 ring-1 ring-border/50">
                            <Link
                              href="https://github.com/veighnsche/llama-orch/tree/main/docs"
                              target="_blank"
                              rel="noopener"
                              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                            >
                              <BookOpen className="size-3.5" />
                              Docs
                            </Link>
                            <Button size="sm" className="h-7 text-xs" data-umami-event="cta:join-waitlist-industries">
                              Join Waitlist
                            </Button>
                          </div>
                        </div>
                      </NavigationMenuContent>
                    </NavigationMenuItem>

                    {/* Resources Dropdown */}
                    <NavigationMenuItem>
                      <NavigationMenuTrigger className="px-2 text-sm font-medium text-foreground/80 hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2">
                        Resources
                      </NavigationMenuTrigger>
                      <NavigationMenuContent className="animate-fade-in md:motion-safe:animate-slide-in-down border border-border">
                        <div className="grid gap-1 p-3 w-[200px]">
                          <ul className="grid gap-1">
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/community"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/community' ? 'page' : undefined}
                                >
                                  <Users className="size-5 mt-0.5 shrink-0" />
                                  <div className="text-sm font-medium leading-none">Community</div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/security"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/security' ? 'page' : undefined}
                                >
                                  <Lock className="size-5 mt-0.5 shrink-0" />
                                  <div className="text-sm font-medium leading-none">Security</div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                            <li>
                              <NavigationMenuLink asChild>
                                <Link
                                  href="/compliance"
                                  className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                                  aria-current={pathname === '/compliance' ? 'page' : undefined}
                                >
                                  <Shield className="size-5 mt-0.5 shrink-0" />
                                  <div className="text-sm font-medium leading-none">Compliance</div>
                                </Link>
                              </NavigationMenuLink>
                            </li>
                          </ul>
                          {/* Quick Start Rail */}
                          <div className="mt-2 flex items-center justify-between rounded-lg bg-muted/30 p-2 ring-1 ring-border/50">
                            <Link
                              href="https://github.com/veighnsche/llama-orch/tree/main/docs"
                              target="_blank"
                              rel="noopener"
                              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                            >
                              <BookOpen className="size-3.5" />
                              Docs
                            </Link>
                            <Button size="sm" className="h-7 text-xs" data-umami-event="cta:join-waitlist-resources">
                              Join Waitlist
                            </Button>
                          </div>
                        </div>
                      </NavigationMenuContent>
                    </NavigationMenuItem>
                  </NavigationMenuList>
                </NavigationMenu>
              </div>

              {/* Zone C: Actions (Desktop) */}
              <div className="hidden md:flex items-center gap-3 justify-self-end">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-9 px-2 gap-1 text-muted-foreground hover:text-foreground"
                  asChild
                >
                  <Link href="https://github.com/veighnsche/llama-orch/tree/main/docs" target="_blank" rel="noopener">
                    <BookOpen className="size-4" />
                    Docs
                  </Link>
                </Button>

                <div className="flex items-center gap-1 rounded-xl p-0.5 bg-muted/40 ring-1 ring-border/60 shadow-[inset_0_0_0_1px_var(--border)]">
                  <IconButton asChild aria-label="Open rbee on GitHub" title="GitHub">
                    <a
                      href="https://github.com/veighnsche/llama-orch"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="motion-safe:hover:animate-pulse"
                    >
                      <GitHubIcon size={20} />
                    </a>
                  </IconButton>

                  <ThemeToggle />
                </div>
                <Button
                  className="bg-primary hover:bg-primary/85 text-primary-foreground h-9"
                  data-umami-event="cta:join-waitlist"
                  aria-label="Join the rbee waitlist"
                  title="Early access • Zero cost to join"
                >
                  Join Waitlist
                </Button>
              </div>

              {/* Mobile Menu Toggle */}
              <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
                <SheetTrigger asChild>
                  <IconButton className="md:hidden" aria-label="Toggle menu">
                    {mobileMenuOpen ? <X className="size-6" aria-hidden /> : <Menu className="size-6" aria-hidden />}
                  </IconButton>
                </SheetTrigger>
                <SheetContent
                  side="top"
                  className="top-16 md:top-14 border-t-0 pt-4 pb-[calc(env(safe-area-inset-bottom)+1rem)] max-h-[calc(100vh-4rem)] md:max-h-[calc(100vh-3.5rem)] overflow-y-auto"
                >
                  <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
                  <div className="sr-only" aria-live="polite" aria-atomic="true">
                    {mobileMenuOpen ? 'Main menu opened' : 'Main menu closed'}
                  </div>
                  <div className="motion-safe:animate-fade-in space-y-2">
                    {/* Sticky CTA Block */}
                    <div className="sticky top-0 z-10 -mx-2 px-2 pb-2 bg-gradient-to-b from-background to-transparent">
                      <div className="flex gap-2">
                        <Button variant="ghost" size="sm" className="flex-1 h-10" asChild>
                          <Link
                            href="https://github.com/veighnsche/llama-orch/tree/main/docs"
                            target="_blank"
                            rel="noopener"
                            onClick={() => setMobileMenuOpen(false)}
                          >
                            <BookOpen className="size-4 mr-1.5" />
                            Docs
                          </Link>
                        </Button>
                        <Button
                          size="sm"
                          className="flex-1 h-10"
                          data-umami-event="cta:join-waitlist-mobile"
                          onClick={() => setMobileMenuOpen(false)}
                        >
                          Join Waitlist
                        </Button>
                      </div>
                    </div>
                    <Accordion type="multiple" className="w-full">
                      {/* Platform Accordion */}
                      <AccordionItem value="platform">
                        <AccordionTrigger className="text-base md:text-lg min-h-12">Platform</AccordionTrigger>
                        <AccordionContent className="motion-safe:animate-slide-in-down space-y-2 pl-4">
                          <NavLink
                            href="/features"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/features' ? 'page' : undefined}
                          >
                            Features
                          </NavLink>
                          <NavLink
                            href="/pricing"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/pricing' ? 'page' : undefined}
                          >
                            Pricing
                          </NavLink>
                          <NavLink
                            href="/use-cases"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/use-cases' ? 'page' : undefined}
                          >
                            Use Cases
                          </NavLink>
                        </AccordionContent>
                      </AccordionItem>

                      {/* Solutions Accordion */}
                      <AccordionItem value="solutions">
                        <AccordionTrigger className="text-base md:text-lg min-h-12">Solutions</AccordionTrigger>
                        <AccordionContent className="motion-safe:animate-slide-in-down space-y-2 pl-4">
                          <NavLink
                            href="/developers"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/developers' ? 'page' : undefined}
                          >
                            Developers
                          </NavLink>
                          <NavLink
                            href="/enterprise"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/enterprise' ? 'page' : undefined}
                          >
                            Enterprise
                          </NavLink>
                          <NavLink
                            href="/gpu-providers"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/gpu-providers' ? 'page' : undefined}
                          >
                            Providers
                          </NavLink>
                        </AccordionContent>
                      </AccordionItem>

                      {/* Industries Accordion */}
                      <AccordionItem value="industries">
                        <AccordionTrigger className="text-base md:text-lg min-h-12">Industries</AccordionTrigger>
                        <AccordionContent className="motion-safe:animate-slide-in-down space-y-2 pl-4">
                          <NavLink
                            href="/industries/startups"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/industries/startups' ? 'page' : undefined}
                          >
                            Startups
                          </NavLink>
                          <NavLink
                            href="/industries/homelab"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/industries/homelab' ? 'page' : undefined}
                          >
                            Homelab
                          </NavLink>
                          <NavLink
                            href="/industries/research"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/industries/research' ? 'page' : undefined}
                          >
                            Research
                          </NavLink>
                          <NavLink
                            href="/industries/legal"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/industries/legal' ? 'page' : undefined}
                          >
                            Legal
                          </NavLink>
                          <NavLink
                            href="/industries/education"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/industries/education' ? 'page' : undefined}
                          >
                            Education
                          </NavLink>
                          <NavLink
                            href="/industries/devops"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/industries/devops' ? 'page' : undefined}
                          >
                            DevOps
                          </NavLink>
                        </AccordionContent>
                      </AccordionItem>

                      {/* Resources Accordion */}
                      <AccordionItem value="resources">
                        <AccordionTrigger className="text-base md:text-lg min-h-12">Resources</AccordionTrigger>
                        <AccordionContent className="motion-safe:animate-slide-in-down space-y-2 pl-4">
                          <NavLink
                            href="/community"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/community' ? 'page' : undefined}
                          >
                            Community
                          </NavLink>
                          <NavLink
                            href="/security"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/security' ? 'page' : undefined}
                          >
                            Security
                          </NavLink>
                          <NavLink
                            href="/compliance"
                            variant="mobile"
                            onClick={() => setMobileMenuOpen(false)}
                            className="py-3 text-lg min-h-12 flex items-center"
                            aria-current={pathname === '/compliance' ? 'page' : undefined}
                          >
                            Compliance
                          </NavLink>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>

                    <Separator className="my-4 opacity-60" />

                    {/* Utility Row */}
                    <div className="flex items-center gap-3 py-2 px-3 rounded-xl ring-1 ring-border/60">
                      <a
                        href="https://github.com/veighnsche/llama-orch"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => setMobileMenuOpen(false)}
                        aria-label="Open rbee on GitHub"
                      >
                        <GitHubIcon size={20} />
                        <span className="text-sm">GitHub</span>
                      </a>
                      <div className="ml-auto">
                        <ThemeToggle />
                      </div>
                    </div>
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
