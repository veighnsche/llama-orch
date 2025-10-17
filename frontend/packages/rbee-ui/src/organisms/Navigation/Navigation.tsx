'use client'

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
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@rbee/ui/atoms/Accordion'
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
import { useState } from 'react'

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
        aria-label="Primary"
        className="fixed top-0 inset-x-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border/60"
      >
        <div className="relative before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-primary/20 before:to-transparent">
          <div className="px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-[auto_1fr_auto] items-center h-14">
              {/* Zone A: Logo + Brand */}
              <BrandLogo priority />

              {/* Zone B: Dropdown Menus (Desktop) */}
              <div className="hidden md:flex items-center justify-center">
                <NavigationMenu viewport={false}>
                  <NavigationMenuList className="gap-1">
                    {/* Product Dropdown */}
                    <NavigationMenuItem>
                      <NavigationMenuTrigger>Product</NavigationMenuTrigger>
                      <NavigationMenuContent>
                        <ul className="grid w-[200px] gap-1 p-2">
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/features" className="block select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <div className="text-sm font-medium leading-none">Features</div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/pricing" className="block select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <div className="text-sm font-medium leading-none">Pricing</div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/use-cases" className="block select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <div className="text-sm font-medium leading-none">Use Cases</div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                        </ul>
                      </NavigationMenuContent>
                    </NavigationMenuItem>

                    {/* Solutions Dropdown */}
                    <NavigationMenuItem>
                      <NavigationMenuTrigger>Solutions</NavigationMenuTrigger>
                      <NavigationMenuContent>
                        <ul className="grid w-[280px] gap-1 p-2">
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/developers" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Code className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">Developers</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">Build AI tools without vendor lock-in</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/enterprise" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Building className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">Enterprise</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">GDPR-compliant AI infrastructure</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/gpu-providers" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Server className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">Providers</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">Earn with your idle GPUs</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                        </ul>
                      </NavigationMenuContent>
                    </NavigationMenuItem>

                    {/* Industries Dropdown */}
                    <NavigationMenuItem>
                      <NavigationMenuTrigger>Industries</NavigationMenuTrigger>
                      <NavigationMenuContent>
                        <ul className="grid w-[560px] grid-cols-2 gap-1 p-2">
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/industries/startups" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Rocket className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">Startups</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">Scale your AI infrastructure</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/industries/homelab" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Home className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">Homelab</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">Self-hosted AI for enthusiasts</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/industries/research" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <FlaskConical className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">Research</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">Reproducible ML experiments</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/industries/compliance" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Shield className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">Compliance</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">EU-native, GDPR-ready</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/industries/education" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <GraduationCap className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">Education</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">Learn distributed AI systems</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/industries/devops" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Settings className="size-5 mt-0.5 shrink-0" />
                                <div>
                                  <div className="text-sm font-medium leading-none mb-1">DevOps</div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">Production-ready orchestration</p>
                                </div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                        </ul>
                      </NavigationMenuContent>
                    </NavigationMenuItem>

                    {/* Resources Dropdown */}
                    <NavigationMenuItem>
                      <NavigationMenuTrigger>Resources</NavigationMenuTrigger>
                      <NavigationMenuContent>
                        <ul className="grid w-[200px] gap-1 p-2">
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/community" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Users className="size-5 mt-0.5 shrink-0" />
                                <div className="text-sm font-medium leading-none">Community</div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/security" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Lock className="size-5 mt-0.5 shrink-0" />
                                <div className="text-sm font-medium leading-none">Security</div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                          <li>
                            <NavigationMenuLink asChild>
                              <Link href="/legal" className="flex items-start gap-3 select-none rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                                <Scale className="size-5 mt-0.5 shrink-0" />
                                <div className="text-sm font-medium leading-none">Legal</div>
                              </Link>
                            </NavigationMenuLink>
                          </li>
                        </ul>
                      </NavigationMenuContent>
                    </NavigationMenuItem>
                  </NavigationMenuList>
                </NavigationMenu>
              </div>

              {/* Zone C: Actions (Desktop) */}
              <div className="hidden md:flex items-center gap-3 justify-self-end">
                <NavLink href="https://github.com/veighnsche/llama-orch/tree/main/docs" target="_blank" rel="noopener" className="flex items-center gap-1.5 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                  <BookOpen className="size-4" />
                  Docs
                </NavLink>
                
                <div className="flex items-center gap-1 rounded-xl p-0.5 bg-muted/30 ring-1 ring-border/60">
                  <IconButton asChild aria-label="Open rbee on GitHub" title="GitHub">
                    <a href="https://github.com/veighnsche/llama-orch" target="_blank" rel="noopener noreferrer">
                      <GitHubIcon size={20} />
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
                  <IconButton className="md:hidden" aria-label="Toggle menu">
                    {mobileMenuOpen ? <X className="size-6" aria-hidden /> : <Menu className="size-6" aria-hidden />}
                  </IconButton>
                </SheetTrigger>
<SheetContent side="top" className="top-14 border-t-0 pt-4 pb-[calc(env(safe-area-inset-bottom)+1rem)] max-h-[calc(100vh-3.5rem)] overflow-y-auto">
                  <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
                  <div className="space-y-2">
                    <Accordion type="multiple" className="w-full">
                      {/* Product Accordion */}
                      <AccordionItem value="product">
                        <AccordionTrigger className="text-lg font-medium">Product</AccordionTrigger>
                        <AccordionContent className="space-y-2 pl-4">
                          <NavLink href="/features" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Features
                          </NavLink>
                          <NavLink href="/pricing" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Pricing
                          </NavLink>
                          <NavLink href="/use-cases" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Use Cases
                          </NavLink>
                        </AccordionContent>
                      </AccordionItem>

                      {/* Solutions Accordion */}
                      <AccordionItem value="solutions">
                        <AccordionTrigger className="text-lg font-medium">Solutions</AccordionTrigger>
                        <AccordionContent className="space-y-2 pl-4">
                          <NavLink href="/developers" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Developers
                          </NavLink>
                          <NavLink href="/enterprise" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Enterprise
                          </NavLink>
                          <NavLink href="/gpu-providers" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Providers
                          </NavLink>
                        </AccordionContent>
                      </AccordionItem>

                      {/* Industries Accordion */}
                      <AccordionItem value="industries">
                        <AccordionTrigger className="text-lg font-medium">Industries</AccordionTrigger>
                        <AccordionContent className="space-y-2 pl-4">
                          <NavLink href="/industries/startups" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Startups
                          </NavLink>
                          <NavLink href="/industries/homelab" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Homelab
                          </NavLink>
                          <NavLink href="/industries/research" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Research
                          </NavLink>
                          <NavLink href="/industries/compliance" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Compliance
                          </NavLink>
                          <NavLink href="/industries/education" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Education
                          </NavLink>
                          <NavLink href="/industries/devops" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            DevOps
                          </NavLink>
                        </AccordionContent>
                      </AccordionItem>

                      {/* Resources Accordion */}
                      <AccordionItem value="resources">
                        <AccordionTrigger className="text-lg font-medium">Resources</AccordionTrigger>
                        <AccordionContent className="space-y-2 pl-4">
                          <NavLink href="/community" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Community
                          </NavLink>
                          <NavLink href="/security" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Security
                          </NavLink>
                          <NavLink href="/legal" variant="mobile" onClick={() => setMobileMenuOpen(false)}>
                            Legal
                          </NavLink>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>

                    <Separator className="my-4 opacity-60" />

                    <NavLink
                      href="https://github.com/veighnsche/llama-orch/tree/main/docs"
                      variant="mobile"
                      onClick={() => setMobileMenuOpen(false)}
                      target="_blank"
                      rel="noopener"
                      className="flex items-center gap-2"
                    >
                      <BookOpen className="size-5" />
                      Docs
                    </NavLink>

                    <a
                      href="https://github.com/veighnsche/llama-orch"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 py-2 text-foreground hover:text-primary transition-colors text-lg"
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      <GitHubIcon size={20} />
                      <span>GitHub</span>
                    </a>

                    <Button
                      className="w-full bg-primary hover:bg-primary/85 text-primary-foreground h-9 mt-4"
                      data-umami-event="cta:join-waitlist"
                      aria-label="Join the rbee waitlist"
                      onClick={() => setMobileMenuOpen(false)}
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
