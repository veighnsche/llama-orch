'use client'

import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight, Github, Star, DollarSign } from 'lucide-react'
import { PulseBadge, TrustIndicator, TerminalWindow, ProgressBar, FloatingKPICard } from '@/components/molecules'
import Image from 'next/image'
import { useState, useEffect } from 'react'

export function HeroSection() {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) {
      setIsVisible(true)
    } else {
      const timer = setTimeout(() => setIsVisible(true), 50)
      return () => clearTimeout(timer)
    }
  }, [])

  return (
    <section aria-labelledby="hero-title" className="relative isolate min-h-[88svh] flex items-center bg-background">
      <div className="container mx-auto px-4 py-24">
        <div className="grid lg:grid-cols-12 gap-12 items-center">
          {/* Cols 1–6: Messaging Stack */}
          <div className="lg:col-span-6 space-y-8">
            {/* Top Badge */}
            <PulseBadge text="100% Open Source • GPL-3.0-or-later" />

            {/* Headline */}
            <h1
              id="hero-title"
              className={`text-5xl md:text-6xl lg:text-6xl font-bold leading-tight text-balance transition-opacity duration-250 ${
                isVisible ? 'opacity-100' : 'opacity-0'
              }`}
            >
              AI Infrastructure.
              <br />
              <span className="text-primary">On Your Terms.</span>
            </h1>

            {/* Subcopy */}
            <p className="text-xl text-muted-foreground leading-8 max-w-[58ch]">
              Run LLMs on your hardware—across any GPUs and machines. Build with AI, keep control, and avoid vendor
              lock-in.
            </p>

            {/* Micro-proof bullets */}
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <span className="text-chart-3" aria-hidden="true">
                  ✓
                </span>
                Your GPUs, your network
              </li>
              <li className="flex items-center gap-2">
                <span className="text-chart-3" aria-hidden="true">
                  ✓
                </span>
                Zero API fees
              </li>
              <li className="flex items-center gap-2">
                <span className="text-chart-3" aria-hidden="true">
                  ✓
                </span>
                Drop-in OpenAI API
              </li>
            </ul>

            {/* CTA Group */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                size="lg"
                className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8 focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                aria-label="Get started with rbee for free"
                data-umami-event="cta:get-started"
              >
                Get Started Free
                <ArrowRight className="ml-2 h-5 w-5" aria-hidden="true" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
                asChild
              >
                <a href="/docs">View Docs</a>
              </Button>
            </div>

            {/* Bottom Support Row: Trust Badges */}
            <ul className="flex flex-wrap gap-6 pt-4" role="list">
              <li>
                <a
                  href="https://github.com/veighnsche/llama-orch"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 rounded-sm"
                >
                  <Star className="h-5 w-5" aria-hidden="true" />
                  <span className="text-sm">Star on GitHub</span>
                  <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5" aria-hidden="true" />
                </a>
              </li>
              <li className="flex items-center gap-3 text-muted-foreground">
                <div
                  className="h-5 px-1.5 flex items-center justify-center text-xs font-bold border border-border rounded-sm"
                  aria-hidden="true"
                >
                  API
                </div>
                <span className="text-sm">OpenAI-Compatible</span>
              </li>
              <li className="flex items-center gap-3 text-muted-foreground">
                <DollarSign className="h-5 w-5" aria-hidden="true" />
                <span className="text-sm">$0 • No Cloud Required</span>
              </li>
            </ul>
          </div>

          {/* Cols 7–12: Visual Stack */}
          <div className="lg:col-span-6 space-y-12">
            {/* Terminal Window with Floating KPI */}
            <div className="relative max-w-[520px] lg:max-w-none mx-auto mb-8">
              <TerminalWindow title="rbee-keeper">
                <div className="space-y-3">
                  <div className="text-muted-foreground">
                    <span className="text-chart-3">$</span> rbee-keeper infer --model llama-3.1-70b
                  </div>
                  <div className="text-foreground pl-4">
                    <span className="text-primary">→</span> Loading model across 3 GPUs...
                  </div>
                  <div className="text-foreground pl-4">
                    <span className="text-chart-3">✓</span> Model ready (2.3s)
                  </div>
                  <div className="text-muted-foreground pl-4">
                    <span className="text-chart-2">Prompt:</span> Generate REST API
                  </div>
                  <div className="text-foreground pl-4 leading-relaxed" aria-live="polite" aria-atomic="true">
                    <span className="text-primary animate-pulse" aria-hidden="true">
                      ▊
                    </span>{' '}
                    Generating code...
                  </div>

                  {/* GPU Utilization */}
                  <div className="pt-4 space-y-2">
                    <div className="text-muted-foreground text-xs">GPU Pool (5 nodes):</div>
                    <div className="space-y-1">
                      <ProgressBar label="Gaming PC 1" percentage={91} />
                      <ProgressBar label="Gaming PC 2" percentage={88} />
                      <ProgressBar label="Gaming PC 3" percentage={76} />
                      <ProgressBar label="Workstation" percentage={85} />
                    </div>
                  </div>

                  {/* Cost Counter */}
                  <div className="pt-2 flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Local Inference</span>
                    <span className="text-chart-3 font-bold">$0.00</span>
                  </div>
                </div>
              </TerminalWindow>

              {/* Floating KPI Card */}
              <FloatingKPICard />
            </div>

            {/* Network Diagram (lg+ only) */}
            <div className="hidden lg:block">
              <div className="aspect-[16/9] rounded-2xl ring-1 ring-border/60 bg-card overflow-hidden">
                <Image
                  src="/images/homelab-network.png"
                  width={1280}
                  height={720}
                  priority
                  className="rounded-2xl ring-1 ring-border/60 w-full h-full object-cover"
                  alt="Isometric 3D illustration of a distributed homelab AI orchestration network in a cozy, dimly-lit room with warm ambient lighting. CENTRAL HUB: Small mini PC (Intel NUC or similar compact form factor) in the center, acting as orchestrator/coordinator, glowing amber status LED, compact and unassuming but clearly the network hub with cables radiating outward. WORKER NODES arranged in a spoke pattern around the hub: (1) Gaming PC #1 (front-left) - full tower with tempered glass showing large illuminated GPU (RTX 4090 style), RGB fans, prominent graphics card, (2) Gaming PC #2 (front-right) - mid-tower with visible dual-GPU setup, glowing amber accents, aggressive gaming aesthetic, (3) Gaming PC #3 (back-left) - compact gaming build with single high-end GPU visible through mesh panel, (4) Workstation tower (back-right) - professional black case with subtle RGB, multiple GPU configuration visible, (5) Mac Studio (back-center, smallest and least prominent) - small silver aluminum cube, minimal presence, tucked in background. NETWORK TOPOLOGY: Thick amber/orange (#f59e0b) ethernet cables connecting mini PC hub to each worker node in a clear star/hub-spoke pattern, suggesting centralized orchestration. Small animated data packets (glowing dots) flowing FROM mini PC TO worker nodes. Each machine labeled with clean sans-serif text: 'Orchestrator' (mini PC), 'Gaming PC 1', 'Gaming PC 2', 'Gaming PC 3', 'Workstation', 'Mac Studio'. HIERARCHY: Mini PC is visually central and elevated slightly, gaming PCs are largest and most prominent with glowing GPUs clearly visible, Mac Studio is smallest and in background. Floating UI overlay showing: GPU pool metrics (5 nodes, 8+ GPUs total), network topology diagram, 'Cost: $0.00/hr' badge in emerald green (#10b981). Background: deep navy blue (#0f172a) gradient. Cinematic volumetric lighting from top highlighting the mini PC hub. Style: minimal, technical, clear visual hierarchy showing distributed GPU compute coordinated by central mini PC. Color palette: navy background, amber/orange cables and GPU glow, emerald green cost indicator, white labels. Perspective: 30-degree isometric. Mood: distributed computing power, homelab orchestration, 'your idle gaming PCs become AI infrastructure' narrative."
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
