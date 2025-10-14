'use client'

import Image from 'next/image'
import { ArrowRight, Shield, Zap, Cpu } from 'lucide-react'
import { SectionContainer, StatsGrid, IconBox } from '@rbee/ui/molecules'
import {
  Badge,
  Button,
  Card,
  Separator,
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  BrandMark,
  BrandWordmark,
} from '@rbee/ui/atoms'

export function WhatIsRbee() {
  return (
    <SectionContainer
      title="What is rbee?"
      bgVariant="secondary"
      maxWidth="5xl"
      className="py-16 md:py-20 motion-safe:animate-in motion-safe:fade-in-50 motion-safe:slide-in-from-bottom-2"
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
        {/* Left column: Content */}
        <div className="space-y-6">
          {/* Top badge */}
          <Badge variant="secondary" className="uppercase tracking-wide">
            Open-source • Self-hosted
          </Badge>

          {/* Headline */}
          <div className="space-y-3">
            <h2 className="text-4xl md:text-5xl font-semibold text-foreground">
              <BrandMark size="xl" className="inline-block align-middle mr-3" />
              <BrandWordmark size="4xl" inline />: your private AI infrastructure
            </h2>

            {/* Pronunciation badge */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge variant="outline" className="text-xs cursor-help" aria-label="rbee is pronounced are-bee">
                  pronounced "are-bee"
                </Badge>
              </TooltipTrigger>
              <TooltipContent>Pronounced like "R.B."</TooltipContent>
            </Tooltip>

            {/* Subhead */}
            <p className="mt-3 text-lg md:text-xl text-muted-foreground leading-relaxed max-w-prose">
              rbee is an open-source AI orchestration platform that unifies every computer in your home or office into a
              single, OpenAI-compatible AI cluster—private, controllable, and yours forever.
            </p>
          </div>

          {/* Separator (visible on md+) */}
          <Separator className="hidden md:block my-6" />

          {/* Value bullets */}
          <ul className="space-y-3 text-base text-foreground">
            <li className="flex items-start gap-3">
              <IconBox icon={Zap} size="sm" variant="rounded" color="primary" />
              <div>
                <strong className="font-semibold">Independence:</strong> Build on your hardware. No surprise model or
                pricing changes.
              </div>
            </li>
            <li className="flex items-start gap-3">
              <IconBox icon={Shield} size="sm" variant="rounded" color="primary" />
              <div>
                <strong className="font-semibold">Privacy:</strong> Code and data never leave your network.
              </div>
            </li>
            <li className="flex items-start gap-3">
              <IconBox icon={Cpu} size="sm" variant="rounded" color="primary" />
              <div>
                <strong className="font-semibold">All GPUs together:</strong> CUDA, Metal, and CPU—scheduled as one.
              </div>
            </li>
          </ul>

          {/* Stat cards grid */}
          <StatsGrid
            variant="cards"
            columns={3}
            className="pt-4"
            stats={[
              { value: '$0', label: 'No API fees (electricity only)' },
              { value: '100%', label: 'Your code never leaves your network' },
              { value: 'All', label: 'CUDA · Metal · CPU — orchestrated' },
            ]}
          />

          {/* Closing micro-copy */}
          <p className="text-base text-muted-foreground leading-relaxed max-w-prose">
            Build AI coders on your own hardware. OpenAI-compatible API, Zed/Cursor-ready. Your models, your rules.
          </p>

          {/* CTA row */}
          <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center md:justify-start">
            <Button size="lg" asChild>
              <a href="#get-started">Get Started Free</a>
            </Button>
            <Button variant="ghost" size="lg" asChild>
              <a href="/technical-deep-dive" className="flex items-center gap-2">
                See Architecture
                <ArrowRight className="size-4" />
              </a>
            </Button>
          </div>

          {/* Optional technical accent */}
          <p className="text-xs text-muted-foreground pt-2">
            <strong>Architecture at a glance:</strong> Smart/Dumb separation • Cascading shutdown • Multi-backend
          </p>
        </div>

        {/* Right column: Visual */}
        <div className="relative rounded-xl overflow-hidden ring-1 ring-border bg-card">
          {/* Network diagram image */}
          <div className="relative aspect-[16/9]">
            <div className="absolute top-4 left-4 z-10">
              <Badge variant="secondary" className="text-xs">
                Local Network
              </Badge>
            </div>
            <Image
              src="/images/homelab-network.png"
              width={1280}
              height={720}
              priority
              className="w-full h-full object-cover will-change-transform motion-safe:transition-transform motion-safe:duration-500 motion-safe:hover:scale-[1.01]"
              alt="Isometric 3D illustration of a distributed homelab AI orchestration network in a cozy, dimly-lit room with warm ambient lighting. CENTRAL HUB: Small mini PC (Intel NUC or similar compact form factor) in the center, acting as orchestrator/coordinator, glowing amber status LED, compact and unassuming but clearly the network hub with cables radiating outward. WORKER NODES arranged in a spoke pattern around the hub: (1) Gaming PC #1 (front-left) - full tower with tempered glass showing large illuminated GPU (RTX 4090 style), RGB fans, prominent graphics card, (2) Gaming PC #2 (front-right) - mid-tower with visible dual-GPU setup, glowing amber accents, aggressive gaming aesthetic, (3) Gaming PC #3 (back-left) - compact gaming build with single high-end GPU visible through mesh panel, (4) Workstation tower (back-right) - professional black case with subtle RGB, multiple GPU configuration visible, (5) Mac Studio (back-center, smallest and least prominent) - small silver aluminum cube, minimal presence, tucked in background. NETWORK TOPOLOGY: Thick amber/orange (#f59e0b) ethernet cables connecting mini PC hub to each worker node in a clear star/hub-spoke pattern, suggesting centralized orchestration. Small animated data packets (glowing dots) flowing FROM mini PC TO worker nodes. Each machine labeled with clean sans-serif text: 'Orchestrator' (mini PC), 'Gaming PC 1', 'Gaming PC 2', 'Gaming PC 3', 'Workstation', 'Mac Studio'. HIERARCHY: Mini PC is visually central and elevated slightly, gaming PCs are largest and most prominent with glowing GPUs clearly visible, Mac Studio is smallest and in background. Floating UI overlay showing: GPU pool metrics (5 nodes, 8+ GPUs total), network topology diagram, 'Cost: $0.00/hr' badge in emerald green (#10b981). Background: deep navy blue (#0f172a) gradient. Cinematic volumetric lighting from top highlighting the mini PC hub. Style: minimal, technical, clear visual hierarchy showing distributed GPU compute coordinated by central orchestrator."
            />
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
