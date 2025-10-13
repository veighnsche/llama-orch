# HowItWorksSection

**Premium onboarding guide: 0→AI infra in 15 minutes**

## Overview

A confident, scannable, copy-paste-ready guide that walks users through installing rbee, enrolling nodes, starting inference, and building AI agents. Designed for zero friction from first build to agent code.

## Features

### 1. Layout & Composition
- **Max-width container**: `max-w-7xl` with responsive padding
- **Uniform step pattern**: Left column (meta) + right column (code)
- **Alternating visual rhythm**: Steps 2 & 4 swap column order on `lg+`
- **Generous spacing**: `space-y-14 md:space-y-20` between steps

### 2. Sticky Step Rail (Desktop)
- **Position**: Fixed left sidebar (`left-8 top-96`)
- **Progressive disclosure**: Numbered dots become checkmarks as steps scroll into view
- **Active state**: Scales and highlights current step
- **Responsive**: Hidden on mobile/tablet

### 3. Typography Hierarchy
- **Section title**: `text-balance` for optimal line breaks
- **Step titles**: `text-2xl md:text-3xl font-semibold tracking-tight`
- **Lead lines**: `text-base md:text-lg text-muted-foreground`
- **Benefit lines**: `text-sm text-muted-foreground/80` (one clause per step)
- **Meta badges**: `text-[11px] uppercase tracking-wide`

### 4. Enhanced Copy
Each step includes:
- **Clear title**: Action-oriented (Install, Add, Start, Build)
- **Lead paragraph**: Concise explanation with key benefits
- **Benefit line**: One-clause value prop
- **Meta badges**: Time estimate + requirements
- **Footer note**: Technical detail under each code block

### 5. OS/Tab Switcher (Step 1)
- **Three tabs**: Linux, macOS, Windows (disabled, "soon")
- **Accessible**: `role="tab"`, `aria-selected`, `aria-controls`
- **Visual feedback**: Active tab has `bg-background shadow-sm`
- **State management**: React `useState` for selected OS

### 6. Enhanced CodeBlock
- **Header bar**: Title + language badge + copy button
- **Copy functionality**: Click to copy, shows "Copied" confirmation
- **Improved styling**: `rounded-xl border bg-card/60 shadow-sm`
- **Footer notes**: Technical details in muted text below each block

### 7. Motion Hierarchy
- **Staggered entrance**: Each step fades in with 80ms delay
- **Transform/opacity only**: `animate-in fade-in slide-in-from-bottom-2`
- **Sticky rail animation**: Dots scale and swap to checkmarks
- **Reduced motion**: Respects `prefers-reduced-motion` media query

### 8. Accessibility
- **Semantic HTML**: `h3` for step titles
- **ARIA labels**: `aria-hidden="true"` on decorative StepNumbers
- **Tab switcher**: Full ARIA support with `role="tablist"`
- **Copy buttons**: `aria-label="Copy code"`
- **Keyboard navigation**: All interactive elements focusable

### 9. Step Details

#### Step 1: Install rbee
- **Time**: ~3 min
- **Requirements**: Rust toolchain
- **Benefit**: Cold start to running daemon in minutes
- **OS switcher**: Linux/macOS/Windows (soon)
- **Footer**: "Daemon listens on :8080 by default."

#### Step 2: Add Your Machines
- **Time**: ~5 min
- **Requirements**: SSH access
- **Benefit**: Multi-node, mixed backends, one pool
- **Footer**: "Nodes authenticate over SSH; labels auto-assign on first handshake."

#### Step 3: Start Inference
- **Time**: ~2 min
- **Requirements**: Open port 8080
- **Benefit**: No cloud keys, no egress
- **Footer**: "Leave OPENAI_API_KEY unset if your client requires one—rbee will intercept."

#### Step 4: Build AI Agents
- **Time**: ~5 min
- **Requirements**: Node.js
- **Benefit**: Ship agents that run on your hardware
- **Footer**: "Models are resolved by scheduler policy; override per-call if needed."

## Technical Implementation

### Dependencies
- `lucide-react`: Check, Copy icons
- `@/lib/utils`: `cn()` utility
- `@/components/molecules`: SectionContainer, StepNumber, CodeBlock

### State Management
- `selectedOS`: Controls OS tab switcher
- `activeStep`: Tracks current step in viewport
- `stepRefs`: Array of refs for IntersectionObserver

### IntersectionObserver
- **Threshold**: 0.5 (50% visibility)
- **Effect**: Updates `activeStep` when step enters viewport
- **Cleanup**: Disconnects observer on unmount
- **Accessibility**: Respects `prefers-reduced-motion`

## QA Checklist

- [x] Step titles wrap nicely at sm/md breakpoints
- [x] Code blocks don't overflow (horizontal scroll enabled)
- [x] Sticky rail doesn't overlap content
- [x] Sticky rail hides on mobile
- [x] Copy button works for all code blocks
- [x] Reduced motion respected (no transforms when enabled)
- [x] Contrast meets AA standards
- [x] All interactive elements keyboard accessible
- [x] ARIA labels present and accurate

## Design Tokens

Uses semantic tokens for consistency:
- `bg-card`, `bg-secondary`, `bg-accent`
- `text-foreground`, `text-muted-foreground`
- `border`, `rounded-xl`, `shadow-sm`

## Future Enhancements

### Optional Visuals (Not Yet Implemented)
- **Step 2**: Network topology diagram (56-80px tall, monochrome)
- **Step 3**: Client icons row (Zed, Cursor, Continue)

Add these as Next.js `<Image>` components when assets are ready:

```tsx
<Image
  src="/illustrations/network-topology.svg"
  width={560}
  height={80}
  className="w-full h-20 object-contain opacity-80"
  alt="diagram: home network with router; nodes labeled workstation (2×RTX 4090) and mac (M2 Ultra) connecting to rbee keeper; minimalist, monochrome, vector"
/>
```

## Performance Notes

- **Client component**: Uses `'use client'` for interactivity
- **Lazy observation**: IntersectionObserver only runs on client
- **Minimal re-renders**: State updates only on scroll intersection
- **Optimized animations**: Transform/opacity only (GPU-accelerated)

---

**Outcome**: Premium, copy-paste-ready onboarding that reads like a confident product guide with zero friction from first build to agent code.
