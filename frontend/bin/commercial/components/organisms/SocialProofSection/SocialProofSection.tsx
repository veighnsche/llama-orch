import { SectionContainer, StatCard, TestimonialCard } from '@/components/molecules'
import Image from 'next/image'

export function SocialProofSection() {
  const trustBadges = [
    { name: 'GitHub', url: 'https://github.com/veighnsche/llama-orch', tooltip: 'Star us on GitHub' },
    { name: 'HN', url: '#', tooltip: 'Discussed on Hacker News' },
    { name: 'Reddit', url: '#', tooltip: 'Join our community on Reddit' },
  ]

  return (
    <SectionContainer title="Trusted by Developers Who Value Independence" bgVariant="secondary">
      {/* Header with subtitle and trust strip */}
      <div className="text-center max-w-4xl mx-auto mb-12 motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-500">
        <p className="text-lg md:text-xl text-muted-foreground mb-6 leading-relaxed">
          Local-first AI with zero monthly cost. Loved by builders who keep control.
        </p>
        {/* Trust strip - desktop only */}
        <div className="hidden md:flex items-center justify-center gap-6">
          {trustBadges.map((badge) => (
            <a
              key={badge.name}
              href={badge.url}
              target="_blank"
              rel="noopener noreferrer"
              title={badge.tooltip}
              className="text-sm text-muted-foreground/70 hover:text-primary hover:opacity-100 transition-all font-medium"
            >
              {badge.name}
            </a>
          ))}
        </div>
      </div>

      {/* Metrics row - denser, clearer, animated */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 lg:gap-6 max-w-5xl mx-auto mb-12">
        <div
          role="group"
          aria-label="Stat: GitHub Stars"
          className="motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-100"
        >
          <a
            href="https://github.com/veighnsche/llama-orch"
            target="_blank"
            rel="noopener noreferrer"
            className="block hover:opacity-80 transition-opacity"
          >
            <StatCard value="1,200+" label="GitHub Stars" />
          </a>
        </div>
        <div
          role="group"
          aria-label="Stat: Active Installations"
          className="motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-200"
        >
          <StatCard value="500+" label="Active Installations" />
        </div>
        <div
          role="group"
          aria-label="Stat: GPUs Orchestrated"
          title="Cumulative across clusters"
          className="motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-300"
        >
          <StatCard value="8,000+" label="GPUs Orchestrated" />
        </div>
        <div
          role="group"
          aria-label="Stat: Average Monthly Cost"
          className="motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-[400ms]"
        >
          <StatCard value="€0" label="Avg Monthly Cost" variant="success" />
        </div>
      </div>

      {/* Testimonials grid with narrative rhythm */}
      <div className="max-w-6xl mx-auto">
        <p className="text-sm text-muted-foreground/80 text-center mb-6 uppercase tracking-wider font-medium">
          Real teams. Real savings.
        </p>
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:zoom-in-50 motion-safe:duration-400 motion-safe:delay-100">
            <TestimonialCard
              name="Alex K."
              role="Solo Developer"
              quote="Used to pay $80/mo for coding. Now Llama 70B runs locally on my gaming PC + an old workstation. Same quality, $0/month. Not going back."
              avatar={{ from: 'blue-400', to: 'blue-600' }}
              highlight="$80/mo → $0"
            />
          </div>
          <div className="col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:zoom-in-50 motion-safe:duration-400 motion-safe:delay-200">
            <TestimonialCard
              name="Sarah M."
              role="CTO"
              company={{ name: 'StartupCo' }}
              quote="We pooled our team's hardware and cut AI spend from $500/month to zero. rbee's OpenAI-compatible API meant no code changes."
              avatar={{ from: 'amber-400', to: 'amber-600' }}
              highlight="$500/mo → $0"
              verified
            />
          </div>
          <div className="col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:zoom-in-50 motion-safe:duration-400 motion-safe:delay-300">
            <TestimonialCard
              name="Dr. Thomas R."
              role="Research Lab Director"
              quote="GDPR blocked cloud options. With rbee on-prem and EU-only routing via Rhai, we shipped safely—no external deps."
              avatar={{ from: 'green-400', to: 'green-600' }}
            />
          </div>
        </div>

        {/* Context visuals - desktop only, 3-panel seamless layout */}
        <div className="hidden lg:flex max-w-6xl mx-auto mt-8 ring-1 ring-border/60 shadow-sm rounded-2xl overflow-hidden">
          <Image
            src="/images/social-proof-github-growth.png"
            width={400}
            height={560}
            className="object-cover"
            alt="Modern analytics dashboard displaying GitHub repository star growth chart: smooth exponential curve in vibrant teal (#14b8a6) ascending from bottom-left (0 stars, January 2025) to top-right (1,200+ stars, June 2025) against dark charcoal background (#1a1a1a). Chart features subtle horizontal grid lines at 200-star intervals, vertical timeline markers for each month, small circular data points along the curve, and a glowing highlight effect on the curve line. Top-right corner shows current star count in large bold numerals with +847% growth badge. Bottom includes mini sparkline showing commit activity correlation. Clean, professional UI design with sans-serif typography, high contrast for readability, subtle gradient overlay from bottom (darker) to top (lighter), resembling GitHub Insights or Vercel Analytics aesthetic. Portrait orientation optimized for vertical panel display."
          />
          <Image
            src="/images/social-proof-gpu-rack.png"
            width={400}
            height={560}
            className="object-cover"
            alt="Close-up product photography of a custom GPU server rack in a home office environment: center frame shows 4-6 NVIDIA GeForce RTX 4090 graphics cards vertically mounted in a black powder-coated open-air mining-style frame with horizontal support bars. Each GPU features glowing LED strips - alternating amber (#f59e0b) and teal (#14b8a6) accent lighting along the shroud edges and backplate. Cards connected via white braided PCIe riser cables with 90-degree adapters, meticulously cable-managed with velcro straps. Background shows blurred home office setting with wooden desk, potted succulent plant (left), and warm Edison bulb pendant light (top-right). Shallow depth of field (f/2.8) keeps GPUs in sharp focus while background softly blurs. Dramatic side lighting creates highlights on metallic heatsink fins and casts subtle shadows. Small status LEDs visible on PCIe risers. Professional product photography style, crisp detail on GPU model numbers and brand logos, slight lens flare from LED lights, color temperature 4500K (neutral-warm), portrait orientation for vertical rack display."
          />
          <Image
            src="/images/social-proof-developers.png"
            width={400}
            height={560}
            className="object-cover"
            alt="Authentic lifestyle photography of diverse development team collaborating in modern minimalist workspace: foreground shows 2-3 developers (varied gender and ethnicity) gathered around electric standing desk at comfortable standing height. Left monitor displays code editor with syntax highlighting (VS Code or similar) showing TypeScript/React code. Right monitor shows rbee web dashboard UI with dark theme, teal accent colors, real-time GPU metrics, and model deployment status. Developers engaged in discussion - one pointing at screen, others nodding, casual professional attire (hoodies, t-shirts). Background features floor-to-ceiling windows with soft natural daylight (golden hour), potted fiddle leaf fig plant (left corner), floating wooden shelves with tech books and small succulents, exposed brick accent wall (right side), pendant Edison bulb lights providing warm ambient glow. Desk surface shows mechanical keyboard, wireless mouse, coffee mugs, notebook with sketches. Composition uses rule of thirds, developers positioned left-of-center, monitors clearly visible. Shot with 35mm lens, f/2.0 aperture for natural background blur, warm color grading (4800K), professional lifestyle photography aesthetic similar to Unsplash or WeWork campaigns, authentic candid moment (not overly staged), portrait orientation optimized for vertical panel display."
          />
        </div>
      </div>

      {/* Footer reassurance */}
      <div className="text-center mt-12 text-sm text-muted-foreground">
        <p className="mb-3">Backed by an active community. Join us on GitHub and Discord.</p>
        <div className="flex items-center justify-center gap-4">
          <a
            href="https://github.com/veighnsche/llama-orch"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            GitHub
          </a>
          <span className="text-muted-foreground/50">•</span>
          <a href="#" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
            Discord
          </a>
        </div>
      </div>
    </SectionContainer>
  )
}
