import { featuresHeroImage } from '@rbee/ui/assets'
import { BrandWordmark } from '@rbee/ui/atoms'
import { FeatureInfoCard } from '@rbee/ui/molecules'
import { HeroTemplate } from '@rbee/ui/templates/HeroTemplate'
import { Cpu, Download, Power } from 'lucide-react'
import Image from 'next/image'

// Feature flags for quick toggles
const SHOW_HONEYCOMB_BG = true
const SHOW_MOSAIC_IMAGE = false // Set to true when real photo exists
const SHOW_STAT_STRIP = true

export function FeaturesHero() {
  const asideContent = (
    <div className="relative order-last lg:order-none">
      <div className="grid grid-cols-2 gap-4 mx-auto lg:mx-0">
        {/* Left column: Image stacked above Programmable Scheduler */}
        <div className="flex flex-col gap-4">
          <Image
            src={featuresHeroImage}
            alt="Modern homelab server rack with glowing amber and blue LEDs, NVIDIA GPUs visible through tempered glass side panel, clean cable management, soft rim lighting from behind, shallow depth of field, cinematic composition, professional photography, dark background with subtle honeycomb pattern, conveying enterprise-grade hardware in a home environment"
            priority
            className="w-full h-auto object-cover rounded"
          />

          <FeatureInfoCard
            icon={<Cpu className="h-5 w-5 text-primary" aria-hidden="true" focusable="false" />}
            title="Programmable Scheduler"
            body="Write custom routing logic in Rhai scripts. Route by cost, latency, GPU type, or compliance requirements. Update policies without recompiling. 40+ built-in helpers for worker selection, VRAM queries, and smart eviction. Choose between Platform mode (immutable, multi-tenant fairness) or Home/Lab mode (fully customizable YAML or Rhai scripts). Optimize for cost savings (20-30%), latency (30-50% faster), or GDPR compliance with EU-only routing."
            tone="primary"
            size="sm"
            delay="duration-700 delay-200"
            className="h-full transition-transform will-change-transform hover:-translate-y-0.5"
          />
        </div>

        {/* Right column: Model Catalog stacked above Cascading Shutdown */}
        <div className="flex flex-col gap-4">
          <FeatureInfoCard
            icon={<Download className="h-5 w-5 text-chart-2" aria-hidden="true" focusable="false" />}
            title="Model Catalog"
            body="One click to load models from Hugging Face. Real-time download progress with bytes transferred and loading status. Automatic VRAM estimation and GPU compatibility checks. Supports multi-modal workloads: LLMs, Stable Diffusion, embeddings, and TTS. Smart caching prevents re-downloads."
            tone="chart2"
            size="sm"
            delay="duration-700 delay-300"
            className="transition-transform will-change-transform hover:-translate-y-0.5"
          />

          <FeatureInfoCard
            icon={<Power className="h-5 w-5 text-chart-3" aria-hidden="true" focusable="false" />}
            title="Cascading Shutdown"
            body="Ctrl+C cleanly tears down the entire distributed system. Orchestrator sends SIGTERM to all pool managers via SSH, which gracefully shut down workers with HTTP POST /shutdown. Models unload cleanly, no orphaned processes, no GPU memory leaks. Reliability guarantee: your system stays clean."
            tone="chart3"
            size="sm"
            delay="duration-700 delay-400"
            className="transition-transform will-change-transform hover:-translate-y-0.5"
          />
        </div>
      </div>

      {/* Optional background image (gated by flag) */}
      {SHOW_MOSAIC_IMAGE && (
        <div className="pointer-events-none select-none absolute inset-x-0 bottom-0 opacity-10 blur-[1px] hidden lg:block">
          {/* Placeholder for real asset - uncomment when image exists */}
          {/* <Image
            src="/images/homelab-rack.jpg"
            alt="high-contrast photo of a tidy homelab GPU mini-rack with soft rim light, shallow depth of field, cinematic 35mm, conveys approachable enterprise hardware at home"
            width={1200}
            height={800}
            priority
            className="w-full h-auto object-cover"
          /> */}
        </div>
      )}
    </div>
  )

  const subcopyContent = (
    <>
      <BrandWordmark size="lg" /> (pronounced &ldquo;are-bee&rdquo;) gives you enterprise AI orchestration that runs on
      your own hardware. OpenAI-compatible. No cloud lock-in. No ongoing API costs. Full control.
    </>
  )

  return (
    <>
      <HeroTemplate
        badge={{ variant: 'none' }}
        headline={{
          variant: 'two-line-highlight',
          prefix: 'Enterprise-grade AI.',
          highlight: 'Homelab simple.',
        }}
        subcopy={subcopyContent}
        subcopyMaxWidth="wide"
        proofElements={{
          variant: 'badges',
          items: [
            { text: 'Runs on your GPUs' },
            { text: 'OpenAI-compatible' },
            { text: 'GDPR-ready' },
            { text: 'CUDA · Metal · CPU' },
          ],
        }}
        ctas={{
          primary: {
            label: 'See all features',
            href: '#feature-list',
          },
          secondary: {
            label: 'How it works',
            href: '#how-it-works',
            variant: 'secondary',
          },
        }}
        aside={asideContent}
        asideAriaLabel="Feature showcase with homelab hardware and key capabilities"
        background={
          SHOW_HONEYCOMB_BG ? { variant: 'honeycomb', size: 'small', fadeDirection: 'bottom' } : { variant: 'gradient' }
        }
        padding="default"
      />

      {/* Stat strip */}
      {SHOW_STAT_STRIP && (
        <div className="container mx-auto px-4 relative z-10 pb-6">
          <div className="mt-6 pt-0 text-sm text-muted-foreground grid grid-cols-1 sm:grid-cols-3 gap-2 text-center sm:text-left">
            <div className="text-left">
              <strong className="font-semibold">42/62</strong> BDD scenarios passing
            </div>
            <div className="text-center">
              <strong className="font-semibold">Zero</strong> cloud dependencies
            </div>
            <div className="text-right">
              Multi-backend: <strong className="font-semibold">CUDA · Metal · CPU</strong>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
