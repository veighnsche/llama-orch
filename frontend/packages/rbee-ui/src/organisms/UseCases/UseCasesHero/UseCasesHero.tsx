import { useCasesHero } from '@rbee/ui/assets'
import { Button } from '@rbee/ui/atoms/Button'
import Image from 'next/image'

export function UseCasesHero() {
  return (
    <section className="relative overflow-hidden py-24 lg:py-28 bg-gradient-to-b from-background to-card">
      {/* Radial glow overlay */}
      <div aria-hidden className="pointer-events-none absolute inset-0 opacity-50">
        <div className="absolute -top-1/3 right-[-20%] h-[60rem] w-[60rem] rounded-full bg-primary/5 blur-3xl" />
      </div>

      <div className="container mx-auto px-4">
        <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
          {/* Left: copy stack */}
          <div className="max-w-2xl animate-in fade-in-50 slide-in-from-left-4">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border/60 bg-secondary px-3 py-1 text-sm text-secondary-foreground">
              <span className="font-medium">OpenAI-compatible</span>
              <span className="text-muted-foreground"> • your hardware, your rules</span>
            </div>

            <h1 className="text-balance text-5xl lg:text-6xl font-bold text-foreground tracking-tight">
              Built for Those Who Value{' '}
              <span className="bg-gradient-to-r from-primary to-foreground bg-clip-text text-transparent">
                Independence
              </span>
            </h1>

            <p className="mt-6 text-xl text-muted-foreground leading-relaxed">
              From solo developers to enterprises, rbee adapts to your needs. Own your AI infrastructure without
              compromising on power or flexibility.
            </p>

            {/* Action rail */}
            <div className="mt-8 flex flex-col sm:flex-row sm:items-center gap-3">
              <Button className="h-11 px-6 text-base animate-in zoom-in-50" asChild>
                <a href="#use-cases">Explore use cases</a>
              </Button>
              <Button variant="secondary" className="h-11 px-6 text-base" asChild>
                <a href="#architecture">See the architecture</a>
              </Button>

              {/* Audience chips */}
              <div className="sm:ml-4 flex flex-wrap gap-2">
                <a
                  href="#developers"
                  className="inline-flex items-center rounded-full border/60 bg-card px-3 py-1 text-sm text-foreground hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring transition-colors"
                >
                  Developers
                </a>
                <a
                  href="#enterprise"
                  className="inline-flex items-center rounded-full border/60 bg-card px-3 py-1 text-sm text-foreground hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring transition-colors"
                >
                  Enterprise
                </a>
                <a
                  href="#homelab"
                  className="inline-flex items-center rounded-full border/60 bg-card px-3 py-1 text-sm text-foreground hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring transition-colors"
                >
                  Homelab
                </a>
              </div>
            </div>

            {/* Quick proof row */}
            <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-emerald-500" aria-hidden /> Self-hosted control
              </span>
              <span className="hidden sm:inline">•</span>
              <span>OpenAI-compatible API</span>
              <span className="hidden sm:inline">•</span>
              <span>CUDA • Metal • CPU</span>
            </div>
          </div>

          {/* Right: visual/story block */}
          <div className="relative max-lg:order-first animate-in fade-in-50 slide-in-from-right-4">
            <div className="rounded-2xl border/60 bg-card/70 p-3 backdrop-blur">
              <Image
                src={useCasesHero}
                width={1080}
                height={760}
                priority
                className="rounded-xl ring-1 ring-border/60"
                alt="cinematic photoreal illustration of intimate homelab desk at night, shot from slightly elevated angle looking down at workspace, FOREGROUND LEFT: black mechanical keyboard with subtle white LED backlighting slightly out of focus creating soft glow, wireless mouse beside it, MIDDLE GROUND CENTER-LEFT: 15-inch MacBook Pro or ThinkPad laptop open showing full-screen terminal window with bright emerald green #10b981 monospace text streaming live AI token generation output 'Generating... token 847/2048' visible, screen has soft blue-white glow illuminating surroundings, small yellow Post-it note stuck to top bezel of laptop screen with handwritten black ink text 'your models your rules' in casual script, MIDDLE GROUND RIGHT: compact desktop mini tower or small rack unit approximately 12 inches tall containing 2-3 NVIDIA RTX 4090 or 3090 graphics cards visible through black mesh front panel with hexagonal perforations, each GPU has warm amber LED strips #f59e0b glowing along the edges creating horizontal light bars, soft orange rim light from GPUs casting warm glow on right side of desk surface and wall behind, faint heat shimmer effect above the GPU unit, DESK SURFACE: dark walnut or black wood finish desk with subtle wood grain texture, clean and minimal with only essential items, soft amber and teal reflections on glossy surface from various light sources, RIGHT EDGE: white ceramic coffee mug with thin wisps of steam rising, small potted succulent plant in concrete pot, BACKGROUND: deep midnight navy blue wall #0f172a with subtle texture, upper left corner has warm brass or copper desk lamp with conical shade creating focused pool of warm yellow light on desk, background fades to soft bokeh with circular light orbs in cool blue and warm amber tones suggesting depth, subtle teal accent light strip along wall edge, LIGHTING: key light from laptop screen (cool blue-white), fill light from desk lamp (warm yellow), accent light from GPUs (warm amber), rim light creating separation from background, overall color temperature is warm with cool accents, professional photography aesthetic similar to Apple product photography or high-end tech reviews, shot with full-frame camera at f/2.8 aperture creating shallow depth of field where laptop and GPU unit are tack sharp while foreground keyboard and background are softly blurred, cinematic color grading with lifted blacks, warm highlights pushed toward amber, cool shadows with slight teal tint, overall mood is cozy yet professional, evokes feelings of calm sovereignty, focused independence, late-night productive coding session, personal control over powerful technology, 3:2 aspect ratio 1080x760 pixels, extremely high detail on GPU hardware showing individual fins and PCB components, terminal text should be legible, steam from coffee should be subtle and realistic"
              />
            </div>

            {/* Caption */}
            <p className="mt-3 text-center text-sm text-muted-foreground">
              Your models, your hardware — no provider lock-in.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
