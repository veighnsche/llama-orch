import { useCasesHero } from "@rbee/ui/assets";
import { Button } from "@rbee/ui/atoms/Button";
import Image from "next/image";

export function UseCasesHero() {
  return (
    <section className="relative overflow-hidden py-20 lg:py-24 bg-gradient-to-b from-background to-card">
      {/* Soft radial glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
      >
        <div className="absolute top-0 right-1/4 h-[40rem] w-[40rem] rounded-full bg-primary/10 blur-3xl" />
      </div>

      <div className="container mx-auto px-4">
        <div className="grid gap-12 lg:grid-cols-[6fr_5fr] lg:items-center">
          {/* Left: copy stack */}
          <div className="animate-in fade-in-50 slide-in-from-left-4">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border bg-card/50 px-4 py-1.5 text-sm text-muted-foreground">
              <span className="font-sans font-medium text-foreground">OpenAI-compatible</span>
            </div>

            <h1 className="text-balance text-5xl lg:text-6xl xl:text-7xl font-bold text-foreground tracking-tight leading-[1.1]">
              Built for Those Who Value{" "}
              <span className="bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                Independence
              </span>
            </h1>

            <p className="mt-6 text-lg lg:text-xl text-muted-foreground leading-relaxed max-w-xl">
              Own your AI infrastructure. From solo developers to enterprises,
              rbee adapts to your needs without compromising power or flexibility.
            </p>

            {/* Two clear CTAs */}
            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <Button
                className="h-12 px-8 text-base"
                asChild
              >
                <a href="#use-cases">Explore use cases</a>
              </Button>
              <Button
                variant="outline"
                className="h-12 px-8 text-base"
                asChild
              >
                <a href="#architecture">See architecture</a>
              </Button>
            </div>

            {/* Proof indicators */}
            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" aria-hidden />
                Self-hosted
              </span>
              <span>OpenAI-compatible</span>
              <span>CUDA · Metal · CPU</span>
            </div>
          </div>

          {/* Right: visual */}
          <div className="relative max-lg:order-first animate-in fade-in-50 slide-in-from-right-4">
            <div className="rounded-2xl border bg-card/50 p-4 backdrop-blur-sm">
              <Image
                src={useCasesHero}
                width={1080}
                height={760}
                priority
                className="rounded-xl"
                alt="cinematic photoreal illustration of intimate homelab desk at night, shot from slightly elevated angle looking down at workspace, FOREGROUND LEFT: black mechanical keyboard with subtle white LED backlighting slightly out of focus creating soft glow, wireless mouse beside it, MIDDLE GROUND CENTER-LEFT: 15-inch MacBook Pro or ThinkPad laptop open showing full-screen terminal window with bright emerald green #10b981 monospace text streaming live AI token generation output 'Generating... token 847/2048' visible, screen has soft blue-white glow illuminating surroundings, small yellow Post-it note stuck to top bezel of laptop screen with handwritten black ink text 'your models your rules' in casual script, MIDDLE GROUND RIGHT: compact desktop mini tower or small rack unit approximately 12 inches tall containing 2-3 NVIDIA RTX 4090 or 3090 graphics cards visible through black mesh front panel with hexagonal perforations, each GPU has warm amber LED strips #f59e0b glowing along the edges creating horizontal light bars, soft orange rim light from GPUs casting warm glow on right side of desk surface and wall behind, faint heat shimmer effect above the GPU unit, DESK SURFACE: dark walnut or black wood finish desk with subtle wood grain texture, clean and minimal with only essential items, soft amber and teal reflections on glossy surface from various light sources, RIGHT EDGE: white ceramic coffee mug with thin wisps of steam rising, small potted succulent plant in concrete pot, BACKGROUND: deep midnight navy blue wall #0f172a with subtle texture, upper left corner has warm brass or copper desk lamp with conical shade creating focused pool of warm yellow light on desk, background fades to soft bokeh with circular light orbs in cool blue and warm amber tones suggesting depth, subtle teal accent light strip along wall edge, LIGHTING: key light from laptop screen (cool blue-white), fill light from desk lamp (warm yellow), accent light from GPUs (warm amber), rim light creating separation from background, overall color temperature is warm with cool accents, professional photography aesthetic similar to Apple product photography or high-end tech reviews, shot with full-frame camera at f/2.8 aperture creating shallow depth of field where laptop and GPU unit are tack sharp while foreground keyboard and background are softly blurred, cinematic color grading with lifted blacks, warm highlights pushed toward amber, cool shadows with slight teal tint, overall mood is cozy yet professional, evokes feelings of calm sovereignty, focused independence, late-night productive coding session, personal control over powerful technology, 3:2 aspect ratio 1080x760 pixels, extremely high detail on GPU hardware showing individual fins and PCB components, terminal text should be legible, steam from coffee should be subtle and realistic"
              />
            </div>

            <p className="mt-4 text-center text-sm text-muted-foreground">
              Your models, your hardware — no lock-in.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
