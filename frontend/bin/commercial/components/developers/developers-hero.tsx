import { Button } from "@/components/ui/button"
import { ArrowRight, Github } from "lucide-react"

export function DevelopersHero() {
  return (
    <section className="relative overflow-hidden border-b border-border bg-background">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/20 via-background to-background" />

      <div className="relative mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-primary"></span>
            </span>
            For Developers Who Build with AI
          </div>

          <h1 className="mb-6 text-balance text-5xl font-bold tracking-tight text-foreground sm:text-6xl lg:text-7xl">
            Build with AI.
            <br />
            <span className="bg-gradient-to-r from-primary to-primary bg-clip-text text-transparent">
              Own Your Infrastructure.
            </span>
          </h1>

          <p className="mb-8 text-balance text-xl leading-relaxed text-muted-foreground">
            Stop depending on AI providers. Use rbee to orchestrate inference across{" "}
            <span className="font-semibold text-foreground">ALL your home network hardware</span>—GPUs, Macs,
            workstations—with <span className="font-semibold text-foreground">zero ongoing costs</span>.
          </p>

          <div className="mb-12 flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Button size="lg" className="group bg-primary text-primary-foreground hover:bg-primary/90">
              Get Started Free
              <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-border text-foreground hover:bg-secondary bg-transparent"
            >
              <Github className="mr-2 h-4 w-4" />
              View on GitHub
            </Button>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5 text-primary" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              100% Open Source
            </div>
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5 text-primary" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              OpenAI-Compatible API
            </div>
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5 text-primary" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              Works with Zed & Cursor
            </div>
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5 text-primary" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              No Cloud Required
            </div>
          </div>
        </div>

        {/* Animated Terminal */}
        <div className="mx-auto mt-16 max-w-4xl">
          <div className="overflow-hidden rounded-lg border border-border bg-card shadow-2xl">
            <div className="flex items-center gap-2 border-b border-border bg-muted px-4 py-3">
              <div className="h-3 w-3 rounded-full bg-destructive" />
              <div className="h-3 w-3 rounded-full bg-primary" />
              <div className="h-3 w-3 rounded-full bg-chart-3" />
              <span className="ml-2 text-sm text-muted-foreground">terminal</span>
            </div>
            <div className="p-6 font-mono text-sm">
              <div className="mb-2 text-muted-foreground">
                <span className="text-chart-3">$</span> rbee-keeper infer --model llama-3.1-70b --prompt
                &quot;Generate API&quot;
              </div>
              <div className="mb-4 text-muted-foreground">
                <span className="animate-pulse">▊</span> Streaming tokens...
              </div>
              <div className="space-y-1 text-foreground">
                <div className="animate-[fadeIn_0.5s_ease-in]">
                  <span className="text-chart-2">export</span> <span className="text-primary">async</span>{" "}
                  <span className="text-chart-4">function</span> <span className="text-chart-3">getUsers</span>
                  () {"{"}
                </div>
                <div className="animate-[fadeIn_0.7s_ease-in] pl-4">
                  <span className="text-chart-2">const</span> response = <span className="text-chart-2">await</span>{" "}
                  <span className="text-chart-3">fetch</span>(
                  <span className="text-primary">&apos;/api/users&apos;</span>)
                </div>
                <div className="animate-[fadeIn_0.9s_ease-in] pl-4">
                  <span className="text-chart-2">return</span> response.<span className="text-chart-3">json</span>()
                </div>
                <div className="animate-[fadeIn_1.1s_ease-in]">{"}"}</div>
              </div>
              <div className="mt-4 flex items-center gap-4 text-muted-foreground">
                <div>GPU 1: 87%</div>
                <div>GPU 2: 92%</div>
                <div>Cost: $0.00</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
