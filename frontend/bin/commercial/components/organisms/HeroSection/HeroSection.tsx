import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight, Github, Star } from "lucide-react"
import { PulseBadge, TrustIndicator, TerminalWindow, ProgressBar } from '@/components/molecules'

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center bg-background">
      <div className="container mx-auto px-4 py-24">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Messaging */}
          <div className="space-y-8">
            <PulseBadge text="100% Open Source • GPL-3.0-or-later" />

            <h1 className="text-5xl lg:text-7xl font-bold text-foreground leading-tight text-balance">
              AI Infrastructure.
              <br />
              <span className="text-primary">On Your Terms.</span>
            </h1>

            <p className="text-xl text-muted-foreground leading-relaxed text-pretty">
              Orchestrate AI inference across any hardware—your GPUs, your network, your rules. Build with AI, monetize
              idle hardware, or ensure compliance. Zero vendor lock-in.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                size="lg"
                className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8"
              >
                Get Started Free
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent"
              >
                View Documentation
              </Button>
            </div>

            {/* Trust Indicators */}
            <div className="flex flex-wrap gap-6 pt-4">
              <TrustIndicator icon={Github} text="Open Source" />
              <TrustIndicator icon={Star} text="On GitHub" variant="primary" />
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="h-5 w-5 flex items-center justify-center text-xs font-bold border border-border rounded-sm">
                  API
                </div>
                <span className="text-sm">OpenAI-Compatible</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="h-5 w-5 flex items-center justify-center text-xs font-bold border border-border rounded-sm">
                  $0
                </div>
                <span className="text-sm">No Cloud Required</span>
              </div>
            </div>
          </div>

          {/* Right: Terminal Visual */}
          <div className="relative">
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
                  <span className="text-chart-2">Prompt:</span> Generate REST API for user management
                </div>
                <div className="text-foreground pl-4 leading-relaxed">
                  <span className="text-primary animate-pulse">▊</span> Generating code...
                </div>

                {/* GPU Utilization */}
                <div className="pt-4 space-y-2">
                  <div className="text-muted-foreground text-xs">GPU Utilization:</div>
                  <div className="space-y-1">
                    <ProgressBar label="workstation" percentage={85} />
                    <ProgressBar label="mac-studio" percentage={72} />
                    <ProgressBar label="gaming-pc" percentage={91} />
                  </div>
                </div>

                {/* Cost Counter */}
                <div className="pt-2 flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Cost:</span>
                  <span className="text-chart-3 font-bold">$0.00</span>
                </div>
              </div>
            </TerminalWindow>
          </div>
        </div>
      </div>
    </section>
  )
}
