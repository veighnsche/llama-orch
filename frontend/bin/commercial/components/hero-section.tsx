import { Button } from "@/components/ui/button"
import { ArrowRight, Github, Star } from "lucide-react"

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center bg-background">
      <div className="container mx-auto px-4 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Messaging */}
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-primary text-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
              </span>
              100% Open Source • MIT License
            </div>

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
              <div className="flex items-center gap-2 text-muted-foreground">
                <Github className="h-5 w-5" />
                <span className="text-sm">Open Source</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <Star className="h-5 w-5 fill-primary text-primary" />
                <span className="text-sm">1,200+ Stars</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="h-5 w-5 flex items-center justify-center text-xs font-bold border border-border rounded">
                  API
                </div>
                <span className="text-sm">OpenAI-Compatible</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="h-5 w-5 flex items-center justify-center text-xs font-bold border border-border rounded">
                  $0
                </div>
                <span className="text-sm">No Cloud Required</span>
              </div>
            </div>
          </div>

          {/* Right: Terminal Visual */}
          <div className="relative">
            <div className="bg-card border border-border rounded-lg overflow-hidden shadow-2xl">
              <div className="flex items-center gap-2 px-4 py-3 bg-muted border-b border-border">
                <div className="flex gap-2">
                  <div className="h-3 w-3 rounded-full bg-red-500"></div>
                  <div className="h-3 w-3 rounded-full bg-amber-500"></div>
                  <div className="h-3 w-3 rounded-full bg-green-500"></div>
                </div>
                <span className="text-muted-foreground text-sm ml-2 font-mono">rbee-keeper</span>
              </div>
              <div className="p-6 font-mono text-sm space-y-3">
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
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-24">workstation</span>
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-primary w-[85%]"></div>
                      </div>
                      <span className="text-muted-foreground text-xs">85%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-24">mac-studio</span>
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-primary w-[72%]"></div>
                      </div>
                      <span className="text-muted-foreground text-xs">72%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-24">gaming-pc</span>
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-primary w-[91%]"></div>
                      </div>
                      <span className="text-muted-foreground text-xs">91%</span>
                    </div>
                  </div>
                </div>

                {/* Cost Counter */}
                <div className="pt-2 flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Cost:</span>
                  <span className="text-chart-3 font-bold">$0.00</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
