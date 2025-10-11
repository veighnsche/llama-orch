import { Button } from "@/components/ui/button"
import { ArrowRight, Github, Star } from "lucide-react"

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Messaging */}
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-amber-500/10 border border-amber-500/20 rounded-full text-amber-400 text-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
              </span>
              100% Open Source • MIT License
            </div>

            <h1 className="text-5xl lg:text-7xl font-bold text-white leading-tight text-balance">
              AI Infrastructure.
              <br />
              <span className="text-amber-400">On Your Terms.</span>
            </h1>

            <p className="text-xl text-slate-300 leading-relaxed text-pretty">
              Orchestrate AI inference across any hardware—your GPUs, your network, your rules. Build with AI, monetize
              idle hardware, or ensure compliance. Zero vendor lock-in.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                size="lg"
                className="bg-amber-500 hover:bg-amber-600 text-slate-950 font-semibold text-lg h-14 px-8"
              >
                Get Started Free
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-slate-600 text-slate-200 hover:bg-slate-800 h-14 px-8 bg-transparent"
              >
                View Documentation
              </Button>
            </div>

            {/* Trust Indicators */}
            <div className="flex flex-wrap gap-6 pt-4">
              <div className="flex items-center gap-2 text-slate-400">
                <Github className="h-5 w-5" />
                <span className="text-sm">Open Source</span>
              </div>
              <div className="flex items-center gap-2 text-slate-400">
                <Star className="h-5 w-5 fill-amber-500 text-amber-500" />
                <span className="text-sm">1,200+ Stars</span>
              </div>
              <div className="flex items-center gap-2 text-slate-400">
                <div className="h-5 w-5 flex items-center justify-center text-xs font-bold border border-slate-600 rounded">
                  API
                </div>
                <span className="text-sm">OpenAI-Compatible</span>
              </div>
              <div className="flex items-center gap-2 text-slate-400">
                <div className="h-5 w-5 flex items-center justify-center text-xs font-bold border border-slate-600 rounded">
                  $0
                </div>
                <span className="text-sm">No Cloud Required</span>
              </div>
            </div>
          </div>

          {/* Right: Terminal Visual */}
          <div className="relative">
            <div className="bg-slate-900 border border-slate-700 rounded-lg overflow-hidden shadow-2xl">
              <div className="flex items-center gap-2 px-4 py-3 bg-slate-800 border-b border-slate-700">
                <div className="flex gap-2">
                  <div className="h-3 w-3 rounded-full bg-red-500"></div>
                  <div className="h-3 w-3 rounded-full bg-amber-500"></div>
                  <div className="h-3 w-3 rounded-full bg-green-500"></div>
                </div>
                <span className="text-slate-400 text-sm ml-2 font-mono">rbee-keeper</span>
              </div>
              <div className="p-6 font-mono text-sm space-y-3">
                <div className="text-slate-400">
                  <span className="text-green-400">$</span> rbee-keeper infer --model llama-3.1-70b
                </div>
                <div className="text-slate-300 pl-4">
                  <span className="text-amber-400">→</span> Loading model across 3 GPUs...
                </div>
                <div className="text-slate-300 pl-4">
                  <span className="text-green-400">✓</span> Model ready (2.3s)
                </div>
                <div className="text-slate-400 pl-4">
                  <span className="text-blue-400">Prompt:</span> Generate REST API for user management
                </div>
                <div className="text-slate-300 pl-4 leading-relaxed">
                  <span className="text-amber-400 animate-pulse">▊</span> Generating code...
                </div>

                {/* GPU Utilization */}
                <div className="pt-4 space-y-2">
                  <div className="text-slate-500 text-xs">GPU Utilization:</div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-400 text-xs w-24">workstation</span>
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-amber-500 w-[85%]"></div>
                      </div>
                      <span className="text-slate-400 text-xs">85%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-400 text-xs w-24">mac-studio</span>
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-amber-500 w-[72%]"></div>
                      </div>
                      <span className="text-slate-400 text-xs">72%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-400 text-xs w-24">gaming-pc</span>
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-amber-500 w-[91%]"></div>
                      </div>
                      <span className="text-slate-400 text-xs">91%</span>
                    </div>
                  </div>
                </div>

                {/* Cost Counter */}
                <div className="pt-2 flex items-center justify-between text-xs">
                  <span className="text-slate-500">Cost:</span>
                  <span className="text-green-400 font-bold">$0.00</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
