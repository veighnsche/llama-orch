import { Github } from "lucide-react"
import { Button } from "@/components/ui/button"

export function TechnicalSection() {
  return (
    <section className="py-24 bg-white">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
            Built by Engineers, for Engineers
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
          {/* Architecture Highlights */}
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-slate-900">Architecture Highlights</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-green-600"></div>
                </div>
                <div>
                  <div className="font-medium text-slate-900">BDD-Driven Development</div>
                  <div className="text-sm text-slate-600">42/62 scenarios passing (68% complete)</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-green-600"></div>
                </div>
                <div>
                  <div className="font-medium text-slate-900">Cascading Shutdown Guarantee</div>
                  <div className="text-sm text-slate-600">No orphaned processes, clean VRAM lifecycle</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-green-600"></div>
                </div>
                <div>
                  <div className="font-medium text-slate-900">Process Isolation</div>
                  <div className="text-sm text-slate-600">Clean VRAM lifecycle, no memory leaks</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-green-600"></div>
                </div>
                <div>
                  <div className="font-medium text-slate-900">Protocol-Aware Orchestration</div>
                  <div className="text-sm text-slate-600">SSE, JSON, binary protocols supported</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-green-600"></div>
                </div>
                <div>
                  <div className="font-medium text-slate-900">Smart/Dumb Separation</div>
                  <div className="text-sm text-slate-600">Centralized intelligence, distributed execution</div>
                </div>
              </li>
            </ul>
          </div>

          {/* Technology Stack */}
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-slate-900">Technology Stack</h3>
            <div className="space-y-3">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="font-medium text-slate-900">Rust</div>
                <div className="text-sm text-slate-600">Performance + safety</div>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="font-medium text-slate-900">Candle ML Framework</div>
                <div className="text-sm text-slate-600">Rust-native ML inference</div>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="font-medium text-slate-900">Rhai Scripting</div>
                <div className="text-sm text-slate-600">Embedded, sandboxed scripting</div>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="font-medium text-slate-900">SQLite</div>
                <div className="text-sm text-slate-600">Embedded database</div>
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="font-medium text-slate-900">Axum + Vue.js</div>
                <div className="text-sm text-slate-600">Async web framework + modern UI</div>
              </div>
            </div>

            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-center justify-between">
              <div>
                <div className="font-bold text-amber-900">100% Open Source</div>
                <div className="text-sm text-amber-700">MIT License</div>
              </div>
              <Button variant="outline" size="sm" className="border-amber-300 bg-transparent">
                <Github className="h-4 w-4 mr-2" />
                View Source
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
