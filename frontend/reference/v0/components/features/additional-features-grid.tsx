import { Shield, Database, Network, Terminal, Code } from "lucide-react"

export function AdditionalFeaturesGrid() {
  return (
    <section className="py-24 bg-white">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
            Everything You Need for AI Infrastructure
          </h2>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center">
              <Shield className="h-6 w-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">Cascading Shutdown</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Press Ctrl+C once. Everything shuts down cleanly. No orphaned processes. No leaked VRAM. Reliable cleanup
              guaranteed.
            </p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center">
              <Database className="h-6 w-6 text-green-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">Model Catalog</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Automatic model provisioning from Hugging Face. Support for GGUF models. Llama, Mistral, Qwen, DeepSeek,
              and more.
            </p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-amber-100 flex items-center justify-center">
              <Network className="h-6 w-6 text-amber-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">Network Orchestration</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Orchestrate across your entire home network. Gaming PCs, workstations, Macs—all working together as one AI
              cluster.
            </p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-slate-100 flex items-center justify-center">
              <Terminal className="h-6 w-6 text-slate-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">CLI & Web UI</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Powerful CLI for automation and scripting. Beautiful Web UI for visual management. Choose your preferred
              workflow.
            </p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-amber-100 flex items-center justify-center">
              <Code className="h-6 w-6 text-amber-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">TypeScript SDK</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              llama-orch-utils provides a TypeScript library for building AI agents. Type-safe, async/await, full IDE
              support.
            </p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center">
              <Shield className="h-6 w-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">Security First</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Five specialized security crates. Defense-in-depth architecture. Timing-safe authentication. Immutable
              audit logs.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
