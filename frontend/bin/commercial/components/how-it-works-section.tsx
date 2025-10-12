export function HowItWorksSection() {
  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
            From Zero to AI Infrastructure in 15 Minutes
          </h2>
        </div>

        <div className="max-w-5xl mx-auto space-y-16">
          {/* Step 1 */}
          <div className="grid lg:grid-cols-2 gap-8 items-center">
            <div className="space-y-4">
              <div className="inline-flex items-center justify-center h-12 w-12 rounded-full bg-primary text-primary-foreground font-bold text-xl">
                1
              </div>
              <h3 className="text-2xl font-bold text-foreground">Install rbee</h3>
              <p className="text-muted-foreground leading-relaxed">
                One command to install rbee on your machine. Works on Linux, macOS, and Windows.
              </p>
            </div>
            <div className="bg-card border border-border rounded-lg p-6 font-mono text-sm">
              <div className="text-chart-3">$ curl -sSL https://rbee.dev/install.sh | sh</div>
              <div className="text-muted-foreground mt-2">$ rbee-keeper daemon start</div>
              <div className="text-foreground mt-2 pl-4">✓ rbee daemon started on port 8080</div>
            </div>
          </div>

          {/* Step 2 */}
          <div className="grid lg:grid-cols-2 gap-8 items-center">
            <div className="space-y-4 lg:order-2">
              <div className="inline-flex items-center justify-center h-12 w-12 rounded-full bg-primary text-primary-foreground font-bold text-xl">
                2
              </div>
              <h3 className="text-2xl font-bold text-foreground">Add Your Machines</h3>
              <p className="text-muted-foreground leading-relaxed">
                Connect all your GPUs across your network. rbee automatically detects CUDA, Metal, and CPU backends.
              </p>
            </div>
            <div className="bg-card border border-border rounded-lg p-6 font-mono text-sm lg:order-1">
              <div className="text-chart-3">$ rbee-keeper setup add-node \</div>
              <div className="text-muted-foreground pl-4">--name workstation \</div>
              <div className="text-muted-foreground pl-4">--ssh-host 192.168.1.10</div>
              <div className="text-foreground mt-2 pl-4">✓ Added node: workstation (2x RTX 4090)</div>
              <div className="text-chart-3 mt-3">$ rbee-keeper setup add-node \</div>
              <div className="text-muted-foreground pl-4">--name mac \</div>
              <div className="text-muted-foreground pl-4">--ssh-host 192.168.1.20</div>
              <div className="text-foreground mt-2 pl-4">✓ Added node: mac (M2 Ultra)</div>
            </div>
          </div>

          {/* Step 3 */}
          <div className="grid lg:grid-cols-2 gap-8 items-center">
            <div className="space-y-4">
              <div className="inline-flex items-center justify-center h-12 w-12 rounded-full bg-primary text-primary-foreground font-bold text-xl">
                3
              </div>
              <h3 className="text-2xl font-bold text-foreground">Start Inference</h3>
              <p className="text-muted-foreground leading-relaxed">
                Point your tools to localhost. Zed, Cursor, or any OpenAI-compatible tool works instantly.
              </p>
            </div>
            <div className="bg-card border border-border rounded-lg p-6 font-mono text-sm">
              <div className="text-chart-3">$ export OPENAI_API_BASE=http://localhost:8080/v1</div>
              <div className="text-muted-foreground mt-3"># Now Zed, Cursor, or any OpenAI-compatible</div>
              <div className="text-muted-foreground"># tool works with your local infrastructure!</div>
            </div>
          </div>

          {/* Step 4 */}
          <div className="grid lg:grid-cols-2 gap-8 items-center">
            <div className="space-y-4 lg:order-2">
              <div className="inline-flex items-center justify-center h-12 w-12 rounded-full bg-primary text-primary-foreground font-bold text-xl">
                4
              </div>
              <h3 className="text-2xl font-bold text-foreground">Build AI Agents</h3>
              <p className="text-muted-foreground leading-relaxed">
                Use the TypeScript SDK to build custom AI agents, tools, and workflows.
              </p>
            </div>
            <div className="bg-card border border-border rounded-lg p-6 font-mono text-sm lg:order-1">
              <div className="text-chart-4">import</div>
              <div className="text-foreground">{" { invoke } "}</div>
              <div className="text-chart-4">from</div>
              <div className="text-chart-3">{" '@llama-orch/utils'"}</div>
              <div className="text-muted-foreground mt-3"></div>
              <div className="text-chart-4">const</div>
              <div className="text-foreground">{" code = "}</div>
              <div className="text-chart-4">await</div>
              <div className="text-chart-2">{" invoke"}</div>
              <div className="text-foreground">{"({"}</div>
              <div className="text-foreground pl-4">
                prompt: <span className="text-chart-3">'Generate API'</span>,
              </div>
              <div className="text-foreground pl-4">
                model: <span className="text-chart-3">'llama-3.1-70b'</span>
              </div>
              <div className="text-foreground">{"})"}</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
