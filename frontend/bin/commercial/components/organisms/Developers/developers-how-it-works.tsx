import { ConsoleOutput } from "@/components/atoms"

export function DevelopersHowItWorks() {
  return (
    <section className="border-b border-border bg-secondary py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            From Zero to AI Infrastructure in 15 Minutes
          </h2>
        </div>

        <div className="mx-auto mt-16 max-w-4xl space-y-12">
          {/* Step 1 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-primary text-xl font-bold text-primary-foreground">
              1
            </div>
            <div className="flex-1">
              <h3 className="mb-3 text-xl font-semibold text-card-foreground">Install rbee</h3>
              <ConsoleOutput showChrome title="terminal" background="dark">
                <div>curl -sSL https://rbee.dev/install.sh | sh</div>
                <div className="text-slate-400">rbee-keeper daemon start</div>
              </ConsoleOutput>
            </div>
          </div>

          {/* Step 2 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-primary text-xl font-bold text-primary-foreground">
              2
            </div>
            <div className="flex-1">
              <h3 className="mb-3 text-xl font-semibold text-card-foreground">Add Your Machines</h3>
              <ConsoleOutput showChrome title="terminal" background="dark">
                <div>rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10</div>
                <div className="text-slate-400">rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20</div>
              </ConsoleOutput>
            </div>
          </div>

          {/* Step 3 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-primary text-xl font-bold text-primary-foreground">
              3
            </div>
            <div className="flex-1">
              <h3 className="mb-3 text-xl font-semibold text-card-foreground">Configure Your IDE</h3>
              <ConsoleOutput showChrome title="terminal" background="dark">
                <div>
                  <span className="text-blue-400">export</span> OPENAI_API_BASE=http://localhost:8080/v1
                </div>
                <div className="text-slate-400"># Now Zed, Cursor, or any OpenAI-compatible tool works!</div>
              </ConsoleOutput>
            </div>
          </div>

          {/* Step 4 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-primary text-xl font-bold text-primary-foreground">
              4
            </div>
            <div className="flex-1">
              <h3 className="mb-3 text-xl font-semibold text-card-foreground">Build AI Agents</h3>
              <ConsoleOutput showChrome title="TypeScript" background="dark" variant="code">
                <div>
                  <span className="text-purple-400">import</span> {"{"} invoke {"}"}{" "}
                  <span className="text-purple-400">from</span>{" "}
                  <span className="text-amber-400">&apos;@llama-orch/utils&apos;</span>;
                </div>
                <div className="mt-2">
                  <span className="text-blue-400">const</span> code = <span className="text-blue-400">await</span>{" "}
                  <span className="text-green-400">invoke</span>({"{"}
                </div>
                <div className="pl-4">
                  prompt: <span className="text-amber-400">&apos;Generate API from schema&apos;</span>,
                </div>
                <div className="pl-4">
                  model: <span className="text-amber-400">&apos;llama-3.1-70b&apos;</span>
                </div>
                <div>{"});"}</div>
              </ConsoleOutput>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
