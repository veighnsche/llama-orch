export function DevelopersHowItWorks() {
  return (
    <section className="border-b border-slate-800 bg-slate-900/30 py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            From Zero to AI Infrastructure in 15 Minutes
          </h2>
        </div>

        <div className="mx-auto mt-16 max-w-4xl space-y-12">
          {/* Step 1 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-amber-500 text-xl font-bold text-slate-950">
              1
            </div>
            <div className="flex-1">
              <h3 className="mb-3 text-xl font-semibold text-white">Install rbee</h3>
              <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900">
                <div className="border-b border-slate-800 bg-slate-800/50 px-4 py-2">
                  <span className="text-sm text-slate-400">terminal</span>
                </div>
                <div className="p-4 font-mono text-sm text-slate-300">
                  <div>curl -sSL https://rbee.dev/install.sh | sh</div>
                  <div className="text-slate-500">rbee-keeper daemon start</div>
                </div>
              </div>
            </div>
          </div>

          {/* Step 2 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-amber-500 text-xl font-bold text-slate-950">
              2
            </div>
            <div className="flex-1">
              <h3 className="mb-3 text-xl font-semibold text-white">Add Your Machines</h3>
              <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900">
                <div className="border-b border-slate-800 bg-slate-800/50 px-4 py-2">
                  <span className="text-sm text-slate-400">terminal</span>
                </div>
                <div className="p-4 font-mono text-sm text-slate-300">
                  <div>rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10</div>
                  <div className="text-slate-500">rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20</div>
                </div>
              </div>
            </div>
          </div>

          {/* Step 3 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-amber-500 text-xl font-bold text-slate-950">
              3
            </div>
            <div className="flex-1">
              <h3 className="mb-3 text-xl font-semibold text-white">Configure Your IDE</h3>
              <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900">
                <div className="border-b border-slate-800 bg-slate-800/50 px-4 py-2">
                  <span className="text-sm text-slate-400">terminal</span>
                </div>
                <div className="p-4 font-mono text-sm text-slate-300">
                  <div>
                    <span className="text-blue-400">export</span> OPENAI_API_BASE=http://localhost:8080/v1
                  </div>
                  <div className="text-slate-500"># Now Zed, Cursor, or any OpenAI-compatible tool works!</div>
                </div>
              </div>
            </div>
          </div>

          {/* Step 4 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-amber-500 text-xl font-bold text-slate-950">
              4
            </div>
            <div className="flex-1">
              <h3 className="mb-3 text-xl font-semibold text-white">Build AI Agents</h3>
              <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900">
                <div className="border-b border-slate-800 bg-slate-800/50 px-4 py-2">
                  <span className="text-sm text-slate-400">TypeScript</span>
                </div>
                <div className="p-4 font-mono text-sm text-slate-300">
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
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
