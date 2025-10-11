import { Network, GitBranch } from "lucide-react"

export function CrossNodeOrchestration() {
  return (
    <section className="py-24 bg-slate-900">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6 text-balance">Cross-Node Orchestration</h2>
          <p className="text-xl text-slate-300 leading-relaxed">
            Seamlessly orchestrate AI workloads across your entire network. One command runs inference on any machine in
            your homelab.
          </p>
        </div>

        <div className="max-w-5xl mx-auto space-y-8">
          {/* SSH Registry Management */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-amber-500/10 flex items-center justify-center flex-shrink-0">
                <Network className="h-6 w-6 text-amber-500" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">SSH Registry Management</h3>
                <p className="text-slate-300 leading-relaxed">
                  Configure remote nodes once, use them forever. rbee-keeper manages SSH connections, validates
                  connectivity, and maintains a registry of all available nodes.
                </p>
              </div>
            </div>

            <div className="bg-slate-950 rounded-lg p-6 font-mono text-sm space-y-2">
              <div className="text-slate-400"># Add a remote node to your cluster</div>
              <div className="text-green-400 mt-2">
                rbee-keeper setup add-node \
                <br />
                {"  "}--name workstation \
                <br />
                {"  "}--ssh-host workstation.home.arpa \
                <br />
                {"  "}--ssh-user vince \
                <br />
                {"  "}--ssh-key ~/.ssh/id_ed25519
              </div>
              <div className="text-slate-400 mt-4"># Run inference on that node</div>
              <div className="text-green-400 mt-2">
                rbee-keeper infer --node workstation \
                <br />
                {"  "}--model hf:meta-llama/Llama-3.1-8B \
                <br />
                {"  "}--prompt "write a short story"
              </div>
            </div>

            <div className="mt-6 grid md:grid-cols-3 gap-4">
              <div className="bg-slate-950 rounded-lg p-4">
                <div className="text-amber-500 font-bold mb-1">Automatic Detection</div>
                <div className="text-slate-400 text-sm">Detects CUDA, Metal, CPU backends and device counts</div>
              </div>
              <div className="bg-slate-950 rounded-lg p-4">
                <div className="text-amber-500 font-bold mb-1">SSH Validation</div>
                <div className="text-slate-400 text-sm">Tests connectivity before saving to registry</div>
              </div>
              <div className="bg-slate-950 rounded-lg p-4">
                <div className="text-amber-500 font-bold mb-1">Zero Config</div>
                <div className="text-slate-400 text-sm">No manual setup on remote nodes required</div>
              </div>
            </div>
          </div>

          {/* Automatic Worker Provisioning */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
                <GitBranch className="h-6 w-6 text-blue-500" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">Automatic Worker Provisioning</h3>
                <p className="text-slate-300 leading-relaxed">
                  rbee automatically spawns workers on remote nodes via SSH. No manual daemon management. Workers start
                  on-demand and shut down cleanly.
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="flex-shrink-0 w-40 text-sm text-slate-400">queen-rbee</div>
                <div className="flex-1 text-slate-300 text-sm">Orchestrator (runs on your main machine)</div>
              </div>
              <div className="flex items-center gap-3 pl-8">
                <div className="flex-shrink-0 w-32 text-sm text-slate-400">↓ SSH</div>
                <div className="flex-1 text-slate-500 text-sm">Connects to remote nodes</div>
              </div>
              <div className="flex items-center gap-3 pl-8">
                <div className="flex-shrink-0 w-40 text-sm text-slate-400">rbee-hive</div>
                <div className="flex-1 text-slate-300 text-sm">Pool manager (spawned on remote node)</div>
              </div>
              <div className="flex items-center gap-3 pl-16">
                <div className="flex-shrink-0 w-32 text-sm text-slate-400">↓ Spawns</div>
                <div className="flex-1 text-slate-500 text-sm">Creates workers as needed</div>
              </div>
              <div className="flex items-center gap-3 pl-16">
                <div className="flex-shrink-0 w-40 text-sm text-slate-400">worker-rbee</div>
                <div className="flex-1 text-slate-300 text-sm">Worker daemon (handles inference)</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
