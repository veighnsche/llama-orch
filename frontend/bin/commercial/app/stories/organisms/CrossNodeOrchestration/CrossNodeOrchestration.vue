<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented CrossNodeOrchestration -->
<script setup lang="ts">
import { Network, GitBranch } from 'lucide-vue-next'
</script>

<template>
  <section class="py-24 bg-background">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">Cross-Node Orchestration</h2>
        <p class="text-xl text-muted-foreground leading-relaxed">
          Seamlessly orchestrate AI workloads across your entire network. One command runs inference on any machine in
          your homelab.
        </p>
      </div>

      <div class="max-w-5xl mx-auto space-y-8">
        <!-- SSH Registry Management -->
        <div class="bg-card border border-border rounded-lg p-8">
          <div class="flex items-start gap-4 mb-6">
            <div class="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
              <Network class="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 class="text-2xl font-bold text-foreground mb-2">SSH Registry Management</h3>
              <p class="text-muted-foreground leading-relaxed">
                Configure remote nodes once, use them forever. rbee-keeper manages SSH connections, validates
                connectivity, and maintains a registry of all available nodes.
              </p>
            </div>
          </div>

          <div class="bg-secondary rounded-lg p-6 font-mono text-sm space-y-2">
            <div class="text-muted-foreground"># Add a remote node to your cluster</div>
            <div class="text-accent mt-2">
              rbee-keeper setup add-node \
              <br />
              &nbsp;&nbsp;--name workstation \
              <br />
              &nbsp;&nbsp;--ssh-host workstation.home.arpa \
              <br />
              &nbsp;&nbsp;--ssh-user vince \
              <br />
              &nbsp;&nbsp;--ssh-key ~/.ssh/id_ed25519
            </div>
            <div class="text-muted-foreground mt-4"># Run inference on that node</div>
            <div class="text-accent mt-2">
              rbee-keeper infer --node workstation \
              <br />
              &nbsp;&nbsp;--model hf:meta-llama/Llama-3.1-8B \
              <br />
              &nbsp;&nbsp;--prompt "write a short story"
            </div>
          </div>

          <div class="mt-6 grid md:grid-cols-3 gap-4">
            <div class="bg-secondary rounded-lg p-4">
              <div class="text-primary font-bold mb-1">Automatic Detection</div>
              <div class="text-muted-foreground text-sm">Detects CUDA, Metal, CPU backends and device counts</div>
            </div>
            <div class="bg-secondary rounded-lg p-4">
              <div class="text-primary font-bold mb-1">SSH Validation</div>
              <div class="text-muted-foreground text-sm">Tests connectivity before saving to registry</div>
            </div>
            <div class="bg-secondary rounded-lg p-4">
              <div class="text-primary font-bold mb-1">Zero Config</div>
              <div class="text-muted-foreground text-sm">No manual setup on remote nodes required</div>
            </div>
          </div>
        </div>

        <!-- Automatic Worker Provisioning -->
        <div class="bg-card border border-border rounded-lg p-8">
          <div class="flex items-start gap-4 mb-6">
            <div class="h-12 w-12 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
              <GitBranch class="h-6 w-6 text-blue-500" />
            </div>
            <div>
              <h3 class="text-2xl font-bold text-foreground mb-2">Automatic Worker Provisioning</h3>
              <p class="text-muted-foreground leading-relaxed">
                rbee automatically spawns workers on remote nodes via SSH. No manual daemon management. Workers start
                on-demand and shut down cleanly.
              </p>
            </div>
          </div>

          <div class="space-y-3">
            <div class="flex items-center gap-3">
              <div class="flex-shrink-0 w-40 text-sm text-muted-foreground">queen-rbee</div>
              <div class="flex-1 text-foreground text-sm">Orchestrator (runs on your main machine)</div>
            </div>
            <div class="flex items-center gap-3 pl-8">
              <div class="flex-shrink-0 w-32 text-sm text-muted-foreground">↓ SSH</div>
              <div class="flex-1 text-muted-foreground text-sm">Connects to remote nodes</div>
            </div>
            <div class="flex items-center gap-3 pl-8">
              <div class="flex-shrink-0 w-40 text-sm text-muted-foreground">rbee-hive</div>
              <div class="flex-1 text-foreground text-sm">Pool manager (spawned on remote node)</div>
            </div>
            <div class="flex items-center gap-3 pl-16">
              <div class="flex-shrink-0 w-32 text-sm text-muted-foreground">↓ Spawns</div>
              <div class="flex-1 text-muted-foreground text-sm">Creates workers as needed</div>
            </div>
            <div class="flex items-center gap-3 pl-16">
              <div class="flex-shrink-0 w-40 text-sm text-muted-foreground">worker-rbee</div>
              <div class="flex-1 text-foreground text-sm">Worker daemon (handles inference)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
