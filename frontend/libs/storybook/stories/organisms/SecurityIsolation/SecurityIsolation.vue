<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented SecurityIsolation -->
<script setup lang="ts">
import { Shield, Lock, CheckCircle2 } from 'lucide-vue-next'
</script>

<template>
  <section class="py-24 bg-background">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">Security & Isolation</h2>
        <p class="text-xl text-muted-foreground leading-relaxed">
          Defense-in-depth architecture with five specialized security crates. Enterprise-grade security for your
          homelab.
        </p>
      </div>

      <div class="max-w-5xl mx-auto space-y-8">
        <!-- Five Security Crates -->
        <div class="bg-card border border-border rounded-lg p-8">
          <div class="flex items-start gap-4 mb-6">
            <div class="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0">
              <Shield class="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h3 class="text-2xl font-bold text-foreground mb-2">Five Specialized Security Crates</h3>
              <p class="text-muted-foreground leading-relaxed">
                Each security concern gets its own Rust crate with focused responsibility. No monolithic security
                module.
              </p>
            </div>
          </div>

          <div class="grid md:grid-cols-2 gap-4">
            <div class="bg-background border border-border rounded-lg p-4">
              <div class="font-bold text-foreground mb-2">auth-min</div>
              <div class="text-muted-foreground text-sm">Timing-safe token comparison, zero-trust authentication</div>
            </div>
            <div class="bg-background border border-border rounded-lg p-4">
              <div class="font-bold text-foreground mb-2">audit-logging</div>
              <div class="text-muted-foreground text-sm">Immutable append-only logs with 7-year retention</div>
            </div>
            <div class="bg-background border border-border rounded-lg p-4">
              <div class="font-bold text-foreground mb-2">input-validation</div>
              <div class="text-muted-foreground text-sm">Injection prevention, schema validation, sanitization</div>
            </div>
            <div class="bg-background border border-border rounded-lg p-4">
              <div class="font-bold text-foreground mb-2">secrets-management</div>
              <div class="text-muted-foreground text-sm">Secure credential storage, key rotation, encryption</div>
            </div>
            <div class="bg-background border border-border rounded-lg p-4">
              <div class="font-bold text-foreground mb-2">deadline-propagation</div>
              <div class="text-muted-foreground text-sm">Timeout enforcement, resource cleanup, cascading shutdown</div>
            </div>
          </div>
        </div>

        <!-- Process Isolation -->
        <div class="bg-card border border-border rounded-lg p-8">
          <div class="flex items-start gap-4 mb-6">
            <div class="h-12 w-12 rounded-lg bg-accent/10 flex items-center justify-center flex-shrink-0">
              <Lock class="h-6 w-6 text-accent" />
            </div>
            <div>
              <h3 class="text-2xl font-bold text-foreground mb-2">Process Isolation</h3>
              <p class="text-muted-foreground leading-relaxed">
                Each worker runs in its own process. Crashes are isolated. No shared state. Clean shutdown guaranteed.
              </p>
            </div>
          </div>

          <div class="space-y-4">
            <div class="flex items-start gap-3">
              <CheckCircle2 class="h-5 w-5 text-accent flex-shrink-0 mt-0.5" />
              <div>
                <div class="font-bold text-foreground">Sandboxed Execution</div>
                <div class="text-muted-foreground text-sm">Workers run in isolated processes with no shared memory</div>
              </div>
            </div>
            <div class="flex items-start gap-3">
              <CheckCircle2 class="h-5 w-5 text-accent flex-shrink-0 mt-0.5" />
              <div>
                <div class="font-bold text-foreground">Cascading Shutdown</div>
                <div class="text-muted-foreground text-sm">
                  rbee-keeper → queen-rbee → rbee-hive → workers (30s timeout, then force-kill)
                </div>
              </div>
            </div>
            <div class="flex items-start gap-3">
              <CheckCircle2 class="h-5 w-5 text-accent flex-shrink-0 mt-0.5" />
              <div>
                <div class="font-bold text-foreground">VRAM Cleanup</div>
                <div class="text-muted-foreground text-sm">
                  Workers free VRAM on shutdown, available for games/other apps
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
