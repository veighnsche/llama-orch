'use client'

import { SectionContainer, StepNumber, CodeBlock } from '@/components/molecules'
import { useState } from 'react'
import { cn } from '@/lib/utils'

type OS = 'linux' | 'macos' | 'windows'

export function HowItWorksSection() {
  const [selectedOS, setSelectedOS] = useState<OS>('linux')

  const installCommands = {
    linux: `$ git clone https://github.com/veighnsche/llama-orch
$ cd llama-orch && cargo build --release
$ rbee-keeper daemon start
  ✓ rbee daemon started on port 8080`,
    macos: `$ git clone https://github.com/veighnsche/llama-orch
$ cd llama-orch && cargo build --release
$ rbee-keeper daemon start
  ✓ rbee daemon started on port 8080`,
    windows: '# Windows support coming soon',
  }

  return (
    <SectionContainer title={<span className="text-balance">From Zero to AI Infrastructure in 15 Minutes</span>}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-14 md:space-y-20">
        {/* Step 1: Install rbee */}
        <div className="grid lg:grid-cols-2 gap-8 lg:gap-10 items-start animate-in fade-in slide-in-from-bottom-2 duration-500">
          <div className="space-y-4">
            <StepNumber number={1} aria-hidden="true" />
            <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">Install rbee</h3>
            <p className="text-base md:text-lg text-muted-foreground leading-relaxed">
              Build from source on Linux or macOS. Windows support is planned.
            </p>
            <p className="text-sm text-muted-foreground/80">Cold start to running daemon in minutes.</p>
            <div className="flex flex-wrap gap-2 pt-1">
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] font-medium text-foreground/90">
                ~3 min
              </span>
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] font-medium text-foreground/90">
                Rust toolchain
              </span>
            </div>

            {/* OS Switcher */}
            <div className="pt-2" role="tablist" aria-label="Operating system">
              <div className="inline-flex rounded-lg bg-secondary p-1 gap-1">
                {(['linux', 'macos', 'windows'] as const).map((os) => (
                  <button
                    key={os}
                    role="tab"
                    aria-selected={selectedOS === os}
                    aria-controls="install-code"
                    aria-label={`${os.charAt(0).toUpperCase() + os.slice(1)} installation`}
                    onClick={() => setSelectedOS(os)}
                    disabled={os === 'windows'}
                    className={cn(
                      'px-3 py-1.5 text-sm font-medium rounded-md transition-all',
                      selectedOS === os
                        ? 'bg-background text-foreground shadow-sm'
                        : 'text-muted-foreground hover:text-foreground',
                      os === 'windows' && 'opacity-50 cursor-not-allowed'
                    )}
                  >
                    {os.charAt(0).toUpperCase() + os.slice(1)}
                    {os === 'windows' && <span className="ml-1 text-[10px]">(soon)</span>}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <CodeBlock
              code={installCommands[selectedOS]}
              title="Install & start daemon"
              language="bash"
              copyable
            />
            <div className="text-xs text-muted-foreground px-1">Daemon listens on :8080 by default.</div>
          </div>
        </div>

        {/* Step 2: Add Your Machines */}
        <div className="grid lg:grid-cols-2 gap-8 lg:gap-10 items-start animate-in fade-in slide-in-from-bottom-2 duration-500">
          <div className="space-y-4 lg:order-2">
            <StepNumber number={2} aria-hidden="true" />
            <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">Add Your Machines</h3>
            <p className="text-base md:text-lg text-muted-foreground leading-relaxed">
              Enroll every GPU host. rbee auto-detects CUDA, Metal, and CPU backends.
            </p>
            <p className="text-sm text-muted-foreground/80">Multi-node, mixed backends, one pool.</p>
            <div className="flex flex-wrap gap-2 pt-1">
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] font-medium text-foreground/90">
                ~5 min
              </span>
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] font-medium text-foreground/90">
                SSH access
              </span>
            </div>
          </div>

          <div className="space-y-3 lg:order-1">
            <CodeBlock
              code={`$ rbee-keeper setup add-node \\
  --name workstation \\
  --ssh-host 192.168.1.10
  ✓ Added node: workstation (2x RTX 4090)

$ rbee-keeper setup add-node \\
  --name mac \\
  --ssh-host 192.168.1.20
  ✓ Added node: mac (M2 Ultra)`}
              title="Enroll nodes"
              language="bash"
              copyable
            />
            <div className="text-xs text-muted-foreground px-1">
              Nodes authenticate over SSH; labels auto-assign on first handshake.
            </div>
          </div>
        </div>

        {/* Step 3: Start Inference */}
        <div className="grid lg:grid-cols-2 gap-8 lg:gap-10 items-start animate-in fade-in slide-in-from-bottom-2 duration-500">
          <div className="space-y-4">
            <StepNumber number={3} aria-hidden="true" />
            <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">Start Inference</h3>
            <p className="text-base md:text-lg text-muted-foreground leading-relaxed">
              Point your tools at localhost. Zed, Cursor, and any OpenAI-compatible client just work.
            </p>
            <p className="text-sm text-muted-foreground/80">No cloud keys, no egress.</p>
            <div className="flex flex-wrap gap-2 pt-1">
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] font-medium text-foreground/90">
                ~2 min
              </span>
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] font-medium text-foreground/90">
                Open port 8080
              </span>
            </div>
          </div>

          <div className="space-y-3">
            <CodeBlock
              code={`$ export OPENAI_API_BASE=http://localhost:8080/v1

# Now Zed, Cursor, or any OpenAI-compatible
# tool works with your local infrastructure!`}
              title="Point clients to rbee"
              language="bash"
              copyable
            />
            <div className="text-xs text-muted-foreground px-1">
              Leave OPENAI_API_KEY unset if your client requires one—rbee will intercept.
            </div>
          </div>
        </div>

        {/* Step 4: Build AI Agents */}
        <div className="grid lg:grid-cols-2 gap-8 lg:gap-10 items-start animate-in fade-in slide-in-from-bottom-2 duration-500">
          <div className="space-y-4 lg:order-2">
            <StepNumber number={4} aria-hidden="true" />
            <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">Build AI Agents</h3>
            <p className="text-base md:text-lg text-muted-foreground leading-relaxed">
              Use the TypeScript SDK to wire tools, memory, and workflows.
            </p>
            <p className="text-sm text-muted-foreground/80">Ship agents that run on your hardware.</p>
            <div className="flex flex-wrap gap-2 pt-1">
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] font-medium text-foreground/90">
                ~5 min
              </span>
              <span className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] font-medium text-foreground/90">
                Node.js
              </span>
            </div>
          </div>

          <div className="space-y-3 lg:order-1">
            <CodeBlock
              code={`import { invoke } from '@llama-orch/utils'

const code = await invoke({
  prompt: 'Generate API',
  model: 'llama-3.1-70b'
})`}
              title="TypeScript SDK"
              language="typescript"
              copyable
            />
            <div className="text-xs text-muted-foreground px-1">
              Models are resolved by scheduler policy; override per-call if needed.
            </div>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
