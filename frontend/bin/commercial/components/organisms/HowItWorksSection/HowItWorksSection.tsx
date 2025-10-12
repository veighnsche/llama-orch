import { SectionContainer, StepNumber, CodeBlock } from '@/components/molecules'

export function HowItWorksSection() {
  return (
    <SectionContainer title="From Zero to AI Infrastructure in 15 Minutes">

      <div className="max-w-5xl mx-auto space-y-16">
        {/* Step 1 */}
        <div className="grid lg:grid-cols-2 gap-8 items-center">
          <div className="space-y-4">
            <StepNumber number={1} />
            <h3 className="text-2xl font-bold text-foreground">Install rbee</h3>
            <p className="text-muted-foreground leading-relaxed">
              Get started from source. Works on Linux and macOS. Windows support planned.
            </p>
          </div>
          <CodeBlock code={`$ git clone https://github.com/veighnsche/llama-orch
$ cd llama-orch && cargo build --release
$ rbee-keeper daemon start
  ✓ rbee daemon started on port 8080`} />
        </div>

        {/* Step 2 */}
        <div className="grid lg:grid-cols-2 gap-8 items-center">
          <div className="space-y-4 lg:order-2">
            <StepNumber number={2} />
            <h3 className="text-2xl font-bold text-foreground">Add Your Machines</h3>
            <p className="text-muted-foreground leading-relaxed">
              Connect all your GPUs across your network. rbee automatically detects CUDA, Metal, and CPU backends.
            </p>
          </div>
          <CodeBlock 
            code={`$ rbee-keeper setup add-node \
  --name workstation \
  --ssh-host 192.168.1.10
  ✓ Added node: workstation (2x RTX 4090)

$ rbee-keeper setup add-node \
  --name mac \
  --ssh-host 192.168.1.20
  ✓ Added node: mac (M2 Ultra)`}
            className="lg:order-1"
          />
        </div>

        {/* Step 3 */}
        <div className="grid lg:grid-cols-2 gap-8 items-center">
          <div className="space-y-4">
            <StepNumber number={3} />
            <h3 className="text-2xl font-bold text-foreground">Start Inference</h3>
            <p className="text-muted-foreground leading-relaxed">
              Point your tools to localhost. Zed, Cursor, or any OpenAI-compatible tool works instantly.
            </p>
          </div>
          <CodeBlock code={`$ export OPENAI_API_BASE=http://localhost:8080/v1

# Now Zed, Cursor, or any OpenAI-compatible
# tool works with your local infrastructure!`} />
        </div>

        {/* Step 4 */}
        <div className="grid lg:grid-cols-2 gap-8 items-center">
          <div className="space-y-4 lg:order-2">
            <StepNumber number={4} />
            <h3 className="text-2xl font-bold text-foreground">Build AI Agents</h3>
            <p className="text-muted-foreground leading-relaxed">
              Use the TypeScript SDK to build custom AI agents, tools, and workflows.
            </p>
          </div>
          <CodeBlock 
            code={`import { invoke } from '@llama-orch/utils'

const code = await invoke({
  prompt: 'Generate API',
  model: 'llama-3.1-70b'
})`}
            language="typescript"
            className="lg:order-1"
          />
        </div>
      </div>
    </SectionContainer>
  )
}
