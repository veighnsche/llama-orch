import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { SectionContainer } from "@/components/primitives"

export function FAQSection() {
  return (
    <SectionContainer
      title="Frequently Asked Questions"
      bgVariant="secondary"
    >

        <div className="max-w-3xl mx-auto">
          <Accordion type="single" collapsible className="space-y-4">
            <AccordionItem value="item-1" className="bg-card border border-border rounded-lg px-6">
              <AccordionTrigger className="text-left font-semibold text-card-foreground hover:no-underline">
                How is this different from Ollama?
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed">
                Ollama is great for single-machine inference. rbee orchestrates across your entire network—multiple
                GPUs, multiple machines, multiple backends (CUDA, Metal, CPU). Plus, rbee has task-based API with SSE
                streaming, programmable Rhai scheduler, and built-in marketplace federation.
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-2" className="bg-card border border-border rounded-lg px-6">
              <AccordionTrigger className="text-left font-semibold text-card-foreground hover:no-underline">
                Do I need to be a Rust expert?
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed">
                No. rbee is distributed as pre-built binaries. Use the CLI or Web UI. If you want to customize routing
                logic, you can write simple Rhai scripts (similar to JavaScript) or use YAML configs.
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-3" className="bg-card border border-border rounded-lg px-6">
              <AccordionTrigger className="text-left font-semibold text-card-foreground hover:no-underline">
                What if I don't have GPUs?
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed">
                rbee works with CPU-only inference too. It's slower, but functional. You can also federate to external
                GPU providers through the marketplace (coming in M3).
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-4" className="bg-card border border-border rounded-lg px-6">
              <AccordionTrigger className="text-left font-semibold text-card-foreground hover:no-underline">
                Is this production-ready?
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed">
                rbee is currently in M0 (milestone 0) with 68% of BDD scenarios passing. It's suitable for development
                and homelab use. Production-grade features (health monitoring, SLAs, marketplace) are coming in M1-M3
                (Q1-Q3 2026).
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-5" className="bg-card border border-border rounded-lg px-6">
              <AccordionTrigger className="text-left font-semibold text-card-foreground hover:no-underline">
                How do I migrate from OpenAI API?
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed">
                Change one environment variable:{" "}
                <code className="bg-muted px-2 py-1 rounded text-sm font-mono">
                  export OPENAI_API_BASE=http://localhost:8080/v1
                </code>
                . That's it. rbee is OpenAI-compatible.
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-6" className="bg-card border border-border rounded-lg px-6">
              <AccordionTrigger className="text-left font-semibold text-card-foreground hover:no-underline">
                What models are supported?
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed">
                Any GGUF model from Hugging Face. Llama, Mistral, Qwen, DeepSeek, etc. Image generation (Stable
                Diffusion) and audio (TTS) coming in M2.
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-7" className="bg-card border border-border rounded-lg px-6">
              <AccordionTrigger className="text-left font-semibold text-card-foreground hover:no-underline">
                Can I sell GPU time to others?
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed">
                Yes, in M3 (Q3 2026). The marketplace federation feature lets you register your rbee instance as a
                provider and earn revenue from excess capacity.
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-8" className="bg-card border border-border rounded-lg px-6">
              <AccordionTrigger className="text-left font-semibold text-card-foreground hover:no-underline">
                What about security?
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed">
                rbee runs entirely on your network. No external API calls. Rhai scripts are sandboxed (50ms timeout,
                memory limits, no file I/O). Platform mode (marketplace) uses immutable schedulers for multi-tenant
                security.
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </SectionContainer>
  )
}
