<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented FAQSection component -->
<script setup lang="ts">
import Accordion from '../../atoms/Accordion/Accordion.vue'
import AccordionItem from '../../atoms/Accordion/AccordionItem.vue'
import AccordionTrigger from '../../atoms/Accordion/AccordionTrigger.vue'
import AccordionContent from '../../atoms/Accordion/AccordionContent.vue'

interface FAQ {
  question: string
  answer: string
}

interface Props {
  title?: string
  faqs?: FAQ[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Frequently Asked Questions',
  faqs: () => [
    {
      question: 'How is this different from Ollama?',
      answer: 'Ollama is great for single-machine inference. rbee orchestrates across your entire network—multiple GPUs, multiple machines, multiple backends (CUDA, Metal, CPU). Plus, rbee has task-based API with SSE streaming, programmable Rhai scheduler, and built-in marketplace federation.',
    },
    {
      question: 'Do I need to be a Rust expert?',
      answer: 'No. rbee is distributed as pre-built binaries. Use the CLI or Web UI. If you want to customize routing logic, you can write simple Rhai scripts (similar to JavaScript) or use YAML configs.',
    },
    {
      question: 'What if I don\'t have GPUs?',
      answer: 'rbee works with CPU-only inference too. It\'s slower, but functional. You can also federate to external GPU providers through the marketplace (coming in M3).',
    },
    {
      question: 'Is this production-ready?',
      answer: 'rbee is currently in M0 (milestone 0) with 68% of BDD scenarios passing. It\'s suitable for development and homelab use. Production-grade features (health monitoring, SLAs, marketplace) are coming in M1-M3 (Q1-Q3 2026).',
    },
    {
      question: 'How do I migrate from OpenAI API?',
      answer: 'Change one environment variable: export OPENAI_API_BASE=http://localhost:8080/v1. That\'s it. rbee is OpenAI-compatible.',
    },
    {
      question: 'What models are supported?',
      answer: 'Any GGUF model from Hugging Face. Llama, Mistral, Qwen, DeepSeek, etc. Image generation (Stable Diffusion) and audio (TTS) coming in M2.',
    },
    {
      question: 'Can I sell GPU time to others?',
      answer: 'Yes, in M3 (Q3 2026). The marketplace federation feature lets you register your rbee instance as a provider and earn revenue from excess capacity.',
    },
    {
      question: 'What about security?',
      answer: 'rbee runs entirely on your network. No external API calls. Rhai scripts are sandboxed (50ms timeout, memory limits, no file I/O). Platform mode (marketplace) uses immutable schedulers for multi-tenant security.',
    },
  ],
})
</script>

<template>
  <section class="py-24 bg-secondary">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <div class="max-w-3xl mx-auto">
        <Accordion type="single" :collapsible="true">
          <AccordionItem
            v-for="(faq, index) in faqs"
            :key="index"
            :value="`item-${index + 1}`"
            class="bg-card border border-border rounded-lg px-6"
          >
            <AccordionTrigger class="text-left font-semibold text-card-foreground hover:no-underline">
              {{ faq.question }}
            </AccordionTrigger>
            <AccordionContent class="text-muted-foreground leading-relaxed">
              {{ faq.answer }}
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </div>
  </section>
</template>
