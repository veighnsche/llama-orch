<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersCodeExamples -->
<script setup lang="ts">
interface CodeExample {
  title: string
  code: string
  language?: string
}

interface Props {
  title?: string
  subtitle?: string
  examples?: CodeExample[]
}

withDefaults(defineProps<Props>(), {
  title: 'Build AI Agents with llama-orch-utils',
  subtitle: 'TypeScript utilities for building LLM pipelines and agentic workflows',
  examples: () => [
    {
      title: 'Simple Code Generation',
      code: `import { invoke } from '@llama-orch/utils';

const response = await invoke({
  prompt: 'Generate a TypeScript function that validates email addresses',
  model: 'llama-3.1-70b',
  maxTokens: 500
});

console.log(response.text);`,
      language: 'TypeScript',
    },
    {
      title: 'File Operations',
      code: `import { FileReader, FileWriter, invoke } from '@llama-orch/utils';

// Read schema
const schema = await FileReader.read('schema.sql');

// Generate API
const code = await invoke({
  prompt: \`Generate TypeScript CRUD API for:\\n\${schema}\`,
  model: 'llama-3.1-70b'
});

// Write result
await FileWriter.write('src/api.ts', code.text);`,
      language: 'TypeScript',
    },
    {
      title: 'Multi-Step Agent',
      code: `import { Thread, invoke, extractCode } from '@llama-orch/utils';

// Build conversation thread
const thread = Thread.create()
  .addSystem('You are a code review expert')
  .addUser('Review this code for security issues')
  .addUser(await FileReader.read('src/auth.ts'));

// Get review
const review = await invoke({
  messages: thread.toMessages(),
  model: 'llama-3.1-70b'
});

// Extract suggestions
const suggestions = extractCode(review.text, 'typescript');
await FileWriter.write('review.md', review.text);`,
      language: 'TypeScript',
    },
  ],
})
</script>

<template>
  <section class="border-b border-border py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
        <p class="text-balance text-lg leading-relaxed text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="mx-auto mt-16 max-w-5xl space-y-8">
        <div v-for="(example, index) in examples" :key="index">
          <h3 class="mb-4 text-xl font-semibold text-foreground">{{ example.title }}</h3>
          <div class="overflow-hidden rounded-lg border border-border bg-secondary">
            <div class="border-b border-border bg-muted px-4 py-2">
              <span class="text-sm text-muted-foreground">{{ example.language || 'code' }}</span>
            </div>
            <div class="p-4 font-mono text-sm">
              <pre class="overflow-x-auto text-foreground">{{ example.code }}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
