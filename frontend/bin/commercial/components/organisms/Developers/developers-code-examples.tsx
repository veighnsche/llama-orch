import { CodeExamplesSection } from '@/components/organisms'

export function DevelopersCodeExamples() {
  return (
    <CodeExamplesSection
      title="Build AI agents with llama-orch-utils"
      subtitle="TypeScript utilities for LLM pipelines and agentic workflows."
      footerNote="Works with any OpenAI-compatible client."
      items={[
        {
          id: 'simple',
          title: 'Simple code generation',
          summary: 'Invoke to generate a TypeScript validator.',
          language: 'TypeScript',
          code: `import { invoke } from '@llama-orch/utils';

const response = await invoke({
  prompt: 'Generate a TypeScript function that validates email addresses',
  model: 'llama-3.1-70b',
  maxTokens: 500
});

console.log(response.text);`,
        },
        {
          id: 'files',
          title: 'File operations',
          summary: 'Read schema → generate API → write file.',
          language: 'TypeScript',
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
        },
        {
          id: 'agent',
          title: 'Multi-step agent',
          summary: 'Threaded review + suggestion extraction.',
          language: 'TypeScript',
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
        },
      ]}
    />
  )
}
