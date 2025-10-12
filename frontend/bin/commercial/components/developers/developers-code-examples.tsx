export function DevelopersCodeExamples() {
  return (
    <section className="border-b border-border py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Build AI Agents with llama-orch-utils
          </h2>
          <p className="text-balance text-lg leading-relaxed text-muted-foreground">
            TypeScript utilities for building LLM pipelines and agentic workflows
          </p>
        </div>

        <div className="mx-auto mt-16 max-w-5xl space-y-8">
          {/* Example 1 */}
          <div>
            <h3 className="mb-4 text-xl font-semibold text-card-foreground">Simple Code Generation</h3>
            <div className="overflow-hidden rounded-lg border border-border bg-card">
              <div className="border-b border-border bg-muted px-4 py-2">
                <span className="text-sm text-muted-foreground">TypeScript</span>
              </div>
              <div className="p-4 font-mono text-sm">
                <pre className="overflow-x-auto text-foreground">
                  {`import { invoke } from '@llama-orch/utils';

const response = await invoke({
  prompt: 'Generate a TypeScript function that validates email addresses',
  model: 'llama-3.1-70b',
  maxTokens: 500
});

console.log(response.text);`}
                </pre>
              </div>
            </div>
          </div>

          {/* Example 2 */}
          <div>
            <h3 className="mb-4 text-xl font-semibold text-card-foreground">File Operations</h3>
            <div className="overflow-hidden rounded-lg border border-border bg-card">
              <div className="border-b border-border bg-muted px-4 py-2">
                <span className="text-sm text-muted-foreground">TypeScript</span>
              </div>
              <div className="p-4 font-mono text-sm">
                <pre className="overflow-x-auto text-foreground">
                  {`import { FileReader, FileWriter, invoke } from '@llama-orch/utils';

// Read schema
const schema = await FileReader.read('schema.sql');

// Generate API
const code = await invoke({
  prompt: \`Generate TypeScript CRUD API for:\\n\${schema}\`,
  model: 'llama-3.1-70b'
});

// Write result
await FileWriter.write('src/api.ts', code.text);`}
                </pre>
              </div>
            </div>
          </div>

          {/* Example 3 */}
          <div>
            <h3 className="mb-4 text-xl font-semibold text-card-foreground">Multi-Step Agent</h3>
            <div className="overflow-hidden rounded-lg border border-border bg-card">
              <div className="border-b border-border bg-muted px-4 py-2">
                <span className="text-sm text-muted-foreground">TypeScript</span>
              </div>
              <div className="p-4 font-mono text-sm">
                <pre className="overflow-x-auto text-foreground">
                  {`import { Thread, invoke, extractCode } from '@llama-orch/utils';

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
await FileWriter.write('review.md', review.text);`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
