'use client'

export default function ResearchPage() {
  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-6">rbee for Research</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Multi-modal AI platform for researchers and ML engineers.
      </p>
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Coming Soon</h2>
        <p>This page is under construction. We're building comprehensive content for AI researchers.</p>
        <h3>What to Expect</h3>
        <ul>
          <li>Multi-modal support (LLMs, Stable Diffusion, TTS, embeddings)</li>
          <li>Proof bundles for reproducibility</li>
          <li>Determinism suite for regression testing</li>
          <li>BDD-tested with executable specs</li>
          <li>Candle-powered (Rust ML framework)</li>
        </ul>
      </div>
    </div>
  )
}
