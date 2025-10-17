'use client'

export default function EducationPage() {
  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-6">rbee for Education</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Learn distributed systems from nature-inspired architecture.
      </p>
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Coming Soon</h2>
        <p>This page is under construction. We're building comprehensive content for students and educators.</p>
        <h3>What to Expect</h3>
        <ul>
          <li>Nature-inspired beehive architecture</li>
          <li>Open source (GPL-3.0) - study real production code</li>
          <li>BDD-tested with Gherkin scenarios</li>
          <li>Rust + Candle for ML</li>
          <li>Smart/dumb architecture patterns</li>
        </ul>
      </div>
    </div>
  )
}
