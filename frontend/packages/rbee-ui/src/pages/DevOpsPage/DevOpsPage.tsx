'use client'

export default function DevOpsPage() {
  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-6">rbee for DevOps</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Production-ready AI orchestration for DevOps engineers and SREs.
      </p>
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Coming Soon</h2>
        <p>This page is under construction. We're building comprehensive content for DevOps engineers.</p>
        <h3>What to Expect</h3>
        <ul>
          <li>Cascading shutdown (prevents orphaned processes)</li>
          <li>Health monitoring (30-second heartbeats)</li>
          <li>Multi-node SSH control</li>
          <li>Lifecycle management (daemon, hive, worker control)</li>
          <li>Proof bundles for debugging</li>
        </ul>
      </div>
    </div>
  )
}
