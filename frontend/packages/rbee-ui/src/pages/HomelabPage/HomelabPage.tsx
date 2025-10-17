'use client'

export default function HomelabPage() {
  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-6">rbee for Homelab</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Self-hosted AI infrastructure for homelab enthusiasts and privacy advocates.
      </p>
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Coming Soon</h2>
        <p>This page is under construction. We're building comprehensive content for homelab enthusiasts.</p>
        <h3>What to Expect</h3>
        <ul>
          <li>SSH-based control for distributed deployments</li>
          <li>Multi-backend support (CUDA, Metal, CPU)</li>
          <li>Web UI + CLI tools</li>
          <li>Model catalog with auto-download</li>
          <li>Complete control and privacy</li>
        </ul>
      </div>
    </div>
  )
}
