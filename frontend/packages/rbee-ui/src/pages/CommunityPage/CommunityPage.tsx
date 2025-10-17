'use client'

export default function CommunityPage() {
  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-6">rbee Community</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Connect with developers, share knowledge, and contribute to the project.
      </p>
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Coming Soon</h2>
        <p>This page is under construction. We're setting up community channels.</p>
        <h3>Community Channels</h3>
        <ul>
          <li>
            <strong>GitHub Discussions:</strong> Ask questions, share ideas, and discuss features
          </li>
          <li>
            <strong>Discord:</strong> Real-time chat with the community (coming soon)
          </li>
          <li>
            <strong>GitHub Issues:</strong> Report bugs and request features
          </li>
          <li>
            <strong>Contributing:</strong> Contribute code, documentation, and examples
          </li>
        </ul>
        <h3>Get Involved</h3>
        <p>
          rbee is an open-source project (GPL-3.0). We welcome contributions from developers, designers, and
          documentation writers.
        </p>
      </div>
    </div>
  )
}
