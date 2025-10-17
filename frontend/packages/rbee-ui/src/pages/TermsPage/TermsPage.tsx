'use client'

export default function TermsPage() {
  return (
    <div className="container mx-auto px-4 py-16 max-w-4xl">
      <h1 className="text-4xl font-bold mb-6">Terms of Service</h1>
      <p className="text-muted-foreground mb-8">Last updated: October 17, 2025</p>
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Coming Soon</h2>
        <p>This page is under construction. We're preparing our comprehensive terms of service.</p>

        <h3>What Will Be Covered</h3>
        <ul>
          <li>Acceptable use policy</li>
          <li>Service availability and support</li>
          <li>Intellectual property rights</li>
          <li>Liability disclaimers</li>
          <li>Termination conditions</li>
          <li>Dispute resolution</li>
        </ul>

        <h3>Open Source License</h3>
        <p>
          rbee is licensed under GPL-3.0-or-later. The software is provided "as is" without warranty of any kind. For
          full license terms, see the{' '}
          <a href="https://github.com/veighnsche/llama-orch/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
            LICENSE file
          </a>{' '}
          in our repository.
        </p>

        <h3>Questions?</h3>
        <p>
          For questions about these terms, please open an issue on our{' '}
          <a href="https://github.com/veighnsche/llama-orch" target="_blank" rel="noopener noreferrer">
            GitHub repository
          </a>
          .
        </p>
      </div>
    </div>
  )
}
