'use client'

export default function PrivacyPage() {
  return (
    <div className="container mx-auto px-4 py-16 max-w-4xl">
      <h1 className="text-4xl font-bold mb-6">Privacy Policy</h1>
      <p className="text-muted-foreground mb-8">Last updated: October 17, 2025</p>
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Coming Soon</h2>
        <p>This page is under construction. We're preparing our comprehensive GDPR-compliant privacy policy.</p>

        <h3>What Will Be Covered</h3>
        <ul>
          <li>Data collection and usage</li>
          <li>Cookie policy</li>
          <li>User rights (access, deletion, portability)</li>
          <li>Data retention policies</li>
          <li>Third-party services</li>
          <li>EU data residency</li>
          <li>Contact information for privacy inquiries</li>
        </ul>

        <h3>Our Commitment</h3>
        <p>
          rbee is built with privacy in mind. As an open-source, self-hosted platform, you maintain complete control
          over your data. We are committed to GDPR compliance and transparent data practices.
        </p>

        <h3>Questions?</h3>
        <p>
          For privacy-related questions, please open an issue on our{' '}
          <a href="https://github.com/veighnsche/llama-orch" target="_blank" rel="noopener noreferrer">
            GitHub repository
          </a>
          .
        </p>
      </div>
    </div>
  )
}
