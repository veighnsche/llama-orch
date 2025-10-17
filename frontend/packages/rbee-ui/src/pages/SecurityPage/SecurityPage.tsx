'use client'

export default function SecurityPage() {
  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-6">Security</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Enterprise-grade security architecture built for compliance and trust.
      </p>
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Coming Soon</h2>
        <p>This page is under construction. We're documenting our comprehensive security architecture.</p>
        <h3>Security Features</h3>
        <ul>
          <li>
            <strong>Audit Logging:</strong> Immutable audit logs with 7-year retention
          </li>
          <li>
            <strong>Tamper Detection:</strong> Blockchain-style hash chains for log integrity
          </li>
          <li>
            <strong>Data Residency:</strong> EU-only data storage and processing
          </li>
          <li>
            <strong>Compliance:</strong> GDPR, SOC2, and ISO 27001 aligned
          </li>
          <li>
            <strong>Security Architecture:</strong> 6 dedicated security crates
          </li>
        </ul>
        <h3>Reporting Security Issues</h3>
        <p>
          If you discover a security vulnerability, please report it via GitHub Security Advisories. We take security
          seriously and will respond promptly.
        </p>
      </div>
    </div>
  )
}
