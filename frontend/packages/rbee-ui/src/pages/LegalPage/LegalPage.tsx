'use client'

import Link from 'next/link'

export default function LegalPage() {
  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-6">Legal</h1>
      <p className="text-xl text-muted-foreground mb-8">Legal information and compliance documentation.</p>
      <div className="grid gap-6 md:grid-cols-2 max-w-4xl">
        <Link
          href="/legal/privacy"
          className="block p-6 border border-border rounded-lg hover:border-primary transition-colors"
        >
          <h2 className="text-2xl font-semibold mb-2">Privacy Policy</h2>
          <p className="text-muted-foreground">
            How we collect, use, and protect your data. GDPR-compliant privacy practices.
          </p>
        </Link>
        <Link
          href="/legal/terms"
          className="block p-6 border border-border rounded-lg hover:border-primary transition-colors"
        >
          <h2 className="text-2xl font-semibold mb-2">Terms of Service</h2>
          <p className="text-muted-foreground">Terms and conditions for using rbee services and software.</p>
        </Link>
      </div>
    </div>
  )
}
