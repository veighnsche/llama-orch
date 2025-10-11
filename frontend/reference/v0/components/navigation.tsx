"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Menu, X, Github } from "lucide-react"
import { useState } from "react"

export function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-950/95 backdrop-blur-sm border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-2 text-xl font-semibold text-amber-500 hover:text-amber-400 transition-colors"
          >
            <span className="text-2xl">🐝</span>
            <span>rbee</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            <Link href="/features" className="text-slate-300 hover:text-white transition-colors">
              Features
            </Link>
            <Link href="/use-cases" className="text-slate-300 hover:text-white transition-colors">
              Use Cases
            </Link>
            <Link href="/pricing" className="text-slate-300 hover:text-white transition-colors">
              Pricing
            </Link>
            <Link href="/developers" className="text-slate-300 hover:text-white transition-colors">
              For Developers
            </Link>
            <Link href="/gpu-providers" className="text-slate-300 hover:text-white transition-colors">
              For Providers
            </Link>
            <Link href="/enterprise" className="text-slate-300 hover:text-white transition-colors">
              For Enterprise
            </Link>

            <a
              href="https://github.com/veighnsche/llama-orch"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-300 hover:text-white transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>

            <Button size="sm" className="bg-amber-500 hover:bg-amber-600 text-slate-950">
              Join Waitlist
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden text-slate-300 hover:text-white"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 space-y-4 border-t border-slate-800">
            <Link
              href="/features"
              className="block text-slate-300 hover:text-white transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              Features
            </Link>
            <Link
              href="/use-cases"
              className="block text-slate-300 hover:text-white transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              Use Cases
            </Link>
            <Link
              href="/pricing"
              className="block text-slate-300 hover:text-white transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              Pricing
            </Link>
            <Link
              href="/developers"
              className="block text-slate-300 hover:text-white transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Developers
            </Link>
            <Link
              href="/gpu-providers"
              className="block text-slate-300 hover:text-white transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Providers
            </Link>
            <Link
              href="/enterprise"
              className="block text-slate-300 hover:text-white transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Enterprise
            </Link>
            <a
              href="https://github.com/veighnsche/llama-orch"
              target="_blank"
              rel="noopener noreferrer"
              className="block text-slate-300 hover:text-white transition-colors"
            >
              GitHub
            </a>
            <Button size="sm" className="w-full bg-amber-500 hover:bg-amber-600 text-slate-950">
              Join Waitlist
            </Button>
          </div>
        )}
      </div>
    </nav>
  )
}
