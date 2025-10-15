import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseSecurity } from './EnterpriseSecurity'

// Created by: TEAM-004

const meta = {
	title: 'Organisms/Enterprise/EnterpriseSecurity',
	component: EnterpriseSecurity,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The EnterpriseSecurity section presents six specialized security crates that harden every layer of the rbee platform. Uses a defense-in-depth approach with detailed security guarantees and quantifiable metrics.

## Composition
This organism contains:
- **Header**: "Enterprise-Grade Security" with defense-in-depth subtitle
- **Six Security Crates Grid**: auth-min, audit-logging, input-validation, secrets-management, jwt-guardian, deadline-propagation
- **auth-min**: Zero-Trust Authentication (timing-safe comparison, token fingerprinting, bearer token parsing, bind policy enforcement)
- **audit-logging**: Compliance Engine (immutable audit trail, 32 event types, tamper detection, 7-year retention)
- **input-validation**: First Line of Defense (SQL injection prevention, command injection prevention, path traversal prevention, resource exhaustion prevention)
- **secrets-management**: Credential Guardian (file-based loading, memory zeroization, permission validation, timing-safe verification)
- **jwt-guardian**: Token Lifecycle Manager (RS256/ES256 validation, clock-skew tolerance, revocation list, short-lived refresh tokens)
- **deadline-propagation**: Performance Enforcer (deadline propagation, remaining time calculation, deadline enforcement, timeout responses)
- **Security Guarantees**: <10% timing variance, 100% token fingerprinting, Zero memory leaks

## When to Use
- On the /enterprise page after the compliance section
- To demonstrate defense-in-depth security architecture
- To provide detailed security crate capabilities
- To show quantifiable security guarantees

## Content Requirements
- **Security Crates**: Detailed descriptions of each crate's capabilities
- **Threat Model**: What attacks each crate prevents
- **Guarantees**: Quantifiable security metrics
- **Documentation Links**: Links to detailed security docs

## Variants
- **Default**: All six security crates
- **Zero-Trust Focus**: Emphasize auth-min and jwt-guardian
- **Isolation Focus**: Emphasize input-validation and secrets-management

## Examples
\`\`\`tsx
import { EnterpriseSecurity } from '@rbee/ui/organisms/Enterprise/EnterpriseSecurity'

// Simple usage - no props needed
<EnterpriseSecurity />
\`\`\`

## Used In
- Enterprise page (/enterprise)

## Related Components
- SecurityCrate
- Card
- Badge

## Accessibility
- **Keyboard Navigation**: All documentation links are keyboard accessible
- **ARIA Labels**: Icons marked as aria-hidden, guarantees have aria-label
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Focus States**: Visible focus indicators on all interactive elements
- **Color Contrast**: Meets WCAG AA standards

## Enterprise Marketing Strategy

### Target Buyer Persona
- **Role**: CISO, Security Architect, IT Security Manager, CTO, Compliance Officer
- **Pain Points**: Security vulnerabilities, compliance requirements, audit failures, data breaches
- **Decision Criteria**: Defense-in-depth architecture, specific threat prevention, quantifiable guarantees

### Security as Selling Point
- **Not security theater**: Real security with quantifiable guarantees
- **Defense-in-depth**: Six specialized crates covering every layer
- **Threat model**: Specific attacks prevented (timing attacks, injection, memory leaks)
- **Compliance alignment**: Security crates support GDPR, SOC2, ISO 27001 requirements

### Security Crates Overview

**auth-min: Zero-Trust Authentication**
- **Threat**: Timing attacks (CWE-208), token leaks, accidental exposure
- **Prevention**: Constant-time comparison, token fingerprinting (SHA-256), bearer token parsing (RFC 6750), bind policy enforcement
- **Guarantee**: <10% timing variance (constant-time operations)

**audit-logging: Compliance Engine**
- **Threat**: Compliance violations, audit failures, tampered logs
- **Prevention**: Immutable audit trail (append-only), 32 event types, tamper detection (hash chains), 7-year retention (GDPR)
- **Guarantee**: Legally defensible proof (immutable, tamper-evident)

**input-validation: First Line of Defense**
- **Threat**: SQL injection, command injection, path traversal, resource exhaustion
- **Prevention**: Validates identifiers, prompts, paths before execution
- **Guarantee**: Trust no input (all inputs validated)

**secrets-management: Credential Guardian**
- **Threat**: Credential leaks, environment variable exposure, memory dumps
- **Prevention**: File-based loading (not env vars), memory zeroization on drop, permission validation (0600), timing-safe verification
- **Guarantee**: Zero memory leaks (zeroization on drop)

**jwt-guardian: Token Lifecycle Manager**
- **Threat**: Token forgery, expired tokens, revoked tokens, clock-skew issues
- **Prevention**: RS256/ES256 signature validation, clock-skew tolerance (±5 min), revocation list (Redis-backed), short-lived refresh tokens (15 min)
- **Guarantee**: Stateless yet secure (signature validation + revocation)

**deadline-propagation: Performance Enforcer**
- **Threat**: Resource exhaustion, SLO violations, cascading failures
- **Prevention**: Deadline propagation (client → worker), remaining time calculation, deadline enforcement (abort if insufficient), timeout responses (504 Gateway Timeout)
- **Guarantee**: Every millisecond counts (protects SLOs)

### Security Guarantees
- **<10% timing variance**: Constant-time operations prevent timing attacks
- **100% token fingerprinting**: No raw tokens in logs (SHA-256 fingerprints only)
- **Zero memory leaks**: Zeroization on drop (secrets never leak)

### Conversion Strategy
- **Primary CTA**: "View Security Docs" (self-serve education)
- **Secondary CTA**: "Request Security Audit" (sales-assisted evaluation)
- **Lead qualification**: Capture security requirements, threat model, compliance needs
- **Proof points**: Quantifiable guarantees, specific threat prevention, compliance alignment
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseSecurity>

export default meta
type Story = StoryObj<typeof meta>

export const EnterprisePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default enterprise security section showing all six security crates: auth-min (Zero-Trust Authentication), audit-logging (Compliance Engine), input-validation (First Line of Defense), secrets-management (Credential Guardian), jwt-guardian (Token Lifecycle Manager), and deadline-propagation (Performance Enforcer). Includes security guarantees: <10% timing variance, 100% token fingerprinting, Zero memory leaks.',
			},
		},
	},
}

export const ZeroTrustFocus: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant emphasizing zero-trust architecture and authentication. This version would focus on: auth-min (constant-time comparison, token fingerprinting, bearer token parsing, bind policy enforcement), jwt-guardian (RS256/ES256 validation, clock-skew tolerance, revocation list, short-lived refresh tokens), and secrets-management (file-based loading, memory zeroization, permission validation). Ideal for enterprises with strong authentication requirements or zero-trust mandates.',
			},
		},
	},
}

export const IsolationFocus: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Variant emphasizing input validation and isolation. This version would focus on: input-validation (SQL injection prevention, command injection prevention, path traversal prevention, resource exhaustion prevention), secrets-management (credential isolation, memory zeroization), and deadline-propagation (resource exhaustion prevention, SLO protection). Ideal for enterprises with strong isolation requirements or those concerned about injection attacks.',
			},
		},
	},
}
