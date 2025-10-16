import { Clock, Eye, KeyRound, Lock, Server, Shield } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export interface SecurityCrateData {
  name: string;
  description: string;
  icon?: LucideIcon;
  hoverColor?: string;
}

/**
 * Shared security crates data used across Features and Enterprise sections
 * 
 * Features section shows 5 core crates (simplified view)
 * Enterprise section shows all 6 crates with full details
 */
export const SECURITY_CRATES: SecurityCrateData[] = [
  {
    name: 'auth-min',
    description: 'Timing-safe tokens, zero-trust auth.',
    icon: Lock,
    hoverColor: 'hover:border-chart-2/50',
  },
  {
    name: 'audit-logging',
    description: 'Append-only logs, 7-year retention.',
    icon: Eye,
    hoverColor: 'hover:border-chart-3/50',
  },
  {
    name: 'input-validation',
    description: 'Injection prevention, schema validation.',
    icon: Shield,
    hoverColor: 'hover:border-primary/50',
  },
  {
    name: 'secrets-management',
    description: 'Encrypted storage, rotation, KMS-friendly.',
    icon: Server,
    hoverColor: 'hover:border-amber-500/50',
  },
  {
    name: 'jwt-guardian',
    description: 'RS256 validation, revocation lists, short-lived tokens.',
    icon: KeyRound,
    hoverColor: 'hover:border-chart-2/50',
  },
  {
    name: 'deadline-propagation',
    description: 'Timeouts, cleanup, cascading shutdown.',
    icon: Clock,
    hoverColor: 'hover:border-chart-3/50',
  },
];

/**
 * Core 5 crates shown in Features section (excludes jwt-guardian)
 */
export const CORE_SECURITY_CRATES = SECURITY_CRATES.filter(
  (crate) => crate.name !== 'jwt-guardian'
);
