import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type { CTATemplateProps, FAQTemplateProps, HeroTemplateProps } from '@rbee/ui/templates'
import { ArrowRight, BookOpen, FileText, Scale } from 'lucide-react'

// Props Objects (in visual order matching page composition)
// ============================================================================

/**
 * Hero section props - Simple title and last updated date
 */
export const termsHeroProps: HeroTemplateProps = {
  badge: {
    variant: 'icon',
    text: 'Legal • Terms of Service',
    icon: <Scale className="h-4 w-4" />,
  },
  headline: {
    variant: 'simple',
    content: 'Terms of Service',
  },
  subcopy: (
    <div className="space-y-2">
      <p>
        These Terms of Service govern your use of rbee (the "Service"), an open-source AI infrastructure orchestration
        platform.
      </p>
      <p className="text-sm text-muted-foreground">
        <strong>Last Updated:</strong> October 17, 2025 • <strong>Effective Date:</strong> October 17, 2025 •{' '}
        <strong>Version:</strong> 1.0
      </p>
    </div>
  ),
  subcopyMaxWidth: 'wide',
  ctas: {
    primary: {
      label: 'Contact Legal Team',
      href: 'mailto:legal@rbee.dev',
    },
    secondary: {
      label: 'View Privacy Policy',
      href: '/legal/privacy',
      variant: 'outline',
    },
  },
  helperText: 'Please read these terms carefully before using rbee.',
  aside: (
    <div className="flex items-center justify-center rounded-lg border border-border bg-card/60 backdrop-blur-sm p-8 shadow-sm">
      <div className="text-center space-y-4">
        <FileText className="h-16 w-16 mx-auto text-muted-foreground" />
        <div className="space-y-2">
          <p className="text-sm font-medium text-card-foreground">Legal Document</p>
          <p className="text-xs text-muted-foreground">Please read carefully</p>
        </div>
      </div>
    </div>
  ),
  asideAriaLabel: 'Legal document icon',
  layout: {
    leftCols: 7,
    rightCols: 5,
    gap: 12,
  },
  background: {
    variant: 'gradient',
  },
  padding: 'default',
}

/**
 * Hero container props
 */
export const termsHeroContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
}

/**
 * FAQ template content - Terms sections as Q&A format
 */
export const termsFaqProps: FAQTemplateProps = {
  badgeText: 'Terms & Conditions',
  searchPlaceholder: 'Search terms…',
  emptySearchKeywords: ['license', 'GPL', 'liability', 'warranty'],
  expandAllLabel: 'Expand all',
  collapseAllLabel: 'Collapse all',
  categories: [
    'Agreement',
    'License',
    'Use Policy',
    'Intellectual Property',
    'Liability',
    'Termination',
    'Dispute Resolution',
    'General',
  ],
  faqItems: [
    {
      value: 'acceptance',
      question: '1. Acceptance of Terms',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            By accessing or using rbee, you agree to be bound by these Terms of Service. If you do not agree to these
            terms, you may not use the Service.
          </p>
          <p>
            <strong>Who can use rbee:</strong>
          </p>
          <ul>
            <li>You must be at least 18 years old or have parental/guardian consent</li>
            <li>You must have the legal capacity to enter into binding contracts</li>
            <li>You must comply with all applicable laws and regulations</li>
          </ul>
          <p>
            <strong>Changes to terms:</strong> We reserve the right to modify these terms at any time. Continued use
            after changes constitutes acceptance of the modified terms.
          </p>
        </div>
      ),
      category: 'Agreement',
    },
    {
      value: 'service-description',
      question: '2. Service Description',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong>What rbee provides:</strong>
          </p>
          <ul>
            <li>Open-source AI infrastructure orchestration software</li>
            <li>Tools to manage and deploy LLMs across distributed hardware</li>
            <li>OpenAI-compatible API for inference tasks</li>
            <li>Web UI and CLI for system management</li>
          </ul>
          <p>
            <strong>Service availability:</strong> rbee is provided "as is" with no guaranteed uptime or availability.
            The software is self-hosted on your own infrastructure.
          </p>
          <p>
            <strong>Beta/Alpha disclaimers:</strong> rbee is currently in active development (M0 milestone). Features
            may change, break, or be removed without notice. Not recommended for production use without thorough
            testing.
          </p>
        </div>
      ),
      category: 'Agreement',
    },
    {
      value: 'gpl-license',
      question: '3. GPL-3.0-or-later License',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            rbee is licensed under the{' '}
            <a
              href="https://www.gnu.org/licenses/gpl-3.0.html"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              GNU General Public License v3.0 or later (GPL-3.0-or-later)
            </a>
            .
          </p>
          <p>
            <strong>Key license obligations:</strong>
          </p>
          <ul>
            <li>You may use, modify, and distribute rbee freely</li>
            <li>You must disclose source code for any modifications you distribute</li>
            <li>Derivative works must also be licensed under GPL-3.0-or-later</li>
            <li>You must preserve copyright and license notices</li>
            <li>No warranty is provided (see Section 7)</li>
          </ul>
          <p>
            <strong>Derivative works:</strong> If you modify rbee and distribute your modifications, you must make the
            source code available under GPL-3.0-or-later. Internal use within your organization does not require
            disclosure.
          </p>
          <p>
            Full license text:{' '}
            <a
              href="https://github.com/veighnsche/llama-orch/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              LICENSE
            </a>
          </p>
        </div>
      ),
      category: 'License',
    },
    {
      value: 'acceptable-use',
      question: '4. Acceptable Use Policy',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong>Permitted uses:</strong>
          </p>
          <ul>
            <li>Personal, educational, and commercial use of rbee</li>
            <li>Hosting AI models for legitimate business purposes</li>
            <li>Research and development activities</li>
            <li>Contributing to the open-source project</li>
          </ul>
          <p>
            <strong>Prohibited uses:</strong>
          </p>
          <ul>
            <li>Illegal activities or content generation</li>
            <li>Harassment, abuse, or harmful content</li>
            <li>Violating third-party intellectual property rights</li>
            <li>Circumventing security measures or access controls</li>
            <li>Distributing malware or conducting attacks</li>
            <li>Impersonating others or misrepresenting affiliation</li>
          </ul>
          <p>
            <strong>User responsibilities:</strong> You are solely responsible for your use of rbee, including all
            content generated, models deployed, and compliance with applicable laws.
          </p>
        </div>
      ),
      category: 'Use Policy',
    },
    {
      value: 'content-guidelines',
      question: '5. Content Guidelines',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong>Your content:</strong> You retain all rights to content you create using rbee. You are responsible
            for ensuring your content complies with all applicable laws.
          </p>
          <p>
            <strong>Model usage:</strong> When using third-party models (e.g., from Hugging Face), you must comply with
            their respective licenses and terms of use.
          </p>
          <p>
            <strong>Community contributions:</strong> By contributing code, documentation, or other materials to the
            rbee project, you grant the project a perpetual, worldwide, non-exclusive license to use your contributions
            under GPL-3.0-or-later.
          </p>
        </div>
      ),
      category: 'Use Policy',
    },
    {
      value: 'intellectual-property',
      question: '6. Intellectual Property',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong>Ownership of code:</strong> rbee source code is owned by its contributors and licensed under
            GPL-3.0-or-later. No proprietary rights are claimed beyond what the GPL allows.
          </p>
          <p>
            <strong>Trademarks:</strong> "rbee" and associated logos are trademarks. You may use them to refer to the
            software but not to imply endorsement without permission.
          </p>
          <p>
            <strong>Copyright notices:</strong> All source files contain copyright notices. You must preserve these
            notices in any distributions or derivative works.
          </p>
          <p>
            <strong>Third-party components:</strong> rbee includes third-party libraries and dependencies, each with
            their own licenses. See{' '}
            <code className="px-1.5 py-0.5 bg-muted rounded text-sm">Cargo.toml</code> and{' '}
            <code className="px-1.5 py-0.5 bg-muted rounded text-sm">package.json</code> files for details.
          </p>
        </div>
      ),
      category: 'Intellectual Property',
    },
    {
      value: 'no-warranty',
      question: '7. Disclaimer of Warranties',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong className="uppercase">
              rbee is provided "as is" and "as available" without warranties of any kind, either express or implied.
            </strong>
          </p>
          <p>
            <strong>No warranty includes:</strong>
          </p>
          <ul>
            <li>No warranty of merchantability or fitness for a particular purpose</li>
            <li>No warranty of non-infringement</li>
            <li>No warranty of error-free or uninterrupted operation</li>
            <li>No warranty of data accuracy or reliability</li>
            <li>No warranty of security or freedom from vulnerabilities</li>
          </ul>
          <p>
            You use rbee at your own risk. We do not warrant that rbee will meet your requirements or that defects will
            be corrected.
          </p>
        </div>
      ),
      category: 'Liability',
    },
    {
      value: 'limitation-liability',
      question: '8. Limitation of Liability',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong className="uppercase">
              To the maximum extent permitted by law, the rbee project, its contributors, and maintainers shall not be
              liable for any damages arising from your use of rbee.
            </strong>
          </p>
          <p>
            <strong>This includes but is not limited to:</strong>
          </p>
          <ul>
            <li>Direct, indirect, incidental, or consequential damages</li>
            <li>Loss of profits, data, or business opportunities</li>
            <li>Hardware damage or system failures</li>
            <li>Security breaches or data loss</li>
            <li>Third-party claims</li>
          </ul>
          <p>
            Some jurisdictions do not allow the exclusion of certain warranties or limitation of liability. In such
            cases, our liability is limited to the maximum extent permitted by law.
          </p>
        </div>
      ),
      category: 'Liability',
    },
    {
      value: 'indemnification',
      question: '9. Indemnification',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            You agree to indemnify, defend, and hold harmless the rbee project, its contributors, and maintainers from
            any claims, damages, losses, liabilities, and expenses (including legal fees) arising from:
          </p>
          <ul>
            <li>Your use or misuse of rbee</li>
            <li>Your violation of these Terms of Service</li>
            <li>Your violation of any third-party rights</li>
            <li>Content you generate or deploy using rbee</li>
            <li>Your violation of applicable laws or regulations</li>
          </ul>
        </div>
      ),
      category: 'Liability',
    },
    {
      value: 'termination',
      question: '10. Termination',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong>Your rights:</strong> You may stop using rbee at any time. Your rights under the GPL-3.0-or-later
            license continue unless you violate the license terms.
          </p>
          <p>
            <strong>Our rights:</strong> We may terminate or suspend access to community resources (GitHub, Discord,
            forums) for violations of these terms or community guidelines.
          </p>
          <p>
            <strong>Effect of termination:</strong> Upon termination, you must cease using rbee and delete all copies in
            your possession, unless the GPL license permits continued use.
          </p>
          <p>
            <strong>Survival clauses:</strong> Sections 6-9 (Intellectual Property, Warranties, Liability,
            Indemnification) survive termination.
          </p>
        </div>
      ),
      category: 'Termination',
    },
    {
      value: 'account-deletion',
      question: '11. Account Deletion',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            rbee is self-hosted software. There are no centralized accounts to delete. If you use community resources
            (GitHub, Discord), follow their respective account deletion procedures.
          </p>
          <p>
            To remove rbee from your infrastructure, uninstall the software and delete any associated data files
            according to the documentation.
          </p>
        </div>
      ),
      category: 'Termination',
    },
    {
      value: 'governing-law',
      question: '12. Governing Law',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            These Terms of Service are governed by the laws of the Netherlands, without regard to conflict of law
            principles.
          </p>
          <p>
            <strong>Jurisdiction:</strong> Any disputes arising from these terms shall be subject to the exclusive
            jurisdiction of the courts of the Netherlands.
          </p>
          <p>
            <strong>EU compliance:</strong> rbee is designed with EU regulations in mind, including GDPR. However, as
            self-hosted software, compliance is ultimately your responsibility.
          </p>
        </div>
      ),
      category: 'Dispute Resolution',
    },
    {
      value: 'dispute-resolution',
      question: '13. Dispute Resolution',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong>Informal resolution:</strong> Before initiating formal proceedings, please contact us at{' '}
            <a href="mailto:legal@rbee.dev" className="text-primary hover:underline">
              legal@rbee.dev
            </a>{' '}
            to attempt informal resolution.
          </p>
          <p>
            <strong>Arbitration:</strong> Not applicable. Disputes will be resolved through the courts of the
            Netherlands.
          </p>
          <p>
            <strong>Class action waiver:</strong> Not applicable under EU law.
          </p>
        </div>
      ),
      category: 'Dispute Resolution',
    },
    {
      value: 'changes-to-terms',
      question: '14. Changes to Terms',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            We reserve the right to modify these Terms of Service at any time. Changes will be effective immediately
            upon posting to this page.
          </p>
          <p>
            <strong>Notification:</strong> Material changes will be announced via:
          </p>
          <ul>
            <li>GitHub repository announcements</li>
            <li>Discord community server</li>
            <li>Email to registered community members (if applicable)</li>
          </ul>
          <p>
            <strong>Your acceptance:</strong> Continued use of rbee after changes constitutes acceptance of the modified
            terms. If you do not agree, you must stop using the Service.
          </p>
        </div>
      ),
      category: 'General',
    },
    {
      value: 'contact-information',
      question: '15. Contact Information',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2">
          <p>
            <strong>Legal inquiries:</strong>
          </p>
          <ul>
            <li>
              Email:{' '}
              <a href="mailto:legal@rbee.dev" className="text-primary hover:underline">
                legal@rbee.dev
              </a>
            </li>
            <li>
              GitHub:{' '}
              <a
                href="https://github.com/veighnsche/llama-orch"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                veighnsche/llama-orch
              </a>
            </li>
          </ul>
          <p>
            <strong>Notice procedures:</strong> Legal notices must be sent to{' '}
            <a href="mailto:legal@rbee.dev" className="text-primary hover:underline">
              legal@rbee.dev
            </a>{' '}
            with "LEGAL NOTICE" in the subject line.
          </p>
          <p>
            <strong>Response time:</strong> We aim to respond to legal inquiries within 5 business days.
          </p>
        </div>
      ),
      category: 'General',
    },
    {
      value: 'severability',
      question: '16. Severability',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
          <p>
            If any provision of these Terms of Service is found to be unenforceable or invalid, that provision will be
            limited or eliminated to the minimum extent necessary so that these terms will otherwise remain in full
            force and effect.
          </p>
        </div>
      ),
      category: 'General',
    },
    {
      value: 'entire-agreement',
      question: '17. Entire Agreement',
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
          <p>
            These Terms of Service, together with the Privacy Policy and GPL-3.0-or-later license, constitute the
            entire agreement between you and the rbee project regarding your use of the Service.
          </p>
          <p>
            These terms supersede all prior agreements, understandings, and communications, whether written or oral.
          </p>
        </div>
      ),
      category: 'General',
    },
  ],
  supportCard: {
    title: 'Need Legal Help?',
    links: [
      { label: 'Privacy Policy', href: '/legal/privacy' },
      { label: 'GPL-3.0 License', href: 'https://www.gnu.org/licenses/gpl-3.0.html' },
      { label: 'GitHub Repository', href: 'https://github.com/veighnsche/llama-orch' },
    ],
    cta: {
      label: 'Contact Legal Team',
      href: 'mailto:legal@rbee.dev',
    },
  },
  jsonLdEnabled: true,
}

/**
 * FAQ container props
 */
export const termsFaqContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * CTA template content - Contact legal team
 */
export const termsCtaProps: CTATemplateProps = {
  eyebrow: 'Questions?',
  title: 'Need clarification on these terms?',
  subtitle: 'Our legal team is here to help. Reach out with any questions or concerns.',
  primary: {
    label: 'Contact Legal Team',
    href: 'mailto:legal@rbee.dev',
    iconRight: ArrowRight,
  },
  secondary: {
    label: 'View Documentation',
    href: '/docs',
    iconLeft: BookOpen,
    variant: 'outline',
  },
  note: 'Typical response time: 5 business days',
  emphasis: 'none',
}

/**
 * CTA container props
 */
export const termsCtaContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}
