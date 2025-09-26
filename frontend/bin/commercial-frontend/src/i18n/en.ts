// en_2.ts — Positioning: Public & Private AI Taps + Toolkit (no plumbing metaphor)

const en = {
  nav: {
    publicTap: 'Public Tap',
    privateTap: 'Private Tap',
    toolkit: 'Toolkit',
    pricing: 'Pricing',
    proof: 'Proof',
    faqs: 'FAQs',
    about: 'About',
    contact: 'Contact',
  },

  footer: {
    brandLine: 'Orchyra — Public & Private AI Taps (NL/EU)',
    about: 'About',
    proof: 'Proof',
    pricing: 'Pricing',
    contactLegal: 'Contact & Legal',
    microcopy: 'Public Tap Terms: prepaid, non-refundable, 12-month validity',
  },

  a11y: {
    language: 'Language',
    navPrimary: 'Primary',
    footerNav: 'Footer',
    addLink: '(add link)',
    perGpuHour: 'per GPU-hour',
  },

  devbar: {
    aria: 'Developer shortcuts',
    label: 'Developer',
    docs: 'Docs',
    api: 'API Reference',
    github: 'GitHub',
  },

  jsonld: {
    desc: 'Agentic AI APIs with a full toolkit. Prepaid Public Tap and managed Private Taps on EU GPUs. Proof-driven and open source.',
  },

  seoTitle: {
    publicTap: 'Public Tap — Prepaid Agentic API | Orchyra',
    privateTap: 'Private Tap — Managed Dedicated GPUs | Orchyra',
    toolkit: 'Toolkit — SDK, Utils & Deterministic Applets | Orchyra',
    pricing: 'Pricing — Credits & GPU-hour | Orchyra',
    proof: 'Proof — Logs, Metrics, SSE | Orchyra',
    faqs: 'FAQs | Orchyra',
    about: 'About | Orchyra',
    contact: 'Contact & Legal | Orchyra',
  },

  seo: {
    home: [
      'private LLM hosting',
      'public LLM API',
      'agentic API',
      'OpenAI-compatible gateway',
      'AI toolkit',
      'deterministic applets',
      'EU data residency',
      'proof-driven operations',
      'vLLM serving',
      'managed GPUs',
    ],
    publicTap: [
      'public tap',
      'prepaid credits',
      'agentic API',
      'OpenAI adapter',
      'open-source models',
    ],
    privateTap: [
      'private LLM hosting',
      'dedicated GPUs',
      'managed inference',
      'SLA',
      'EU AI Act',
      'OpenAI-compatible',
    ],
    toolkit: [
      'AI toolkit',
      'SDK',
      'utils',
      'ts/js/rs',
      'deterministic applets',
      'code review ai',
      'summarizer ai',
    ],
    pricing: ['pricing', 'prepaid credits', 'GPU-hour', 'A100', 'H100'],
    proof: ['proof', 'prometheus', 'sse', 'metrics', 'observability'],
    faqs: ['faq', 'pricing', 'credits', 'models', 'EU AI Act', 'vLLM'],
    about: ['about', 'independent', 'OSS', 'proof-first'],
    contact: ['contact', 'legal', 'terms', 'credits'],
  },

  seoDesc: {
    home: 'Agentic AI APIs with a full toolkit. Start on the Public Tap, move to a managed Private Tap on EU GPUs. Proofs, logs, and docs included.',
    publicTap:
      'Shared agentic API with prepaid credits. Works with the SDK, utils, and the OpenAI adapter. Start fast.',
    privateTap:
      'Your own managed AI API on dedicated GPUs. EU-friendly, observable, and OpenAI-compatible.',
    toolkit:
      'Toolkit for TS/JS/Rust (Python/Mojo soon): SDK, utils, and deterministic applets to build internal AI tools.',
    pricing:
      'Draft pricing for Public Tap credits and Private Tap GPU-hour snapshots. Subject to change after benchmarking.',
    proof:
      'Proof-first operations: deployment reports, metrics, SSE transcripts, and documentation you can audit.',
    faqs: 'Answers on pricing vs OpenAI, model choice, credits, serving efficiency, and EU AI Act posture.',
    about:
      'Independent EU-based provider of agentic AI APIs and a practical toolkit. Proof-driven and open source.',
    contact:
      'Contact & Legal for Orchyra. Email, LinkedIn, GitHub. Public Tap terms and data/logs availability.',
  },

  home: {
    hero: {
      eyebrow: 'Reliable EU-Based AI APIs',
      h1: 'Transparent, compliant AI infrastructure you can trust.',
      sub: 'Start small with prepaid Public APIs, then grow into your own Private endpoint on dedicated GPUs. Every step is proof-driven, auditable, and cost-predictable — so your business can adopt AI with confidence.',
      ctaMenu: 'Explore service plans',
      ctaProofs: 'See operational proofs',
      quickPublic: 'Try Public API',
      quickPrivate: 'Request Private demo',
      badge: {
        proofs: 'Proof-driven',
        eu: 'EU-compliant',
        predictable: 'Predictable costs',
        oss: 'Open source',
      },
    },

    why: {
      title: 'Why standardize your AI APIs',
      intro: 'Reliable agentic AI needs clear APIs, evidence, and control — not another black box.',
      problemsTitle: 'Without standards, teams hit walls',
      outcomesTitle: 'What you get instead',
      b1: 'Shadow AI and duplicate data flows across tools',
      b2: 'No audit trail: unclear who ran what and when',
      b3: 'Lock-in to models or hosts you can’t swap',
      b4: 'GDPR risk from unmanaged logs and caches',
      o1: 'Reproducible runs with signed logs and job IDs',
      o2: 'EU data residency and policy-controlled retention',
      o3: 'OpenAI-compatible gateway with swappable models',
      o4: 'Operational guardrails: quotas, approvals, rate limits',
      ctaProof: 'Inspect proofs',
      ctaMenu: 'View service menu',
    },

    badge: {
      audit: 'Auditable',
      gdpr: 'GDPR/EU data',
      oss: 'Open source',
    },

    three: {
      title: 'Three pieces, one platform',
      i1: 'Toolkit: SDK, utils, and deterministic applets for IT teams',
      i2: 'Public Tap: prepaid agentic API, start in minutes',
      i3: 'Private Tap: your own managed endpoint on dedicated GPUs',
    },

    public: {
      title: 'Public Tap — test fast, pay upfront',
      p1: 'Buy credits, point your SDK, and ship demos. No invoices, no surprises.',
      b1: 'Works with SDK + utils and OpenAI adapter',
      b2: 'Curated OSS models, SSE streaming, and logs',
      b3: 'Prepaid credits: €50 / €200 / €500 packs (baseline €1.20 per 1M tokens) — Draft',
    },

    private: {
      title: 'Private Tap — managed, on your own GPUs',
      p1: 'When privacy, control, or performance matter, get a dedicated API surface. I provision GPUs, deploy your models, and operate the stack.',
      b1: 'A100 80GB — €1.80 / GPU-hour + €250 / month base fee — Draft',
      b2: 'H100 80GB — €3.80 / GPU-hour + €400 / month base fee — Draft',
      b3: 'Scale 1×/2×/4×/8×; optional OpenAI-compatible gateway',
    },

    toolkit: {
      title: 'Toolkit — build the AI tools you need',
      p1: 'For TS/JS/Rust (Python/Mojo coming soon). Deterministic applets you control, running on my agentic API.',
      b1: 'Codebase maintenance helpers',
      b2: 'Code review assistant',
      b3: 'Summarizer AI and document QA',
      b4: 'Internal chat & automation applets',
      note: 'You control the tools — I provide the API and proofs.',
      cta: 'Explore the Toolkit',
    },

    proof: {
      title: 'Proof you can audit',
      b1: 'Deployment report with SSE transcripts and performance metrics',
      b2: 'Prometheus dashboards and alert thresholds',
      b3: 'Version pinning and rollback plan',
      b4: 'Documentation bundle aligned with EU transparency expectations',
    },

    audience: {
      title: 'Designed for IT teams and agencies — usable by anyone',
      b1: 'IT: private endpoints without hiring a platform team',
      b2: 'Agencies: start on Public Tap, upgrade to Private',
      b3: 'Compliance-sensitive orgs: logs and artifacts for audits',
    },

    more: {
      faqs: 'Read the FAQs',
      about: 'About Vince',
    },
  },

  publicTap: {
    h1: 'Public Tap — Prepaid Agentic API',
    what: 'What it is',
    whatP:
      'A shared API on curated open-source models for fast experiments and demos. Works with the SDK, utils, and OpenAI adapter.',
    who: "Who it's for",
    why: 'Why it matters',
    terms: 'Terms',
    pricing: 'Draft pricing',
    bWho1: 'Developers and startups testing ideas',
    bWho2: 'Agencies building demos and POCs',
    bWho3: 'IT teams evaluating APIs and SDK fit',
    bWhy1: 'Predictable costs with prepaid credits',
    bWhy2: 'Low-friction onboarding; start in minutes',
    bWhy3: 'Aligned with open-source transparency',
    bTerms1: 'Credits are non-refundable',
    bTerms2: '12-month validity',
    bPrice1: 'Baseline: €1.20 per 1M tokens (input + output combined)',
    bPrice2: 'Starter: €50 → ~41M tokens',
    bPrice3: 'Builder: €200 → ~166M tokens',
    bPrice4: 'Pro: €500 → ~416M tokens',
    note: 'Draft — numbers subject to revision after provider benchmarking.',
  },

  privateTap: {
    h1: 'Private Tap — Managed Dedicated GPUs',
    what: 'What it is',
    whatP:
      'Your own isolated agentic API box, provisioned on dedicated GPUs. Bring any OSS model (e.g., from Hugging Face). Optional OpenAI-compatible gateway.',
    value: 'Value',
    bVal1: 'Privacy & control with your own endpoint and quotas',
    bVal2: 'Observability with metrics and logs for real SLOs',
    bVal3: 'High-throughput serving (e.g., vLLM) engineered for production',
    pricing: 'Draft pricing (snapshot)',
    bPrice1: 'A100 80GB — €1.80 / GPU-hour + €250 / month base fee',
    bPrice2: 'H100 80GB — €3.80 / GPU-hour + €400 / month base fee',
    bPrice3: 'Scale 1×/2×/4×/8× GPUs',
    note: 'Draft — examples based on public provider rates; subject to change.',
  },

  toolkit: {
    h1: 'Toolkit — SDK, Utils & Deterministic Applets',
    what: 'What it is',
    whatP:
      'A practical toolkit for TS/JS/Rust (Python/Mojo soon) with deterministic applets. Use it to build and control internal AI tools that run on the agentic API.',
    lang: 'Languages',
    bLang1: 'TypeScript / JavaScript',
    bLang2: 'Rust',
    bLang3: 'Python & Mojo (coming soon)',
    applets: 'Included applets (examples)',
    bApp1: 'Codebase maintenance helpers',
    bApp2: 'Code review assistant',
    bApp3: 'Summarizer AI (docs, tickets, chats)',
    bApp4: 'Document QA and retrieval',
    bApp5: 'Internal chat & automation bots',
    control: 'Control',
    bCtl1: 'Deterministic applets with predictable behavior',
    bCtl2: 'Your policies, quotas, and approvals',
    bCtl3: 'Works with OpenAI adapter when needed',
    cta: 'View Toolkit docs',
    note: 'Open source core; build your own applets or extend mine.',
  },

  pricing: {
    h1: 'Pricing (Draft)',
    credits: 'Public Tap — Credits',
    gpu: 'Private Tap — GPU-hour',
    note1: 'Draft — subject to change after benchmarking.',
    note2: 'Provider rates vary by region and availability.',
  },

  proof: {
    h1: 'Proof — Logs, Metrics, SSE',
    artifacts: 'Artifacts',
    b1: 'Deployment report with SSE transcripts and performance metrics',
    b2: 'Prometheus dashboard snapshots and alert thresholds',
    b3: 'Version pinning and rollback plan',
    b4: 'Documentation bundle for internal governance',
    visuals: 'Coming visuals',
    v1: 'Grafana/Prometheus screenshots',
    v2: 'Pipeline diagram with “certified” stamp',
    microcopy: 'This is infrastructure. You shouldn’t have to trust it — you should see it.',
  },

  faqs: {
    h1: 'Frequently Asked Questions',
    q1: 'Is this cheaper than OpenAI?',
    a1: 'No. If minimum token price is your only goal, use OpenAI. If you want transparency, EU-friendly deployments, model choice, and dedicated GPUs, consider a Private Tap. For testing, start with the Public Tap.',
    q2: 'Can I bring any OSS model?',
    a2: 'Yes. For Private Taps you can choose models from Hugging Face or other sources; VRAM is validated during setup.',
    q3: 'How do prepaid credits work?',
    a3: 'Buy a pack, receive a token balance, and consume until zero. Credits are non-refundable and valid for 12 months. Your balance is visible via dashboard/API.',
    q4: 'What makes your serving stack efficient?',
    a4: 'We use high-throughput serving engines (e.g., vLLM with PagedAttention and continuous batching) and expose metrics for real SLOs.',
    q5: 'Are you EU AI Act ready?',
    a5: 'We operate with the Act’s transparency spirit: logged, documented deployments and artifacts that support your internal governance.',
  },

  about: {
    h1: 'About Orchyra',
    identity: 'Identity',
    identityP:
      'Independent, EU-based provider of agentic AI APIs and a practical toolkit. Focused on robust engineering and proof-first operations.',
    usp: 'What makes us different',
    usp1: 'Open source core — orchestrator, SDK, and utils',
    usp2: 'Proof-first operations — logs, metrics, SSE transcripts, version pinning',
    usp3: 'Independent & local — personal, accountable, practical',
    usp4: 'Simple commercial model — credits and GPU-hour packs',
    approach: 'Approach',
    approachP:
      'Standardized APIs, observable operations, and a toolkit of deterministic applets your team controls.',
    cta: 'Talk to Vince',
  },

  contact: {
    h1: 'Contact & Legal',
    contact: 'Contact',
    email: 'Email',
    linkedin: 'LinkedIn',
    github: 'GitHub',
    legal: 'Public Tap Terms',
    l1: 'Prepaid credits are non-refundable',
    l2: '12-month validity from purchase',
    l3: 'Service halts at zero balance until recharge',
    note: 'Legal copy to be finalized; this page is informational only.',
    dataLogs: 'Data & Logs',
    dataLogsP:
      'Deployment reports, metrics, and relevant logs are available to customers as part of the proof-first approach.',
  },
}

export default en
