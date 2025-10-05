# VIBE CHECK
A consolidated reference for the Vibe Coding Standard, Vibe Check Certificate, Project Governance, and Guardrails.
This document consolidates and supersedes the following files:
- `docs/VIBE_CODING.md`
- `docs/VIBE_CHECK_CERTIFICATE.md`
- `docs/GOVERNANCE.md`
- `docs/WHY_LLMS_ARE_STUPID.md`
This document also serves as the binding operating rules for any IDE or CI assistant acting within this repository. Any attempt by an assistant to propose policy or license changes to pass tests, weaken enforcement (e.g., fail-on-skip), or otherwise manufacture green results is a violation of this policy and grounds for rejection and certification failure.
## Table of Contents
- [The Vibe Coding Standard](#the-vibe-coding-standard)
- [Assistant Operating Rules](#assistant-operating-rules)
- [Vibe Check Certificate](#vibe-check-certificate)
- [Project Governance](#project-governance)
- [Guardrails for LLM-assisted workflows (WHY_LLMS_ARE_STUPID)](#guardrails-for-llm-assisted-workflows-why_llms_are_stupid)
---
## The Vibe Coding Standard
This document defines the Vibe Coding standard, a development and licensing philosophy centered on a single, powerful idea: **for software that matters, the observable proof of its behavior is more important than the source code that generates it.**
Traditional open source focuses on the freedom to view and modify code, but it does not guarantee that the code behaves as claimed. Vibe Coding shifts the focus to verifiable proof, ensuring you can trust what the code *does*.
### The Core Principle: Proof > Code
In a standard software project, the source code is the ultimate truth. But code is just a set of instructions. Its actual behavior can be influenced by dependencies, build environments, and runtime conditions. This gap between what the code *says* and what the software *does* is where bugs and security vulnerabilities hide.
Vibe Coding closes this gap by elevating the **Proof Bundle** to a first-class citizen of the project. The source code's primary purpose becomes generating the proof.
### The Proof Bundle Specification
The Proof Bundle is a comprehensive, verifiable collection of evidence that demonstrates the software's behavior for a specific revision. To be compliant, a Proof Bundle must contain:
1. **Test Evidence**: Complete reports from unit, integration, and end-to-end test suites.
2. **Structured Logs**: Detailed logs from all test runs, captured at multiple verbosity levels.
3. **Visual Evidence**: Screenshots and/or videos of key user flows, providing visual confirmation of the application's state and behavior.
4. **Performance & Coverage Metrics**: Reports on code coverage and basic performance benchmarks.
5. **Security Scans**: Evidence of dependency vulnerability scans and other security checks.
6. **Provenance Record (`provenance.json`)**: A machine-readable file containing cryptographically signed records that bind the bundle to a specific code revision, build environment, and timestamp.
7. **Artifact Manifest (`manifest.sha256`)**: A manifest containing the checksums (e.g., SHA256) of every other file within the bundle. This allows for independent verification that the evidence has not been altered.
This bundle allows any developer, reviewer, or end user to inspect the evidence and verify the software's claims without having to run the code themselves.
### The Role of the Vibe Copyleft License
The [Vibe Copyleft License](../../LICENSE) is the legal instrument designed to enforce this standard. Its key conditions are:
- **Proof is Mandatory**: You cannot distribute the software without its corresponding, complete Proof Bundle.
- **Remixes are Copylefted**: Any derivative work must also be released under the Vibe Copyleft license, including its own complete Proof Bundle.
- **Proof Integrity**: The proof must reflect the software's actual production behavior. Test-only hardening or modifications that would make the software appear more secure or reliable than it is are explicitly forbidden.
By ensuring that proof travels with the code, Vibe Coding creates an ecosystem of transparent, verifiable, and trustworthy software.
### Anti-False Positives in Testing
False positives in testing are expressly prohibited under the Vibe Coding Standard. Tests must reflect the product’s real behavior without masking, shortcuts, or conditional bypasses.
Definition:
- Supported Scope: The set of environments, platforms, configurations, and features explicitly declared as supported in the project's documentation.
- If no scope is declared, the default Supported Scope is all shipped tests and documented targets.
Principles:
- No masking: Do not pre-create state, artifacts, or symlinks that make tests pass. The product under test must create and manage its own state; the harness only observes.
- No harness mutation of product-owned artifacts: The test harness must not alter artifacts owned by the product (e.g., system symlinks, managed directories). Tests that rely on such mutations are invalid.
- Fail-on-skip is the default: Within the declared Supported Scope, any skipped test is a failure. If no scope is declared, all shipped tests and documented targets are in-scope by default. Do not introduce feature flags to reframe this policy.
- Verify sources of truth: Assertions must validate the product’s own outputs, logs, and observable external effects. Avoid indirect proxies that can hide defects.
- Fix infrastructure, don’t paper over it: Image or environment gaps must be fixed at the source, not hidden by conditional logic or test gating.
Reviewer checklist (required for PRs that touch tests, harnesses, or documentation):
- Are any tests pre-creating artifacts or state that the product should create?
- Are there new skips or gates? If yes, they must be justified and will fail the run by default within the declared Supported Scope. If no scope is declared, all shipped tests and documented targets are in-scope by default.
- Do assertions check the product’s real behavior rather than a mocked or indirect signal?
- Are policy changes clearly documented and approved by the project owner?
Enforcement:
- Any discovered false positive invalidates the affected proof and requires immediate remediation.
- Under the Vibe Check Certificate, any false positive is grounds for failure or revocation until fixed and re-verified.
---
## Assistant Operating Rules
These rules are mandatory for any automated assistant (IDE, CI, or chatbot) contributing to this repository.
### Non-negotiable constraints
- Do not propose changes to the license, policies, or enforcement semantics to make tests pass.
- Fail-on-skip is mandatory within the Supported Scope. Do not introduce, suggest, or rely on feature flags or modes that relax this.
- The test harness must not mutate artifacts owned by the product under test.
- Assertions must target product-generated artifacts, logs, and observable effects, not proxies.
### Prohibited suggestions
- Any change that reframes or downgrades the Anti-False Positives policy.
- Adding skips, gates, or conditional logic to bypass failing behavior within Supported Scope.
- Editing certification text or governance to justify otherwise failing tests.
- Pre-creating files, symlinks, or state that the product itself should create.
### When blocked
If tests fail and you cannot proceed without breaching policy:
1. Stop and surface the exact failing evidence (logs, artifacts, diffs, screenshots).
2. Propose fixes only within product code, configuration, or infrastructure images to address root causes.
3. If environment gaps exist, propose changes to the base setup rather than conditional workarounds.
### Review and evidence expectations
- Link to concrete artifacts (paths in this repo, logs under , or build output) for every claim.
- Prefer minimal, evidence-based changes with explicit justification.
### Enforcement
- Violations invalidate the affected proof and will be rejected in review.
- Repeated violations may trigger Vibe-Certification revocation until remediated and re-verified.
### Assistant acknowledgement template
Assistants should include the following acknowledgement in PR descriptions or major doc/test changes:
> Acknowledgement: This change adheres to VIBE_CHECK Assistant Operating Rules. No policy/license changes were proposed. No test harness mutation of product-owned artifacts. Fail-on-skip remains enforced. Evidence for behavior is provided via logs/artifacts linked in this PR.
---
## Vibe Check Certificate
This section outlines the **Vibe Check Certificate**, a tiered certification system for projects under the Vibe Copyleft license. The certification levels—Bronze, Silver, Gold, and Platinum—reflect the degree of compliance with the Vibe Coding Standard, the completeness of the Proof Bundle, and the commitment to transparency and verifiability.
### Certification Levels
#### Bronze Level
- **Basic Compliance**: The project meets the minimum requirements of the Vibe Copyleft license.
- **Proof Bundle**: Includes basic test evidence and a provenance record.
- **Transparency**: The Proof Bundle is available, though it may lack comprehensive visual evidence or detailed metrics.
- **Community Review**: Limited or no formal review process beyond the availability of the Proof Bundle.
**Purpose**: Bronze certification indicates a foundational commitment to the Vibe Coding philosophy, suitable for early-stage projects or those with minimal resources for proof generation.
#### Silver Level
- **Enhanced Compliance**: The project exceeds basic requirements with a more robust Proof Bundle.
- **Proof Bundle**: Includes detailed test evidence, structured logs, and basic visual evidence (e.g., screenshots of key user flows).
- **Transparency**: The Proof Bundle is well-documented and accessible, with a manifest for verification.
- **Community Review**: Evidence of some community or third-party review of the Proof Bundle.
**Purpose**: Silver certification reflects a stronger dedication to verifiable proof, suitable for projects aiming to build trust with a broader audience.
#### Gold Level
- **Advanced Compliance**: The project demonstrates a high level of adherence to the Vibe Coding Standard.
- **Proof Bundle**: Comprehensive, including full test suites (unit, integration, end-to-end), detailed logs at multiple verbosity levels, extensive visual evidence (screenshots and videos), performance and coverage metrics, and basic security scans.
- **Transparency**: The Proof Bundle is meticulously organized, with checksums and provenance records ensuring integrity.
- **Community Review**: Significant community engagement and review, with public feedback incorporated into the project’s governance.
**Purpose**: Gold certification signifies a mature project with a strong focus on transparency and verifiability, ideal for critical software components.
#### Platinum Level
- **Exemplary Compliance**: The project sets the highest standard for Vibe-Certified software.
- **Proof Bundle**: Exhaustive and exemplary, covering all aspects of the Vibe Coding Standard—full test evidence, comprehensive logs, rich visual evidence, advanced performance and coverage metrics, thorough security scans, and detailed provenance records.
- **Transparency**: The Proof Bundle is a model of clarity and accessibility, serving as a reference for other projects. It includes interactive elements for reviewers (e.g., navigable HTML summaries).
- **Community Review**: Extensive and continuous review by a diverse community, with a transparent issue resolution process and a history of addressing proof flaws promptly.
- **Reproducibility**: The project provides detailed instructions and tools for fully reproducing the Proof Bundle from scratch in a clean environment.
**Purpose**: Platinum certification is reserved for flagship projects that exemplify the Vibe Coding philosophy, providing the highest level of trust and verifiability for mission-critical software.
### Certification Process
1. **Self-Assessment**: Project maintainers evaluate their compliance with the Vibe Coding Standard and the completeness of their Proof Bundle against the criteria for each level.
2. **Submission**: Projects submit their Proof Bundle and a certification request to the Vibe-Certified review body (as outlined in the [Governance Model](#project-governance)).
3. **Review**: The review body assesses the submission, focusing on the Proof Bundle’s integrity, adherence to the Zero False-Positives Requirement, and the project’s compliance with the certification criteria.
4. **Award**: Upon successful review, the project is awarded the appropriate certification level and receives a badge for display in their repository (e.g., "Vibe-Certified Bronze").
5. **Maintenance**: Certification is subject to periodic re-evaluation to ensure ongoing compliance, especially after major updates or reported issues.
### Zero False-Positives Requirement (All Levels)
Definitions:
- Supported Scope: The set of environments, platforms, configurations, and features explicitly declared as supported in the project's documentation.
- Default: If no scope is declared, the Supported Scope defaults to all shipped tests and documented targets.
Requirements (applies to Bronze, Silver, Gold, and Platinum):
- Any false positive in testing is an automatic certification failure or grounds for revocation until remediated and re-verified.
- Fail-on-skip is mandatory within the Supported Scope. Any skipped test within the Supported Scope constitutes failure. If no scope is declared, any skipped shipped test constitutes failure.
- The harness must not mutate artifacts owned by the product under test. Masking via pre-created artifacts, symlinks, or conditional logic is prohibited.
- Proof Bundles must include: (a) an explicit enumeration of executed tests and environments; (b) a summary of skips with an affirmative statement of zero skips within the Supported Scope; (c) evidence that assertions validate the product’s real behavior (not indirect proxies that mask defects).
### Responsibilities of Certified Projects
- **Maintain Proof Integrity**: Certified projects must continuously update and distribute their Proof Bundle with each release, reflecting the software’s true behavior.
- **Zero False Positives**: Projects must maintain fail-on-skip within the Supported Scope (defaulting to all shipped tests and documented targets if none is declared), avoid masking, and promptly remediate any discovered false positive, triggering re-verification when necessary.
- **Respond to Issues**: Projects must address reported application bugs and proof flaws promptly, as outlined in the [Governance Model](#project-governance).
- **Uphold Transparency**: Certified projects are expected to maintain or exceed the transparency level of their awarded certification.
Failure to meet these responsibilities may result in the revocation of certification status.
### Benefits of Certification
- **Trust**: Certification signals to users and developers that the project adheres to a verifiable standard of trustworthiness.
- **Community**: Higher certification levels foster greater community engagement and collaboration.
- **Visibility**: Certified projects are showcased within the Vibe-Certified ecosystem, increasing their reach and impact.
By striving for higher levels of certification, projects under the Vibe Copyleft license contribute to a culture of transparency and trust in software development.
---
## Project Governance
This section outlines the governance model for a Vibe-Certified project, including the review process, how to handle issues, and the responsibilities of downstream users.
### The Review Process
This project is designed for continuous, transparent review. The primary artifact for review is the **Proof Bundle**, not the source code.
1. **Start with the Proof**: Reviewers should begin at the project's `README.md`, which links directly to the latest Proof Bundle.
2. **Review the Evidence**: The proof is organized to be easily navigable. Reviewers should examine the test results, logs, and visual evidence to verify the application's behavior.
3. **Verify Authenticity**: The Proof Bundle's authenticity can be confirmed by validating the checksums in the `manifest.sha256` file and inspecting the `provenance.json` record.
4. **Reproduce from Scratch**: For the highest level of assurance, reviewers can follow the reproducibility instructions in the `examples/` directory to regenerate the entire Proof Bundle from a clean environment.
### Reporting Issues
Issues in a Vibe-Certified project fall into two categories:
1. **Application Bugs**: A mismatch between the *expected* outcome and the *actual* outcome, as demonstrated by the evidence in the Proof Bundle.
2. **Proof Flaws**: An issue where the proof itself is incomplete, misleading, or fails to accurately represent the application's behavior. **This is considered the more severe category of issue.** A  that fails to render all artifacts (including screenshots and logs) in its primary HTML summary is considered incomplete and misleading.
**How to Report:**
- All issues should be reported via the project's public issue tracker.
- When reporting an issue, please link directly to the specific evidence in the Proof Bundle that demonstrates the discrepancy.
- Reports of proof flaws will be treated with the highest priority.
### Responsibilities of Remixers
Anyone who creates and distributes a derivative work ("Remix") under the Vibe Copyleft license has the following responsibilities:
- **Maintain Proof Integrity**: You must generate and distribute a complete and accurate Proof Bundle for your Remix, compliant with [The Vibe Coding Standard](#the-vibe-coding-standard).
- **Uphold Separation**: You must continue to enforce the strict separation of product code from any code used for verification (e.g., tests, proof-generation scripts).
- **Provide Transparency**: Your users must have access to the same level of transparency and verifiability as is provided by this original project.
Failure to meet these responsibilities is a violation of the license. The goal is to build a chain of trust where every link in the software supply chain is verifiable.
---
## Guardrails for LLM-assisted workflows (WHY_LLMS_ARE_STUPID)
This is a curated, project-relevant list of LLM failure modes and the specific guardrails we enforce in this repository to prevent false positives. It is not a diary and contains no external project references.
### Purpose
Keep tests faithful to the product. Avoid shortcuts that manufacture green results. Uphold the Vibe Coding Standard and the Vibe Check Certificate’s Zero False-Positives Requirement.
### Curated lessons applied to this project
- **Policy is default, not a toggle**
  - Do not invent feature flags or special modes to relax core policies (e.g., fail-on-skip). Defaults are unconditional unless explicitly changed with owner approval.
- **No harness mutation of product-owned artifacts**
  - Tests must observe, not manipulate. Do not pre-create files/symlinks/state that the product should create. Do not alter system state to force passes.
- **Fail on any in-scope skip**
  - Within the Supported Scope (as defined in [The Vibe Coding Standard](#the-vibe-coding-standard)), any skipped test is a failure. If no scope is declared, all shipped tests and documented targets are in-scope by default.
- **Verify the actual execution path**
  - Claims about how tests run must be grounded in code or logs. Do not assert coverage or runner behavior without verifiable evidence.
- **Assert on real outputs, not proxies**
  - Prefer assertions on product-generated artifacts, logs, and observable effects. Avoid indirect proxies that can mask defects.
- **Fix infra at the source**
  - Address environment/image problems in the base setup. Do not add conditional logic or skips to paper over infrastructure issues.
- **Governance first for policy/doc changes**
  - Any change to policy, scope, or certification semantics requires explicit owner approval and documentation updates.
- **PR checklist for test/harness changes**
  - No pre-created state that should come from the product.
  - Any skips are justified and will fail the run within Supported Scope.
  - Assertions tie to real behavior and artifacts.
  - Links to logs/code proving runner paths when relevant.
- **Proof Bundle expectations**
  - Enumerate executed tests/environments; affirm zero skips within Supported Scope. Include evidence that assertions validate real behavior.
### How to use this with the repo
- Treat this list as non-normative guidance that complements the binding policies in [The Vibe Coding Standard](#the-vibe-coding-standard) and the [Vibe Check Certificate](#vibe-check-certificate).
- When uncertain, assume a proposed shortcut risks false positives and seek minimal, evidence-based experiments instead.
