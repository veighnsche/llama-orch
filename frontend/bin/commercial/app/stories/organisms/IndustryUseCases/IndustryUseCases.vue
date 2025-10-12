<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-009: Implemented IndustryUseCases -->
<script setup lang="ts">
interface Industry {
  title: string
  description: string
}

interface Props {
  title?: string
  subtitle?: string
  industries?: Industry[]
}

withDefaults(defineProps<Props>(), {
  title: 'Industry-Specific Solutions',
  subtitle: 'rbee adapts to the unique compliance and security requirements of regulated industries.',
  industries: () => [
    {
      title: 'Financial Services',
      description:
        'GDPR compliance, audit trails, data residency controls. AI-powered code review and risk analysis without sending sensitive financial data to external APIs.',
    },
    {
      title: 'Healthcare',
      description:
        'HIPAA-compliant infrastructure. Patient data never leaves your network. AI-assisted medical coding, documentation, and research without privacy concerns.',
    },
    {
      title: 'Legal',
      description:
        'Attorney-client privilege maintained. Document analysis, contract review, and legal research with AI while keeping all client information confidential.',
    },
    {
      title: 'Government',
      description:
        'Sovereign AI infrastructure. No foreign cloud dependencies. Complete audit trails for compliance with government security standards.',
    },
    {
      title: 'Education',
      description:
        'Student data protection. AI-powered tutoring, grading assistance, and research tools without sending student information to external services.',
    },
    {
      title: 'Manufacturing',
      description:
        'Protect trade secrets and proprietary designs. AI-assisted CAD, quality control, and process optimization without exposing intellectual property.',
    },
  ],
})
</script>

<template>
  <section class="py-24 bg-secondary">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
        <p class="text-xl text-muted-foreground leading-relaxed">
          {{ subtitle }}
        </p>
      </div>

      <div class="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <div v-for="(industry, i) in industries" :key="i" class="bg-background border border-border rounded-lg p-6 space-y-4">
          <h3 class="text-lg font-bold text-foreground">{{ industry.title }}</h3>
          <p class="text-muted-foreground text-sm leading-relaxed">
            {{ industry.description }}
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
