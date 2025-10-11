<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersTestimonials from './DevelopersTestimonials.vue'
</script>

<template>
  <Story title="organisms/DevelopersTestimonials">
    <Variant title="Default">
      <DevelopersTestimonials />
    </Variant>

    <Variant title="Single Testimonial">
      <DevelopersTestimonials
        title="What Developers Say"
        :testimonials="[
          {
            quote: 'Best tool for self-hosted AI. Simple setup, powerful features.',
            author: 'John D.',
            role: 'Full Stack Developer',
            avatar: '👨‍💻',
          },
        ]"
        :stats="[
          { value: '100%', label: 'Open Source', highlighted: true },
          { value: '$0', label: 'Monthly Cost', highlighted: false },
        ]"
      />
    </Variant>
  </Story>
</template>
