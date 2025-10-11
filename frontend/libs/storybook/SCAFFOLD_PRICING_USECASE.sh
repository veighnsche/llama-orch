#!/bin/bash
# Created by: TEAM-FE-000
# Scaffold the 7 missing Pricing and Use Cases components

COMPONENTS=(
  "PricingHero"
  "PricingTiers"
  "PricingComparisonTable"
  "PricingFAQ"
  "UseCasesHero"
  "UseCasesGrid"
  "IndustryUseCases"
)

for component in "${COMPONENTS[@]}"; do
  DIR="/home/vince/Projects/llama-orch/frontend/libs/storybook/stories/organisms/$component"
  
  # Create .vue file
  cat > "$DIR/$component.vue" << 'EOFVUE'
<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TODO: Port from /frontend/reference/v0/app/pricing/page.tsx or use-cases/page.tsx -->
<script setup lang="ts">
// TODO: Define props interface
// TODO: Import dependencies
</script>

<template>
  <section class="COMPONENT_CLASS">
    <!-- TODO: Implement component -->
    <div class="container mx-auto px-4">
      <p>TODO: Implement COMPONENT_NAME</p>
    </div>
  </section>
</template>

<style scoped>
/* TODO: Add component styles */
</style>
EOFVUE

  # Replace placeholders
  sed -i "s/COMPONENT_NAME/$component/g" "$DIR/$component.vue"
  sed -i "s/COMPONENT_CLASS/$(echo $component | sed 's/\([A-Z]\)/-\L\1/g' | sed 's/^-//')/g" "$DIR/$component.vue"
  
  # Create .story.ts file
  cat > "$DIR/$component.story.ts" << 'EOFSTORY'
// Created by: TEAM-FE-000 (Scaffolding)
// TODO: Implement story for COMPONENT_NAME
// Port from: /frontend/reference/v0/app/pricing/page.tsx or use-cases/page.tsx

import COMPONENT_NAME from './COMPONENT_NAME.vue'

export default {
  title: 'organisms/COMPONENT_NAME',
  component: COMPONENT_NAME,
}

export const Default = () => ({
  components: { COMPONENT_NAME },
  template: '<COMPONENT_NAME />',
})

// TODO: Add more story variants
EOFSTORY

  # Replace placeholders
  sed -i "s/COMPONENT_NAME/$component/g" "$DIR/$component.story.ts"
  
  echo "✅ Created organisms/$component"
done

echo ""
echo "✅ All 7 Pricing and Use Cases components scaffolded!"
