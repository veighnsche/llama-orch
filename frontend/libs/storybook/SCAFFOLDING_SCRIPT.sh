#!/bin/bash
# Created by: TEAM-FE-000
# Script to create scaffolding for all atoms, molecules, and organisms

STORYBOOK_DIR="/home/vince/Projects/llama-orch/frontend/libs/storybook/stories"

# Priority 1 Atoms (Core UI - 20 components)
ATOMS_P1=(
  "Input"
  "Textarea"
  "Label"
  "Checkbox"
  "RadioGroup"
  "Switch"
  "Slider"
  "Avatar"
  "Separator"
  "Spinner"
  "Skeleton"
  "Progress"
  "Kbd"
  "Card"
  "Alert"
  "Toast"
  "Dialog"
  "Tooltip"
)

# Priority 2 Atoms (Advanced UI - 25 components)
ATOMS_P2=(
  "DropdownMenu"
  "ContextMenu"
  "Menubar"
  "NavigationMenu"
  "Select"
  "Command"
  "Tabs"
  "Breadcrumb"
  "Pagination"
  "Sheet"
  "Popover"
  "HoverCard"
  "AlertDialog"
  "Accordion"
  "Collapsible"
  "Toggle"
  "ToggleGroup"
  "AspectRatio"
  "ScrollArea"
  "Resizable"
  "Table"
  "Calendar"
  "Chart"
)

# Priority 3 Atoms (Specialized - 15 components)
ATOMS_P3=(
  "Form"
  "Field"
  "InputGroup"
  "InputOTP"
  "Sidebar"
  "Empty"
  "Item"
  "ButtonGroup"
)

# Molecules (15 components)
MOLECULES=(
  "FormField"
  "SearchBar"
  "PasswordInput"
  "NavItem"
  "BreadcrumbItem"
  "StatCard"
  "FeatureCard"
  "TestimonialCard"
  "PricingCard"
  "ImageWithCaption"
  "ConfirmDialog"
  "DropdownAction"
  "TabPanel"
  "AccordionItem"
)

# Organisms - Navigation
ORGANISMS_NAV=(
  "Navigation"
  "Footer"
)

# Organisms - Home Page
ORGANISMS_HOME=(
  "HeroSection"
  "WhatIsRbee"
  "AudienceSelector"
  "EmailCapture"
  "ProblemSection"
  "SolutionSection"
  "HowItWorksSection"
  "FeaturesSection"
  "UseCasesSection"
  "ComparisonSection"
  "PricingSection"
  "SocialProofSection"
  "TechnicalSection"
  "FAQSection"
  "CTASection"
)

# Organisms - Developers Page
ORGANISMS_DEVELOPERS=(
  "DevelopersHero"
  "DevelopersProblem"
  "DevelopersSolution"
  "DevelopersHowItWorks"
  "DevelopersFeatures"
  "DevelopersCodeExamples"
  "DevelopersUseCases"
  "DevelopersPricing"
  "DevelopersTestimonials"
  "DevelopersCTA"
)

# Organisms - Enterprise Page
ORGANISMS_ENTERPRISE=(
  "EnterpriseHero"
  "EnterpriseProblem"
  "EnterpriseSolution"
  "EnterpriseHowItWorks"
  "EnterpriseFeatures"
  "EnterpriseSecurity"
  "EnterpriseCompliance"
  "EnterpriseComparison"
  "EnterpriseUseCases"
  "EnterpriseTestimonials"
  "EnterpriseCTA"
)

# Organisms - GPU Providers Page
ORGANISMS_PROVIDERS=(
  "ProvidersHero"
  "ProvidersProblem"
  "ProvidersSolution"
  "ProvidersHowItWorks"
  "ProvidersFeatures"
  "ProvidersMarketplace"
  "ProvidersEarnings"
  "ProvidersSecurity"
  "ProvidersUseCases"
  "ProvidersTestimonials"
  "ProvidersCTA"
)

# Organisms - Features Page
ORGANISMS_FEATURES=(
  "FeaturesHero"
  "CoreFeaturesTabs"
  "MultiBackendGPU"
  "CrossNodeOrchestration"
  "IntelligentModelManagement"
  "RealTimeProgress"
  "ErrorHandling"
  "SecurityIsolation"
  "AdditionalFeaturesGrid"
)

# Function to create component files
create_component() {
  local type=$1
  local name=$2
  local dir="$STORYBOOK_DIR/$type/$name"
  
  mkdir -p "$dir"
  
  # Create .vue component file
  cat > "$dir/$name.vue" << 'EOF'
<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TODO: Implement this component -->
<script setup lang="ts">
// TODO: Define props interface
// TODO: Import dependencies
</script>

<template>
  <div class="TODO-component-name">
    <!-- TODO: Implement component -->
    <p>TODO: Implement COMPONENT_NAME</p>
  </div>
</template>

<style scoped>
/* TODO: Add component styles */
</style>
EOF

  # Replace COMPONENT_NAME placeholder
  sed -i "s/COMPONENT_NAME/$name/g" "$dir/$name.vue"
  sed -i "s/TODO-component-name/$(echo $name | sed 's/\([A-Z]\)/-\L\1/g' | sed 's/^-//')/g" "$dir/$name.vue"
  
  # Create .story.ts file
  cat > "$dir/$name.story.ts" << 'EOF'
// Created by: TEAM-FE-000 (Scaffolding)
// TODO: Implement story for COMPONENT_NAME

import COMPONENT_NAME from './COMPONENT_NAME.vue'

export default {
  title: 'TYPE/COMPONENT_NAME',
  component: COMPONENT_NAME,
}

export const Default = () => ({
  components: { COMPONENT_NAME },
  template: '<COMPONENT_NAME />',
})

// TODO: Add more story variants
EOF

  # Replace placeholders
  sed -i "s/COMPONENT_NAME/$name/g" "$dir/$name.story.ts"
  sed -i "s/TYPE/$type/g" "$dir/$name.story.ts"
  
  echo "✅ Created $type/$name"
}

# Create all atoms
echo "Creating Priority 1 Atoms..."
for component in "${ATOMS_P1[@]}"; do
  create_component "atoms" "$component"
done

echo "Creating Priority 2 Atoms..."
for component in "${ATOMS_P2[@]}"; do
  create_component "atoms" "$component"
done

echo "Creating Priority 3 Atoms..."
for component in "${ATOMS_P3[@]}"; do
  create_component "atoms" "$component"
done

# Create all molecules
echo "Creating Molecules..."
for component in "${MOLECULES[@]}"; do
  create_component "molecules" "$component"
done

# Create all organisms
echo "Creating Navigation Organisms..."
for component in "${ORGANISMS_NAV[@]}"; do
  create_component "organisms" "$component"
done

echo "Creating Home Page Organisms..."
for component in "${ORGANISMS_HOME[@]}"; do
  create_component "organisms" "$component"
done

echo "Creating Developers Page Organisms..."
for component in "${ORGANISMS_DEVELOPERS[@]}"; do
  create_component "organisms" "$component"
done

echo "Creating Enterprise Page Organisms..."
for component in "${ORGANISMS_ENTERPRISE[@]}"; do
  create_component "organisms" "$component"
done

echo "Creating Providers Page Organisms..."
for component in "${ORGANISMS_PROVIDERS[@]}"; do
  create_component "organisms" "$component"
done

echo "Creating Features Page Organisms..."
for component in "${ORGANISMS_FEATURES[@]}"; do
  create_component "organisms" "$component"
done

echo ""
echo "✅ Scaffolding complete!"
echo ""
echo "Summary:"
echo "- Atoms: $(ls -1 $STORYBOOK_DIR/atoms | wc -l) components"
echo "- Molecules: $(ls -1 $STORYBOOK_DIR/molecules | wc -l) components"
echo "- Organisms: $(ls -1 $STORYBOOK_DIR/organisms | wc -l) components"
echo ""
echo "Next: Run 'pnpm story:dev' to see all components in Histoire"
