<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-DX-004: Added 13 comprehensive button variants (size, color, state, icon, width) -->
<script setup lang="ts">
import Button from './Button.vue'
import { Check, ArrowRight, Download, X } from 'lucide-vue-next'
</script>

<template>
  <Story title="atoms/Button">
    <!-- Size Variants -->
    <Variant title="Small">
      <Button size="sm">Small Button</Button>
    </Variant>
    
    <Variant title="Default Size">
      <Button>Default Button</Button>
    </Variant>
    
    <Variant title="Large">
      <Button size="lg">Large Button</Button>
    </Variant>
    
    <!-- Color/Variant Variants -->
    <Variant title="Primary">
      <Button variant="default">Primary Button</Button>
    </Variant>
    
    <Variant title="Secondary">
      <Button variant="secondary">Secondary Button</Button>
    </Variant>
    
    <Variant title="Destructive">
      <Button variant="destructive">Destructive Button</Button>
    </Variant>
    
    <Variant title="Outline">
      <Button variant="outline">Outline Button</Button>
    </Variant>
    
    <Variant title="Ghost">
      <Button variant="ghost">Ghost Button</Button>
    </Variant>
    
    <Variant title="Link">
      <Button variant="link">Link Button</Button>
    </Variant>
    
    <!-- State Variants -->
    <Variant title="Disabled">
      <Button disabled>Disabled Button</Button>
    </Variant>
    
    <!-- Icon Variants -->
    <Variant title="Icon Left">
      <Button>
        <Check :size="16" />
        With Icon Left
      </Button>
    </Variant>
    
    <Variant title="Icon Right">
      <Button>
        Icon Right
        <ArrowRight :size="16" />
      </Button>
    </Variant>
    
    <Variant title="Icon Only">
      <Button size="icon" aria-label="Close">
        <X :size="16" />
      </Button>
    </Variant>
    
    <!-- Width Variant -->
    <Variant title="Full Width">
      <Button class="w-full">
        <Download :size="16" />
        Full Width Button
      </Button>
    </Variant>
  </Story>
</template>
