<!-- Created by: TEAM-FE-002 -->
<script setup lang="ts">
import Badge from './Badge.vue'
</script>

<template>
  <Story title="atoms/Badge">
    <Variant title="Default">
      <Badge>Default Badge</Badge>
    </Variant>

    <Variant title="Secondary">
      <Badge variant="secondary">Secondary</Badge>
    </Variant>

    <Variant title="Destructive">
      <Badge variant="destructive">Destructive</Badge>
    </Variant>

    <Variant title="Outline">
      <Badge variant="outline">Outline</Badge>
    </Variant>

    <Variant title="Most Popular">
      <Badge class="bg-primary text-primary-foreground">Most Popular</Badge>
    </Variant>
  </Story>
</template>
