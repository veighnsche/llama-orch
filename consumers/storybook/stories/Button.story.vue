<script setup lang="ts">
import { reactive, computed } from 'vue'
import Button from './Button.vue'

type AsType = 'button' | 'a' | 'router-link' | undefined

const state = reactive({
  label: 'Click me',
  as: 'button' as AsType,
  to: '/about' as string,
  href: 'https://example.com',
  type: 'button' as 'button' | 'submit' | 'reset',
  variant: 'primary' as 'primary' | 'ghost' | 'link',
  size: 'md' as 'sm' | 'md' | 'lg',
  iconOnly: false,
  disabled: false,
  block: false,
})

const effectiveContent = computed(() => (state.iconOnly ? '★' : state.label))

const propsForButton = computed(() => ({
  as: state.as,
  to: state.as === 'router-link' ? state.to : undefined,
  href: state.as === 'a' ? state.href : undefined,
  type: state.as === 'button' || !state.as ? state.type : undefined,
  variant: state.variant,
  size: state.size,
  iconOnly: state.iconOnly,
  disabled: state.disabled,
  block: state.block,
}))
</script>

<template>
  <Story title="UI/Button" :layout="{ type: 'grid', width: 280 }">
    <!-- Playground with controls -->
    <Variant title="Playground">
      <div style="padding: 8px">
        <Button v-bind="propsForButton">
          {{ effectiveContent }}
        </Button>
      </div>
    </Variant>

    <!-- Variant presets: by variant -->
    <Variant title="Primary">
      <Button variant="primary"> Primary </Button>
    </Variant>
    <Variant title="Ghost">
      <Button variant="ghost"> Ghost </Button>
    </Variant>
    <Variant title="Link">
      <Button variant="link"> Link </Button>
    </Variant>

    <!-- Sizes -->
    <Variant title="Size: sm">
      <Button size="sm"> Small </Button>
    </Variant>
    <Variant title="Size: md">
      <Button size="md"> Medium </Button>
    </Variant>
    <Variant title="Size: lg">
      <Button size="lg"> Large </Button>
    </Variant>

    <!-- Icon only -->
    <Variant title="Icon sm">
      <Button size="sm" icon-only> ★ </Button>
    </Variant>
    <Variant title="Icon md">
      <Button size="md" icon-only> ★ </Button>
    </Variant>
    <Variant title="Icon lg">
      <Button size="lg" icon-only> ★ </Button>
    </Variant>

    <!-- Block -->
    <Variant title="Block">
      <div style="width: 340px">
        <Button block> Block button </Button>
      </div>
    </Variant>

    <!-- Disabled -->
    <Variant title="Disabled">
      <Button disabled> Disabled </Button>
    </Variant>

    <!-- As anchor and router-link -->
    <Variant title="Anchor">
      <Button as="a" href="https://example.com"> Anchor link </Button>
    </Variant>
    <Variant title="RouterLink">
      <Button as="router-link" :to="'/about'"> Router link </Button>
    </Variant>

    <template #controls>
      <HstText v-model="state.label" title="label" />
      <HstSelect
        v-model="state.as"
        title="as"
        :options="[{ value: undefined, label: 'auto' }, 'button', 'a', 'router-link']"
      />
      <HstSelect v-model="state.type" title="type" :options="['button', 'submit', 'reset']" />
      <HstText v-model="state.href" title="href (when as = 'a')" />
      <HstSelect
        v-model="state.to"
        title="to (when as = 'router-link')"
        :options="['/', '/about', '/contact']"
      />
      <HstSelect v-model="state.variant" title="variant" :options="['primary', 'ghost', 'link']" />
      <HstSelect v-model="state.size" title="size" :options="['sm', 'md', 'lg']" />
      <HstCheckbox v-model="state.iconOnly" title="iconOnly" />
      <HstCheckbox v-model="state.disabled" title="disabled" />
      <HstCheckbox v-model="state.block" title="block" />
    </template>
  </Story>
</template>
