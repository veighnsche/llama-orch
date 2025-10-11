<!-- TEAM-FE-004: Created Tabs story -->
<script setup lang="ts">
import { TabsRoot } from 'radix-vue'
import TabsList from './TabsList.vue'
import TabsTrigger from './TabsTrigger.vue'
import TabsContent from './TabsContent.vue'
import { Card } from 'rbee-storybook/stories'
</script>

<template>
  <Story title="atoms/Tabs">
    <Variant title="Default">
      <TabsRoot default-value="tab1" class="w-full max-w-md">
        <TabsList>
          <TabsTrigger value="tab1">Account</TabsTrigger>
          <TabsTrigger value="tab2">Password</TabsTrigger>
          <TabsTrigger value="tab3">Settings</TabsTrigger>
        </TabsList>
        
        <TabsContent value="tab1" class="mt-4">
          <Card class="p-4">
            <h3 class="text-foreground font-semibold mb-2">Account</h3>
            <p class="text-muted-foreground text-sm">
              Make changes to your account here. Click save when you're done.
            </p>
          </Card>
        </TabsContent>
        
        <TabsContent value="tab2" class="mt-4">
          <Card class="p-4">
            <h3 class="text-foreground font-semibold mb-2">Password</h3>
            <p class="text-muted-foreground text-sm">
              Change your password here. After saving, you'll be logged out.
            </p>
          </Card>
        </TabsContent>
        
        <TabsContent value="tab3" class="mt-4">
          <Card class="p-4">
            <h3 class="text-foreground font-semibold mb-2">Settings</h3>
            <p class="text-muted-foreground text-sm">
              Configure your application settings and preferences.
            </p>
          </Card>
        </TabsContent>
      </TabsRoot>
    </Variant>

    <Variant title="Disabled Tab">
      <TabsRoot default-value="tab1" class="w-full max-w-md">
        <TabsList>
          <TabsTrigger value="tab1">Enabled</TabsTrigger>
          <TabsTrigger value="tab2" :disabled="true">Disabled</TabsTrigger>
          <TabsTrigger value="tab3">Also Enabled</TabsTrigger>
        </TabsList>
        
        <TabsContent value="tab1" class="mt-4">
          <Card class="p-4">
            <p class="text-muted-foreground text-sm">First tab content</p>
          </Card>
        </TabsContent>
        
        <TabsContent value="tab2" class="mt-4">
          <Card class="p-4">
            <p class="text-muted-foreground text-sm">This tab is disabled</p>
          </Card>
        </TabsContent>
        
        <TabsContent value="tab3" class="mt-4">
          <Card class="p-4">
            <p class="text-muted-foreground text-sm">Third tab content</p>
          </Card>
        </TabsContent>
      </TabsRoot>
    </Variant>

    <Variant title="Vertical Orientation">
      <TabsRoot default-value="tab1" orientation="vertical" class="flex flex-row gap-4 max-w-2xl">
        <TabsList class="flex-col h-auto">
          <TabsTrigger value="tab1">Overview</TabsTrigger>
          <TabsTrigger value="tab2">Analytics</TabsTrigger>
          <TabsTrigger value="tab3">Reports</TabsTrigger>
        </TabsList>
        
        <div class="flex-1">
          <TabsContent value="tab1">
            <Card class="p-4">
              <h3 class="text-foreground font-semibold mb-2">Overview</h3>
              <p class="text-muted-foreground text-sm">View your dashboard overview</p>
            </Card>
          </TabsContent>
          
          <TabsContent value="tab2">
            <Card class="p-4">
              <h3 class="text-foreground font-semibold mb-2">Analytics</h3>
              <p class="text-muted-foreground text-sm">Detailed analytics and metrics</p>
            </Card>
          </TabsContent>
          
          <TabsContent value="tab3">
            <Card class="p-4">
              <h3 class="text-foreground font-semibold mb-2">Reports</h3>
              <p class="text-muted-foreground text-sm">Generate and view reports</p>
            </Card>
          </TabsContent>
        </div>
      </TabsRoot>
    </Variant>
  </Story>
</template>
