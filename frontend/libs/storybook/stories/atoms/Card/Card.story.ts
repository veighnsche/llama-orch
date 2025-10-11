// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented Card story with all compositions

import Card from './Card.vue'
import CardHeader from './CardHeader.vue'
import CardTitle from './CardTitle.vue'
import CardDescription from './CardDescription.vue'
import CardContent from './CardContent.vue'
import CardFooter from './CardFooter.vue'

export default {
  title: 'atoms/Card',
  component: Card,
}

export const Default = () => ({
  components: { Card, CardHeader, CardTitle, CardDescription, CardContent },
  template: `
    <Card style="max-width: 400px;">
      <CardHeader>
        <CardTitle>Card Title</CardTitle>
        <CardDescription>Card description goes here</CardDescription>
      </CardHeader>
      <CardContent>
        <p>This is the card content area.</p>
      </CardContent>
    </Card>
  `,
})

export const WithFooter = () => ({
  components: { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter },
  template: `
    <Card style="max-width: 400px;">
      <CardHeader>
        <CardTitle>Card with Footer</CardTitle>
        <CardDescription>This card includes a footer</CardDescription>
      </CardHeader>
      <CardContent>
        <p>Card content goes here.</p>
      </CardContent>
      <CardFooter>
        <button style="padding: 8px 16px; background: #000; color: #fff; border-radius: 6px;">Action</button>
      </CardFooter>
    </Card>
  `,
})

export const SimpleCard = () => ({
  components: { Card, CardContent },
  template: `
    <Card style="max-width: 400px;">
      <CardContent>
        <p>Simple card with just content.</p>
      </CardContent>
    </Card>
  `,
})

export const AllCompositions = () => ({
  components: { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter },
  template: `
    <div style="display: flex; flex-direction: column; gap: 24px; max-width: 600px;">
      <Card>
        <CardHeader>
          <CardTitle>Full Card</CardTitle>
          <CardDescription>With all subcomponents</CardDescription>
        </CardHeader>
        <CardContent>
          <p>This card demonstrates all available subcomponents working together.</p>
        </CardContent>
        <CardFooter>
          <button style="padding: 8px 16px; background: #000; color: #fff; border-radius: 6px;">Primary</button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Header Only</CardTitle>
          <CardDescription>Card with just a header</CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardContent>
          <p>Content only card without header or footer.</p>
        </CardContent>
      </Card>
    </div>
  `,
})
