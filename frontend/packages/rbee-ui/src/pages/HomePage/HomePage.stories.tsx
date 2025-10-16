import type { Meta, StoryObj } from "@storybook/react";
import HomePage from "./HomePage";

const meta = {
  title: "Pages/HomePage",
  component: HomePage,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof HomePage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
