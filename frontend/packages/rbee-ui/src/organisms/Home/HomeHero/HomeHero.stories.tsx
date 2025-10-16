import type { Meta, StoryObj } from "@storybook/react";
// import { homeHeroProps } from '../../../
import { HomeHero } from "./HomeHero";
import { homeHeroProps } from "@rbee/ui/pages";

const meta = {
  title: "Templates/HomeHero",
  component: HomeHero,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof HomeHero>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: homeHeroProps,
};
