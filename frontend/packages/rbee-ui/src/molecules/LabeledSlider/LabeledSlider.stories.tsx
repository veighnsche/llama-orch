import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { LabeledSlider } from "./LabeledSlider";

const meta = {
  title: "Molecules/LabeledSlider",
  component: LabeledSlider,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof LabeledSlider>;

export default meta;
type Story = StoryObj<typeof meta>;

export const HoursPerDay: Story = {
  args: {} as any,
  render: () => {
    const [value, setValue] = useState([20]);
    return (
      <div className="max-w-md">
        <LabeledSlider
          label="Hours Available Per Day"
          value={value}
          onValueChange={setValue}
          min={1}
          max={24}
          step={1}
          ariaLabel="Hours available per day"
          formatValue={(v) => `${v}h`}
          minLabel="1h"
          maxLabel="24h"
          helperText={(v) => `≈ ${v * 30}h / mo`}
        />
      </div>
    );
  },
};

export const Utilization: Story = {
  args: {} as any,
  render: () => {
    const [value, setValue] = useState([80]);
    return (
      <div className="max-w-md">
        <LabeledSlider
          label="Expected Utilization"
          value={value}
          onValueChange={setValue}
          min={10}
          max={100}
          step={5}
          ariaLabel="Expected utilization"
          formatValue={(v) => `${v}%`}
          minLabel="10%"
          maxLabel="100%"
        />
      </div>
    );
  },
};

export const Volume: Story = {
  args: {} as any,
  render: () => {
    const [value, setValue] = useState([50]);
    return (
      <div className="max-w-md">
        <LabeledSlider
          label="Volume"
          value={value}
          onValueChange={setValue}
          min={0}
          max={100}
          step={1}
          ariaLabel="Volume level"
          formatValue={(v) => `${v}%`}
          minLabel="Mute"
          maxLabel="Max"
        />
      </div>
    );
  },
};

export const Price: Story = {
  args: {} as any,
  render: () => {
    const [value, setValue] = useState([500]);
    return (
      <div className="max-w-md">
        <LabeledSlider
          label="Budget"
          value={value}
          onValueChange={setValue}
          min={0}
          max={1000}
          step={50}
          ariaLabel="Budget amount"
          formatValue={(v) => `€${v}`}
          minLabel="€0"
          maxLabel="€1,000"
          helperText="Monthly budget limit"
        />
      </div>
    );
  },
};

export const WithoutLabels: Story = {
  args: {} as any,
  render: () => {
    const [value, setValue] = useState([5]);
    return (
      <div className="max-w-md">
        <LabeledSlider
          label="Rating"
          value={value}
          onValueChange={setValue}
          min={1}
          max={10}
          step={1}
          ariaLabel="Rating value"
          formatValue={(v) => `${v}/10`}
        />
      </div>
    );
  },
};

export const MultipleSliders: Story = {
  args: {} as any,
  render: () => {
    const [hours, setHours] = useState([20]);
    const [util, setUtil] = useState([80]);
    return (
      <div className="max-w-md space-y-6">
        <LabeledSlider
          label="Hours Per Day"
          value={hours}
          onValueChange={setHours}
          min={1}
          max={24}
          step={1}
          ariaLabel="Hours available per day"
          formatValue={(v) => `${v}h`}
          minLabel="1h"
          maxLabel="24h"
          helperText={(v) => `≈ ${v * 30}h / mo`}
        />
        <LabeledSlider
          label="Utilization"
          value={util}
          onValueChange={setUtil}
          min={10}
          max={100}
          step={5}
          ariaLabel="Expected utilization"
          formatValue={(v) => `${v}%`}
          minLabel="10%"
          maxLabel="100%"
        />
      </div>
    );
  },
};
