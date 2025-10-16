import { cn } from "@rbee/ui/utils";
import * as React from "react";

export interface GPUSelectorModel {
  name: string;
  baseRate: number;
  vram: number;
}

export interface GPUSelectorProps {
  /** List of GPU models to display */
  models: GPUSelectorModel[];
  /** Currently selected GPU */
  selectedModel: GPUSelectorModel;
  /** Callback when a GPU is selected */
  onSelect: (model: GPUSelectorModel) => void;
  /** Label for the selector */
  label: string;
  /** Format function for hourly rate display */
  formatHourly: (rate: number) => string;
  /** Optional ref to the container */
  containerRef?: React.RefObject<HTMLDivElement | null>;
  /** Additional CSS classes */
  className?: string;
}

/**
 * GPUSelector molecule - displays a list of GPU models as selectable radio buttons
 *
 * @example
 * ```tsx
 * <GPUSelector
 *   models={gpuModels}
 *   selectedModel={selectedGPU}
 *   onSelect={setSelectedGPU}
 *   label="Select GPU Model"
 *   formatHourly={(rate) => `€${rate}/hr`}
 * />
 * ```
 */
export function GPUSelector({
  models,
  selectedModel,
  onSelect,
  label,
  formatHourly,
  containerRef,
  className,
}: GPUSelectorProps) {
  return (
    <div ref={containerRef} className={className}>
      <label className="mb-3 block text-sm font-medium text-muted-foreground">
        {label}
      </label>
      <div
        role="radiogroup"
        aria-label="GPU model selection"
        className="grid max-h-[320px] gap-2 overflow-y-auto pr-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border hover:scrollbar-thumb-muted-foreground"
      >
        {models.map((gpu) => {
          const isSelected = selectedModel.name === gpu.name;
          return (
            <button
              key={gpu.name}
              role="radio"
              aria-checked={isSelected}
              onClick={() => onSelect(gpu)}
              className={cn(
                "relative min-w-0 rounded-lg border p-3 text-left transition-transform hover:translate-y-0.5",
                isSelected
                  ? "animate-in zoom-in-95 border-primary bg-primary/10 motion-reduce:animate-none"
                  : "border-border bg-background/50 hover:border-border/70"
              )}
            >
              <div className="flex items-center justify-between">
                <div className="min-w-0 font-sans">
                  <div className="font-medium text-foreground">{gpu.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {gpu.vram}GB VRAM
                  </div>
                  <div className="text-[11px] text-muted-foreground">
                    Base rate
                  </div>
                </div>
                <div className="tabular-nums text-sm text-primary">
                  {formatHourly(gpu.baseRate)}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export { GPUSelector as default };
