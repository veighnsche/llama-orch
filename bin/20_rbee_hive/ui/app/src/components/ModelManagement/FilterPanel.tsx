// TEAM-381: Filter Panel - Controls for filtering HuggingFace search results

import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Checkbox,
  RadioGroup,
  RadioGroupItem,
  Label,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@rbee/ui/atoms'
import type { FilterState } from './types'

interface FilterPanelProps {
  filters: FilterState
  onFiltersChange: (filters: FilterState) => void
}

export function FilterPanel({ filters, onFiltersChange }: FilterPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Filters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Format Filter */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Format</Label>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="format-gguf"
                checked={filters.formats.includes('gguf')}
                onCheckedChange={(checked) => {
                  const newFormats = checked
                    ? [...filters.formats, 'gguf']
                    : filters.formats.filter((f) => f !== 'gguf')
                  onFiltersChange({ ...filters, formats: newFormats })
                }}
              />
              <Label htmlFor="format-gguf" className="text-sm font-normal">
                GGUF (Quantized)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="format-safetensors"
                checked={filters.formats.includes('safetensors')}
                onCheckedChange={(checked) => {
                  const newFormats = checked
                    ? [...filters.formats, 'safetensors']
                    : filters.formats.filter((f) => f !== 'safetensors')
                  onFiltersChange({ ...filters, formats: newFormats })
                }}
              />
              <Label htmlFor="format-safetensors" className="text-sm font-normal">
                SafeTensors (Full)
              </Label>
            </div>
          </div>
        </div>

        {/* Architecture Filter */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Architecture</Label>
          <div className="space-y-2">
            {['llama', 'mistral', 'phi', 'gemma', 'qwen'].map((arch) => (
              <div key={arch} className="flex items-center space-x-2">
                <Checkbox
                  id={`arch-${arch}`}
                  checked={filters.architectures.includes(arch)}
                  onCheckedChange={(checked) => {
                    const newArchs = checked
                      ? [...filters.architectures, arch]
                      : filters.architectures.filter((a) => a !== arch)
                    onFiltersChange({ ...filters, architectures: newArchs })
                  }}
                />
                <Label htmlFor={`arch-${arch}`} className="text-sm font-normal capitalize">
                  {arch}
                </Label>
              </div>
            ))}
          </div>
        </div>

        {/* Size Filter */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Max Size</Label>
          <RadioGroup
            value={filters.maxSize}
            onValueChange={(value) => onFiltersChange({ ...filters, maxSize: value })}
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="5gb" id="size-5gb" />
              <Label htmlFor="size-5gb" className="text-sm font-normal">
                &lt; 5GB (Small)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="15gb" id="size-15gb" />
              <Label htmlFor="size-15gb" className="text-sm font-normal">
                &lt; 15GB (Medium)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="30gb" id="size-30gb" />
              <Label htmlFor="size-30gb" className="text-sm font-normal">
                &lt; 30GB (Large)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="all" id="size-all" />
              <Label htmlFor="size-all" className="text-sm font-normal">
                All
              </Label>
            </div>
          </RadioGroup>
        </div>

        {/* License Filter */}
        <div className="flex items-center space-x-2">
          <Checkbox
            id="open-source"
            checked={filters.openSourceOnly}
            onCheckedChange={(checked) =>
              onFiltersChange({ ...filters, openSourceOnly: !!checked })
            }
          />
          <Label htmlFor="open-source" className="text-sm font-normal">
            Open Source Only
          </Label>
        </div>

        {/* Sort By */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Sort By</Label>
          <Select
            value={filters.sortBy}
            onValueChange={(value) =>
              onFiltersChange({ ...filters, sortBy: value as FilterState['sortBy'] })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="downloads">Downloads</SelectItem>
              <SelectItem value="likes">Likes</SelectItem>
              <SelectItem value="recent">Recent</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  )
}
