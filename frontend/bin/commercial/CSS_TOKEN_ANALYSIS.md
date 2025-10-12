# CSS Token Standardization Analysis

**Files Analyzed:** 148
**Total Lines:** 14,488

---

## Colors

**Unique tokens:** 76
**Total usage:** 3373

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `text-muted-foreground` | 571 | 103 |
| `text-sm` | 376 | 83 |
| `text-foreground` | 373 | 73 |
| `border-border` | 265 | 68 |
| `bg-primary` | 165 | 47 |
| `text-primary` | 165 | 53 |
| `text-center` | 164 | 52 |
| `text-chart-3` | 150 | 28 |
| `bg-card` | 134 | 48 |
| `text-xl` | 103 | 42 |
| `border-b` | 70 | 43 |
| `text-xs` | 64 | 21 |
| `bg-background` | 61 | 39 |
| `text-balance` | 54 | 38 |
| `text-lg` | 50 | 22 |
| `text-card-foreground` | 45 | 16 |
| `text-pretty` | 42 | 13 |
| `bg-chart-3` | 38 | 10 |
| `border-primary` | 37 | 27 |
| `bg-gradient-to-b` | 37 | 18 |

---

## Font Weight

**Unique tokens:** 4
**Total usage:** 488

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `font-bold` | 240 | 63 |
| `font-medium` | 142 | 45 |
| `font-semibold` | 102 | 35 |
| `font-normal` | 4 | 3 |

---

## Hover

**Unique tokens:** 22
**Total usage:** 93

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `bg-primary` | 14 | 13 |
| `bg-secondary` | 12 | 10 |
| `text-foreground` | 10 | 6 |
| `border-destructive` | 10 | 4 |
| `no-underline` | 8 | 1 |
| `border-primary` | 7 | 4 |
| `bg-card` | 6 | 3 |
| `text-primary` | 5 | 3 |
| `opacity-100` | 3 | 3 |
| `bg-sidebar-accent` | 3 | 1 |
| `text-sidebar-accent-foreground` | 3 | 1 |
| `translate-x-1` | 2 | 2 |
| `scale` | 1 | 1 |
| `border-border` | 1 | 1 |
| `bg-destructive` | 1 | 1 |
| `text-destructive-foreground` | 1 | 1 |
| `text-red-50` | 1 | 1 |
| `underline` | 1 | 1 |
| `ring-4` | 1 | 1 |
| `after` | 1 | 1 |

---

## Opacity

**Unique tokens:** 6
**Total usage:** 34

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `opacity-50` | 18 | 13 |
| `opacity-100` | 7 | 4 |
| `opacity-0` | 4 | 3 |
| `opacity-75` | 2 | 2 |
| `opacity-70` | 2 | 2 |
| `opacity-90` | 1 | 1 |

---

## Rounded

**Unique tokens:** 13
**Total usage:** 447

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `rounded-lg` | 260 | 59 |
| `rounded-full` | 89 | 32 |
| `rounded` | 49 | 19 |
| `rounded-md` | 23 | 14 |
| `rounded-xl` | 14 | 8 |
| `rounded-xs` | 3 | 3 |
| `rounded-none` | 2 | 1 |
| `rounded-sm` | 2 | 1 |
| `rounded-tl` | 1 | 1 |
| `rounded-b` | 1 | 1 |
| `rounded-t` | 1 | 1 |
| `rounded-l` | 1 | 1 |
| `rounded-r` | 1 | 1 |

---

## Shadow

**Unique tokens:** 8
**Total usage:** 35

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `shadow-xs` | 9 | 9 |
| `shadow` | 8 | 8 |
| `shadow-lg` | 5 | 4 |
| `shadow-sm` | 4 | 3 |
| `shadow-2xl` | 3 | 3 |
| `shadow-none` | 3 | 2 |
| `shadow-md` | 2 | 2 |
| `shadow-xl` | 1 | 1 |

---

## Sizing

**Unique tokens:** 35
**Total usage:** 781

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `h-5` | 113 | 25 |
| `w-5` | 113 | 24 |
| `h-12` | 55 | 19 |
| `w-12` | 51 | 17 |
| `h-6` | 51 | 19 |
| `w-6` | 51 | 19 |
| `h-8` | 39 | 10 |
| `h-4` | 38 | 18 |
| `w-4` | 36 | 16 |
| `w-8` | 22 | 5 |
| `h-1` | 22 | 5 |
| `w-1` | 21 | 4 |
| `h-2` | 16 | 10 |
| `h-14` | 14 | 5 |
| `w-2` | 14 | 8 |
| `h-16` | 14 | 5 |
| `w-16` | 13 | 4 |
| `w-3` | 12 | 6 |
| `h-7` | 11 | 5 |
| `w-32` | 10 | 3 |

### Standardization Opportunities

- **h-X**: 13 variants, 393 uses
  - `h-1`: 22 uses
  - `h-10`: 7 uses
  - `h-12`: 55 uses
  - `h-14`: 14 uses
  - `h-16`: 14 uses
  - `h-2`: 16 uses
  - `h-3`: 7 uses
  - `h-4`: 38 uses
  - `h-5`: 113 uses
  - `h-6`: 51 uses
  - `h-7`: 11 uses
  - `h-8`: 39 uses
  - `h-9`: 6 uses
- **min-h-X**: 4 variants, 4 uses
  - `min-h-0`: 1 uses
  - `min-h-16`: 1 uses
  - `min-h-4`: 1 uses
  - `min-h-44`: 1 uses
- **min-w-X**: 2 variants, 9 uses
  - `min-w-0`: 8 uses
  - `min-w-5`: 1 uses
- **w-X**: 16 variants, 375 uses
  - `w-0`: 4 uses
  - `w-1`: 21 uses
  - `w-10`: 6 uses
  - `w-12`: 51 uses
  - `w-14`: 9 uses
  - `w-16`: 13 uses
  - `w-2`: 14 uses
  - `w-3`: 12 uses
  - `w-32`: 10 uses
  - `w-4`: 36 uses
  - `w-40`: 3 uses
  - `w-5`: 113 uses
  - `w-6`: 51 uses
  - `w-7`: 9 uses
  - `w-8`: 22 uses
  - `w-9`: 1 uses

---

## Spacing

**Unique tokens:** 89
**Total usage:** 1774

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `p-4` | 161 | 29 |
| `gap-2` | 157 | 46 |
| `mb-4` | 137 | 36 |
| `mb-2` | 89 | 28 |
| `p-8` | 83 | 32 |
| `gap-3` | 75 | 26 |
| `mb-6` | 60 | 31 |
| `gap-4` | 60 | 32 |
| `p-6` | 57 | 25 |
| `px-6` | 51 | 38 |
| `mt-0` | 41 | 10 |
| `gap-8` | 41 | 34 |
| `py-24` | 40 | 40 |
| `mb-3` | 37 | 13 |
| `space-y-2` | 36 | 16 |
| `px-4` | 36 | 22 |
| `mb-1` | 36 | 14 |
| `space-y-4` | 30 | 14 |
| `mt-1` | 29 | 7 |
| `space-y-3` | 28 | 16 |

### Standardization Opportunities

- **gap-X**: 10 variants, 389 uses
  - `gap-0`: 1 uses
  - `gap-1`: 25 uses
  - `gap-12`: 6 uses
  - `gap-16`: 2 uses
  - `gap-2`: 157 uses
  - `gap-3`: 75 uses
  - `gap-4`: 60 uses
  - `gap-6`: 21 uses
  - `gap-7`: 1 uses
  - `gap-8`: 41 uses
- **m-X**: 2 variants, 2 uses
  - `m-0`: 1 uses
  - `m-2`: 1 uses
- **mb-X**: 9 variants, 406 uses
  - `mb-1`: 36 uses
  - `mb-12`: 5 uses
  - `mb-16`: 26 uses
  - `mb-2`: 89 uses
  - `mb-24`: 1 uses
  - `mb-3`: 37 uses
  - `mb-4`: 137 uses
  - `mb-6`: 60 uses
  - `mb-8`: 15 uses
- **ml-X**: 4 variants, 17 uses
  - `ml-0`: 1 uses
  - `ml-1`: 1 uses
  - `ml-2`: 13 uses
  - `ml-4`: 2 uses
- **mt-X**: 9 variants, 157 uses
  - `mt-0`: 41 uses
  - `mt-1`: 29 uses
  - `mt-12`: 14 uses
  - `mt-16`: 10 uses
  - `mt-2`: 23 uses
  - `mt-24`: 1 uses
  - `mt-4`: 14 uses
  - `mt-6`: 11 uses
  - `mt-8`: 14 uses
- **mx-X**: 3 variants, 7 uses
  - `mx-1`: 5 uses
  - `mx-2`: 1 uses
  - `mx-3`: 1 uses
- **my-X**: 3 variants, 7 uses
  - `my-0`: 1 uses
  - `my-1`: 4 uses
  - `my-2`: 2 uses
- **p-X**: 8 variants, 322 uses
  - `p-0`: 4 uses
  - `p-1`: 3 uses
  - `p-12`: 3 uses
  - `p-2`: 4 uses
  - `p-3`: 7 uses
  - `p-4`: 161 uses
  - `p-6`: 57 uses
  - `p-8`: 83 uses
- **pb-X**: 3 variants, 4 uses
  - `pb-3`: 2 uses
  - `pb-4`: 1 uses
  - `pb-6`: 1 uses
- **pl-X**: 5 variants, 32 uses
  - `pl-10`: 1 uses
  - `pl-16`: 2 uses
  - `pl-2`: 2 uses
  - `pl-4`: 21 uses
  - `pl-8`: 6 uses
- **pt-X**: 7 variants, 19 uses
  - `pt-0`: 2 uses
  - `pt-12`: 1 uses
  - `pt-2`: 2 uses
  - `pt-3`: 3 uses
  - `pt-4`: 8 uses
  - `pt-6`: 1 uses
  - `pt-8`: 2 uses
- **px-X**: 6 variants, 138 uses
  - `px-1`: 1 uses
  - `px-2`: 20 uses
  - `px-3`: 11 uses
  - `px-4`: 36 uses
  - `px-6`: 51 uses
  - `px-8`: 19 uses
- **py-X**: 10 variants, 115 uses
  - `py-0`: 2 uses
  - `py-1`: 18 uses
  - `py-16`: 1 uses
  - `py-2`: 27 uses
  - `py-20`: 1 uses
  - `py-24`: 40 uses
  - `py-3`: 18 uses
  - `py-32`: 4 uses
  - `py-4`: 2 uses
  - `py-6`: 2 uses
- **space-y-X**: 8 variants, 143 uses
  - `space-y-1`: 20 uses
  - `space-y-12`: 1 uses
  - `space-y-16`: 1 uses
  - `space-y-2`: 36 uses
  - `space-y-3`: 28 uses
  - `space-y-4`: 30 uses
  - `space-y-6`: 18 uses
  - `space-y-8`: 9 uses

---

## Text

**Unique tokens:** 11
**Total usage:** 760

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `text-sm` | 376 | 83 |
| `text-xl` | 103 | 42 |
| `text-xs` | 64 | 21 |
| `text-2xl` | 61 | 26 |
| `text-lg` | 50 | 22 |
| `text-4xl` | 47 | 35 |
| `text-3xl` | 25 | 13 |
| `text-5xl` | 23 | 22 |
| `text-6xl` | 6 | 6 |
| `text-base` | 3 | 3 |
| `text-7xl` | 2 | 2 |

### Standardization Opportunities


---

## Transitions

**Unique tokens:** 13
**Total usage:** 73

### Most Used Tokens

| Token | Count | Files |
|-------|-------|-------|
| `transition-all` | 21 | 12 |
| `transition-colors` | 17 | 10 |
| `transition` | 9 | 9 |
| `transition-transform` | 7 | 5 |
| `duration-300` | 5 | 4 |
| `transition-opacity` | 3 | 3 |
| `duration-200` | 3 | 3 |
| `duration-500` | 2 | 2 |
| `ease-linear` | 2 | 1 |
| `ease-in` | 1 | 1 |
| `transition-shadow` | 1 | 1 |
| `transition-none` | 1 | 1 |
| `duration-1000` | 1 | 1 |

---

## Color Token Analysis

### Background Colors

| Token | Count |
|-------|-------|
| `bg-primary` | 165 |
| `bg-card` | 134 |
| `bg-background` | 61 |
| `bg-chart-3` | 38 |
| `bg-gradient-to-b` | 37 |
| `bg-muted` | 35 |
| `bg-secondary` | 34 |
| `bg-transparent` | 22 |
| `bg-border` | 16 |
| `bg-destructive` | 13 |
| `bg-gradient-to-br` | 11 |
| `bg-input` | 9 |
| `bg-accent` | 6 |
| `bg-sidebar-accent` | 5 |
| `bg-black` | 4 |
| `bg-sidebar` | 4 |
| `bg-gradient-to-r` | 3 |
| `bg-chart-2` | 3 |
| `bg-popover` | 2 |
| `bg-foreground` | 2 |
| `bg-sidebar-border` | 2 |
| `bg-red-500` | 1 |
| `bg-amber-500` | 1 |
| `bg-green-500` | 1 |
| `bg-clip-text` | 1 |
| `bg-white` | 1 |

### Border Colors

| Token | Count |
|-------|-------|
| `border-border` | 265 |
| `border-b` | 70 |
| `border-primary` | 37 |
| `border-destructive` | 31 |
| `border-chart-3` | 12 |
| `border-t` | 12 |
| `border-ring` | 8 |
| `border-input` | 6 |
| `border-l` | 5 |
| `border-r` | 3 |
| `border-b-0` | 2 |
| `border-sidebar-border` | 2 |
| `border-collapse` | 1 |
| `border-chart-2` | 1 |
| `border-muted` | 1 |
| `border-dashed` | 1 |
| `border-l-transparent` | 1 |
| `border-t-transparent` | 1 |
| `border-y` | 1 |
| `border-transparent` | 1 |

### Text Colors

| Token | Count |
|-------|-------|
| `text-muted-foreground` | 571 |
| `text-sm` | 376 |
| `text-foreground` | 373 |
| `text-primary` | 165 |
| `text-center` | 164 |
| `text-chart-3` | 150 |
| `text-xl` | 103 |
| `text-xs` | 64 |
| `text-balance` | 54 |
| `text-lg` | 50 |
| `text-card-foreground` | 45 |
| `text-pretty` | 42 |
| `text-destructive` | 33 |
| `text-primary-foreground` | 27 |
| `text-left` | 15 |
| `text-chart-2` | 13 |
| `text-sidebar-accent-foreground` | 10 |
| `text-white` | 8 |
| `text-sidebar-foreground` | 7 |
| `text-right` | 6 |
| `text-chart-4` | 6 |
| `text-slate-950` | 5 |
| `text-accent-foreground` | 4 |
| `text-base` | 3 |
| `text-popover-foreground` | 2 |
| `text-transparent` | 1 |
| `text-destructive-foreground` | 1 |
| `text-red-300` | 1 |
| `text-red-50` | 1 |
| `text-current` | 1 |

