#!/bin/bash

# Update Storybook titles for reorganized components

# Home organisms
find src/organisms/Home -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Organisms/\([^']*\)'|title: 'Organisms/Home/\1'|g" {} \;

# Shared organisms  
find src/organisms/Shared -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Organisms/\([^']*\)'|title: 'Organisms/Shared/\1'|g" {} \;

# Tables molecules
find src/molecules/Tables -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Tables/\1'|g" {} \;

# Enterprise molecules
find src/molecules/Enterprise -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Enterprise/\1'|g" {} \;

# Pricing molecules
find src/molecules/Pricing -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Pricing/\1'|g" {} \;

# UseCases molecules
find src/molecules/UseCases -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/UseCases/\1'|g" {} \;

# Providers molecules
find src/molecules/Providers -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Providers/\1'|g" {} \;

# Developers molecules
find src/molecules/Developers -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Developers/\1'|g" {} \;

# ErrorHandling molecules
find src/molecules/ErrorHandling -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/ErrorHandling/\1'|g" {} \;

# Navigation molecules
find src/molecules/Navigation -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Navigation/\1'|g" {} \;

# Content molecules
find src/molecules/Content -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Content/\1'|g" {} \;

# Branding molecules
find src/molecules/Branding -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Branding/\1'|g" {} \;

# Layout molecules
find src/molecules/Layout -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Layout/\1'|g" {} \;

# Stats molecules
find src/molecules/Stats -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/Stats/\1'|g" {} \;

# UI molecules
find src/molecules/UI -name "*.stories.tsx" -type f -exec sed -i "s|title: 'Molecules/\([^']*\)'|title: 'Molecules/UI/\1'|g" {} \;

echo "Storybook titles updated!"
