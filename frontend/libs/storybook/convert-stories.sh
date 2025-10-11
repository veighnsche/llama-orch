#!/bin/bash
# TEAM-FE-004: Script to convert all .story.ts files to .story.vue format
# This is required because Histoire only supports .story.vue format, not .story.ts

echo "🔄 Converting all .story.ts files to .story.vue format..."
echo ""

# Counter for tracking
converted=0
skipped=0
errors=0

# Find all .story.ts files
while IFS= read -r ts_file; do
    # Skip empty lines
    [ -z "$ts_file" ] && continue
    # Skip if already has a .vue version
    vue_file="${ts_file%.ts}.vue"
    if [ -f "$vue_file" ]; then
        echo "⏭️  Skipping $ts_file (already has .vue version)"
        ((skipped++))
        continue
    fi

    # Extract directory and component name
    dir=$(dirname "$ts_file")
    filename=$(basename "$ts_file" .story.ts)
    component_file="$dir/$filename.vue"
    
    # Determine story title based on directory structure
    if [[ "$dir" == *"/atoms/"* ]]; then
        story_title="atoms/$filename"
    elif [[ "$dir" == *"/molecules/"* ]]; then
        story_title="molecules/$filename"
    elif [[ "$dir" == *"/organisms/"* ]]; then
        story_title="organisms/$filename"
    else
        story_title="$filename"
    fi

    # Check if component file exists
    if [ ! -f "$component_file" ]; then
        echo "⚠️  Warning: Component file not found for $ts_file"
        echo "   Expected: $component_file"
        ((errors++))
        continue
    fi

    # Create the .story.vue file with proper variable substitution
    {
        echo "<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->"
        echo "<script setup lang=\"ts\">"
        echo "import $filename from './$filename.vue'"
        echo "</script>"
        echo ""
        echo "<template>"
        echo "  <Story title=\"$story_title\">"
        echo "    <Variant title=\"Default\">"
        echo "      <$filename />"
        echo "    </Variant>"
        echo "  </Story>"
        echo "</template>"
    } > "$vue_file"

    echo "✅ Converted: $ts_file"
    
    # Delete the old .story.ts file
    rm "$ts_file"
    
    ((converted++))
done < <(find stories -name "*.story.ts" -type f)

echo ""
echo "📊 Conversion Summary:"
echo "   ✅ Converted: $converted files"
echo "   ⏭️  Skipped: $skipped files"
echo "   ❌ Errors: $errors files"
echo ""
echo "🎉 Done! All .story.ts files have been converted to .story.vue format."
echo ""
echo "⚠️  Note: These are basic conversions with a single 'Default' variant."
echo "   You may want to add more variants to show different props/states."
