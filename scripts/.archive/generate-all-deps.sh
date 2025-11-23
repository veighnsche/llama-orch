#!/bin/bash
# Generate dependency graphs in all formats
# Usage: ./scripts/generate-all-deps.sh [output-dir]

set -e

OUTPUT_DIR="${1:-.docs/architecture/dependencies}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔧 Generating dependency graphs..."
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate all formats
echo "📊 Generating statistics..."
python3 "$SCRIPT_DIR/dependency-graph.py" --format stats > "$OUTPUT_DIR/stats.md"

echo "📋 Generating JSON..."
python3 "$SCRIPT_DIR/dependency-graph.py" --format json --output "$OUTPUT_DIR/dependencies.json"

echo "🔷 Generating Mermaid diagram..."
python3 "$SCRIPT_DIR/dependency-graph.py" --format mermaid --output "$OUTPUT_DIR/dependencies.mmd"

echo "🔶 Generating GraphViz DOT (dark mode)..."
python3 "$SCRIPT_DIR/dependency-graph.py" --format dot --output "$OUTPUT_DIR/dependencies.dot"

# Try to render DOT to images if graphviz is installed
if command -v dot &> /dev/null; then
    echo "🖼️  Rendering PNG (dark mode)..."
    dot -Tpng "$OUTPUT_DIR/dependencies.dot" -o "$OUTPUT_DIR/dependencies.png"
    
    echo "🖼️  Rendering SVG (dark mode)..."
    dot -Tsvg "$OUTPUT_DIR/dependencies.dot" -o "$OUTPUT_DIR/dependencies.svg"
    
    echo ""
    echo "✅ Generated all formats:"
    echo "   - $OUTPUT_DIR/stats.md"
    echo "   - $OUTPUT_DIR/dependencies.json"
    echo "   - $OUTPUT_DIR/dependencies.mmd"
    echo "   - $OUTPUT_DIR/dependencies.dot"
    echo "   - $OUTPUT_DIR/dependencies.png"
    echo "   - $OUTPUT_DIR/dependencies.svg"
else
    echo ""
    echo "✅ Generated all formats:"
    echo "   - $OUTPUT_DIR/stats.md"
    echo "   - $OUTPUT_DIR/dependencies.json"
    echo "   - $OUTPUT_DIR/dependencies.mmd"
    echo "   - $OUTPUT_DIR/dependencies.dot"
    echo ""
    echo "ℹ️  Install graphviz to generate PNG/SVG images:"
    echo "   sudo apt install graphviz  # Ubuntu/Debian"
    echo "   brew install graphviz      # macOS"
fi

echo ""
echo "🎉 Done!"
