from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS = PROJECT_ROOT / "inputs"
OUTPUTS = PROJECT_ROOT / "outputs"
TEMPLATE_FILE = PROJECT_ROOT / "template.md"
CALC_HINTS_FILE = PROJECT_ROOT / "calculation_hints.md"
PROMPT_FILE = PROJECT_ROOT / "prompt.md"

ENGINE_VERSION = "v1.0.0"
