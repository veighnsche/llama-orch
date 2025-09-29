from __future__ import annotations
from pathlib import Path

def write_text(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def make_minimal_inputs(root: Path) -> Path:
    """Create a minimal valid inputs tree under root and return its path."""
    inputs = root / "inputs"
    # simulation.yaml
    write_text(
        inputs / "simulation.yaml",
        """
run:
  output_dir: outputs
  random_runs_per_simulation: 1
stochastic:
  simulations_per_run: 1
  percentiles: [10, 50, 90]
targets:
  horizon_months: 3
  private_margin_threshold_pct: 10
  require_monotonic_growth_public_active_customers: true
  require_monotonic_growth_private_active_customers: true
  autoscaling_util_tolerance_pct: 25
  public_growth_min_mom_pct: 0
        """.strip()
    )
    # operator/public_tap.yaml
    write_text(
        inputs / "operator" / "public_tap.yaml",
        """
pricing_policy:
  public_tap:
    target_margin_pct: 50
    round_increment_eur_per_1k: 0.01
    min_floor_eur_per_1k: 0
    max_cap_eur_per_1k: 1000000
autoscaling:
  target_utilization_pct: 75
  peak_factor: 1.2
  min_instances_per_model: 0
  max_instances_per_model: 10
        """.strip()
    )
    # operator/private_tap.yaml
    write_text(
        inputs / "operator" / "private_tap.yaml",
        """
pricing_policy:
  private_tap:
    default_markup_over_provider_cost_pct: 40
    management_fee_eur_per_month: 0
        """.strip()
    )
    # operator/general.yaml (finance)
    write_text(
        inputs / "operator" / "general.yaml",
        """
finance:
  fixed_costs_monthly_eur:
    rent: 1000
    admin: 500
        """.strip()
    )
    # facts/market_env.yaml
    write_text(
        inputs / "facts" / "market_env.yaml",
        """
finance:
  eur_usd_fx_rate:
    value: 0.9
        """.strip()
    )
    # operator/curated_public_tap_models.csv
    write_text(
        inputs / "operator" / "curated_public_tap_models.csv",
        "model,typical_vram_gb\nllama-3-8b,16\n".strip(),
    )
    # operator/curated_gpu.csv
    write_text(
        inputs / "operator" / "curated_gpu.csv",
        "provider,gpu_vram_gb,price_per_gpu_hr,gpu\nprovA,24,2.0,RTX 4090\n".strip(),
    )
    # variables CSVs with exact headers
    hdr = "variable_id,scope,path,type,unit,min,max,step,default,treatment,notes\n"
    write_text(inputs / "variables" / "general.csv", hdr)
    write_text(inputs / "variables" / "public_tap.csv", hdr)
    write_text(inputs / "variables" / "private_tap.csv", hdr)
    return inputs
