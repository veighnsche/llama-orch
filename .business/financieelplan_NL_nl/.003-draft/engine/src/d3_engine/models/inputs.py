"""Input models (scaffold)."""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


# ---------------------------
# Enums & simple aliases
# ---------------------------
PipelineName = Literal["public", "private"]
LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
OverheadDriver = Literal["revenue", "gpu_hours", "tokens"]

ScopeName = Literal["general", "public_tap", "private_tap"]
VarType = Literal["numeric", "discrete"]
Treatment = Literal["fixed", "low_to_high", "random"]
Unit = Literal[
    "percent",
    "fraction",
    "EUR",
    "EUR_per_month",
    "months",
    "tokens",
    "count",
]


# ---------------------------
# simulation.yaml models
# ---------------------------
class RunConfig(BaseModel):
    pipelines: List[PipelineName] = Field(default_factory=lambda: ["public", "private"])
    random_seed: Optional[int] = None
    output_dir: str = ".003-draft/outputs"
    fail_on_warning: bool = False
    max_concurrency: Optional[int] = None
    random_runs_per_simulation: int = 1

    @field_validator("random_runs_per_simulation")
    @classmethod
    def _validate_random_runs(cls, v: int) -> int:
        if v < 1:
            raise ValueError("random_runs_per_simulation MUST be >= 1")
        return v


class StochasticConfig(BaseModel):
    simulations_per_run: int = 1000
    percentiles: List[int] = Field(default_factory=lambda: [10, 50, 90])
    random_seed: Optional[int] = None

    @field_validator("simulations_per_run")
    @classmethod
    def _validate_sims(cls, v: int) -> int:
        if v < 1:
            raise ValueError("simulations_per_run MUST be >= 1")
        return v


class StressConfig(BaseModel):
    provider_price_drift_pct: float = 0.0
    tps_downshift_pct: float = 0.0
    fx_widen_buffer_pct: float = 0.0


class ConsolidationConfig(BaseModel):
    overhead_allocation_driver: OverheadDriver = "revenue"
    include_loan_in_cashflow: bool = True
    report_percentiles: List[int] = Field(default_factory=lambda: [10, 50, 90])


class UIConfig(BaseModel):
    show_credit_packs: List[int] = Field(default_factory=lambda: [5, 10, 20, 50, 100, 200, 500])
    halt_at_zero_simulation: bool = True


class LoggingConfig(BaseModel):
    level: LogLevel = "INFO"
    write_run_summary: bool = True


class TargetsConfig(BaseModel):
    horizon_months: int = 18
    private_margin_threshold_pct: float = 20.0
    require_monotonic_growth_public_active_customers: bool = True
    require_monotonic_growth_private_active_customers: bool = True
    public_growth_min_mom_pct: Optional[float] = None

    @field_validator("horizon_months")
    @classmethod
    def _horizon_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("horizon_months MUST be >= 1")
        return v


class SimulationPlan(BaseModel):
    run: RunConfig
    stochastic: StochasticConfig
    stress: StressConfig = StressConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()
    ui: UIConfig = UIConfig()
    logging: LoggingConfig = LoggingConfig()
    targets: TargetsConfig = TargetsConfig()


# ---------------------------
# operator/*.yaml models
# Keep flexible with extra keys allowed, but enforce known blocks when present
# ---------------------------
class PublicAutoscaling(BaseModel):
    target_utilization_pct: float = 75.0
    peak_factor: float = 1.2
    min_instances_per_model: int = 0
    max_instances_per_model: int = 100
    # Simulator policy (optional in v0.1 but validated if provided)
    evaluation_interval_s: int = 60
    scale_up_threshold_pct: float = 70.0
    scale_down_threshold_pct: float = 50.0
    scale_up_step_replicas: int = 1
    scale_down_step_replicas: int = 1
    stabilization_window_s: int = 300
    warmup_s: int = 120
    cooldown_s: int = 120
    capacity_peak_percentile: Optional[int] = None

    @model_validator(mode="after")
    def _bounds(self) -> "PublicAutoscaling":
        if not (1 <= self.target_utilization_pct <= 100):
            raise ValueError("target_utilization_pct MUST be 1..100")
        if self.peak_factor < 1.0:
            raise ValueError("peak_factor MUST be >= 1.0")
        if self.min_instances_per_model < 0:
            raise ValueError("min_instances_per_model MUST be >= 0")
        if self.max_instances_per_model < 1:
            raise ValueError("max_instances_per_model MUST be >= 1")
        if self.min_instances_per_model > self.max_instances_per_model:
            raise ValueError("min_instances_per_model MUST be <= max_instances_per_model")
        if self.evaluation_interval_s <= 0:
            raise ValueError("evaluation_interval_s MUST be > 0")
        if not (0 < self.scale_down_threshold_pct < self.scale_up_threshold_pct <= 100):
            raise ValueError("thresholds MUST satisfy 0 < scale_down < scale_up ≤ 100")
        if self.scale_up_step_replicas < 1 or self.scale_down_step_replicas < 1:
            raise ValueError("scale step replicas MUST be ≥ 1")
        if self.stabilization_window_s < 0 or self.warmup_s < 0 or self.cooldown_s < 0:
            raise ValueError("timings MUST be ≥ 0")
        if self.capacity_peak_percentile is not None and not (1 <= self.capacity_peak_percentile <= 100):
            raise ValueError("capacity_peak_percentile MUST be 1..100 when set")
        return self


class OperatorGeneral(BaseModel):
    model_config = {"extra": "allow"}
    finance: Dict[str, Any] = Field(default_factory=dict)
    tax: Dict[str, Any] = Field(default_factory=dict)
    reserves: Dict[str, Any] = Field(default_factory=dict)
    loan: Dict[str, Any] = Field(default_factory=dict)


class OperatorPublicTap(BaseModel):
    model_config = {"extra": "allow"}
    pricing_policy: Dict[str, Any] = Field(default_factory=dict)
    acquisition: Dict[str, Any] = Field(default_factory=dict)
    autoscaling: PublicAutoscaling = Field(default_factory=PublicAutoscaling)


class OperatorPrivateTap(BaseModel):
    model_config = {"extra": "allow"}
    pricing_policy: Dict[str, Any] = Field(default_factory=dict)
    acquisition: Dict[str, Any] = Field(default_factory=dict)
    vendor_weights: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------
# variables/*.csv rows
# ---------------------------
class VariableRow(BaseModel):
    variable_id: str
    scope: ScopeName
    path: str
    type: VarType
    unit: Unit
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    default: Optional[float] = None
    treatment: Treatment
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _validate_numeric(self) -> "VariableRow":
        if self.type == "numeric":
            if self.min is None or self.max is None or self.step is None or self.default is None:
                raise ValueError("numeric row MUST have min,max,step,default")
            if self.min > self.max:
                raise ValueError("min MUST be <= max")
            if self.step <= 0:
                raise ValueError("step MUST be > 0")
            if not (self.min <= self.default <= self.max):
                raise ValueError("default MUST be within [min,max]")
        else:
            # discrete: expect values hint in notes (validated elsewhere)
            pass
        return self


# ---------------------------
# facts/* (minimal placeholders, extra allowed)
# ---------------------------
class MarketEnv(BaseModel):
    model_config = {"extra": "allow"}
    finance: Dict[str, Any] = Field(default_factory=dict)


# Convenient container for operator bundle
class OperatorBundle(BaseModel):
    general: OperatorGeneral
    public_tap: OperatorPublicTap
    private_tap: OperatorPrivateTap
