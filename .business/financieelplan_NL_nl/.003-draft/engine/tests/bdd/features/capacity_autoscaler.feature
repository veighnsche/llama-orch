Feature: Autoscaling planner and simulator
  Capacity planning and autoscaling events should be produced deterministically

  Scenario: Capacity planner outputs the capacity plan CSV
    Given the inputs directory ../inputs
    And a fresh outputs directory
    When I run the engine CLI with pipelines "public" and seed 424242
    Then the command should exit with code 0
    And the outputs directory should contain CSV "public_tap_capacity_plan.csv" with headers model,gpu,avg_tokens_per_hour,peak_tokens_per_hour,tps,cap_tokens_per_hour_per_instance,instances_needed,target_utilization_pct,capacity_violation

  Scenario: Autoscaler simulator emits scaling events CSV (headers)
    Given the inputs directory ../inputs
    And a fresh outputs directory
    When I run the engine CLI with pipelines "public" and seed 424242
    Then the command should exit with code 0
    And the outputs directory should contain CSV "public_tap_scaling_events.csv" with headers timestamp_s,model,gpu,demand_tokens_per_hour,effective_capacity,replicas_prev,replicas_new,reason,util_pct
    And CSV "public_tap_scaling_events.csv" should have at least 1 data rows
    And the run_summary should contain analysis KPI path "autoscaling.p95_util_pct"
