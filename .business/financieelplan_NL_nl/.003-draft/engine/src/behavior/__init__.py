"""Customer behavior layer (scaffold).

Provides monthly series for Public and Private simulations:
- budgets → conversions → tokens (Public)
- budgets → conversions → hours (Private)
- retention/churn → active clients
- RNG sampling helpers

Entry points (spec 24_customer_behavior.md):
- public.simulate_months(plan, variables, rng)
- private.simulate_months(plan, variables, rng)
"""
