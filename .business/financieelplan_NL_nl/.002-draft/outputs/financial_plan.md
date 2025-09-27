# Financial Plan — Template

## 0) Executive Summary

**Business model (one line):** Prepaid-only AI hosting — no debtors, no refunds, and no idle GPU costs.

- **Loan request:** €None for None months @ None% (flat)  
- **Monthly repayment:** €748.75 (total repay €44925.0)  
- **Fixed baseline (per month):** personal €3000.0 + business €0.0 + loan €748.75 = **€3748.75**  

**Revenue model:**

- **Public Tap:** flat €None per 1k tokens (no discounts).  
- **Private Tap:** prepaid GPU-hours with markup (provider cost + 50.0%) + management fee.  

**Safeguards:**

- All inflows are prepaid and **non-refundable**.  
- No service without prepaid balance.  
- GPUs are only rented when prepaid demand exists → no over-exposure.  

**Targets:**

- **Required monthly prepaid inflow (baseline):** **€None**  
- **Runway target:** 12 months

### 0.1 Diagram — Prepaid Model Flow (Mermaid)

```mermaid
flowchart TD
  A[Customer Prepay] --> B[Credits Wallet]
  B --> C[Service Usage]
  C --> D[GPU Rentals]
  B --> E[Loan Repayment]
  B --> F[Fixed Costs]
  B --> G[Marketing Reserve]
```

## 1) Inputs (Ground Truth)

### 1.1 Prepaid Policy

- **Top-up:** min €25, max €10000, expiry 12 months  
- **Refunds:** True (credits are non-refundable, except where legally required)  
- **Auto-refill:** default False, cap €200  
- **Private Tap:** prepaid only; billed in 60 min blocks  
  - Mgmt fee: €None/month  
  - GPU-hour markup: % above provider cost  
  - FX buffer: 5.0%

---

### 1.2 Catalog (Products Offered)

- **Models (allow-list):** Llama-3.1-8B, Llama-3.1-70B, Mixtral-8x7B, Mixtral-8x22B, Qwen2.5-7B, Qwen2.5-32B, Qwen2.5-72B, Yi-1.5-6B, Yi-1.5-9B, Yi-1.5-34B, DeepSeek-Coder-6.7B, DeepSeek-Coder-33B, DeepSeek-Coder-V2-16B, DeepSeek-Coder-V2-236B  
- **GPUs considered:** A10, A100 40GB (PCIe), A100 80GB (SXM/PCIe), H100 80GB (PCIe/SXM), H200 141GB, L4, L40S, RTX 3090, RTX 4090

---

### 1.3 Price Inputs

- Public Tap prices: defined **per model** (see `price_sheet.csv`)  
  - Example row: Model = Llama 3.1 8B → Sell price = €N/A / 1k tokens  
- Private Tap markup target: **50.0%** over provider GPU cost  
- Management fee: **€None / month**  

---

### 1.4 Fixed Costs (Monthly)

- Personal baseline: **€3000.0**  
- Business overhead: **€0.0**  
- Loan repayment: **€748.75**  
- **Total fixed costs (with loan): €3748.75**

---

### 1.5 Tax & Billing

- VAT: 21%  
- EU B2B reverse-charge: True  
- Stripe Tax enabled: True  
- Revenue recognition: **prepaid liability until consumed** (prepaid liability until consumed)

---

## 2) Public Tap — Cost & Price per Model

For each model offered on the Public Tap:

- **Provider cost per 1M tokens** is calculated from GPU rental prices (min / median / max across providers).  
- **Sell price per 1M tokens** comes from `price_sheet.csv` (unit_price_eur_per_1k_tokens × 1000).  
- **Gross margin** = Sell price − Provider cost.  

---

### 2.1 Model Economics (per 1M tokens)

| Model | GPU (median) | Cost €/1M (min) | Cost €/1M (median) | Cost €/1M (max) | Sell €/1M | Gross Margin €/1M | Gross Margin % |
|-------|--------------|----------------:|-------------------:|----------------:|----------:|------------------:|---------------:|
| Llama-3.1-8B | RTX 3090 | 3.60 | 7.46 | 11.32 | 0.00 | 0.00 | 0.00 |
| Llama-3.1-70B | RTX 3090 | 12.61 | 26.11 | 39.62 | 0.00 | 0.00 | 0.00 |
| Mixtral-8x7B | RTX 3090 | 10.51 | 21.76 | 33.02 | 0.00 | 0.00 | 0.00 |
| Mixtral-8x22B | RTX 3090 | 15.76 | 32.64 | 49.52 | 0.00 | 0.00 | 0.00 |
| Qwen2.5-7B | RTX 3090 | 3.60 | 7.46 | 11.32 | 0.00 | 0.00 | 0.00 |
| Qwen2.5-32B | RTX 3090 | 8.41 | 17.41 | 26.41 | 0.00 | 0.00 | 0.00 |
| Qwen2.5-72B | RTX 3090 | 12.61 | 26.11 | 39.62 | 0.00 | 0.00 | 0.00 |
| Yi-1.5-6B | RTX 3090 | 3.60 | 7.46 | 11.32 | 0.00 | 0.00 | 0.00 |
| Yi-1.5-9B | RTX 3090 | 4.20 | 8.71 | 13.21 | 0.00 | 0.00 | 0.00 |
| Yi-1.5-34B | RTX 3090 | 8.41 | 17.41 | 26.41 | 0.00 | 0.00 | 0.00 |
| DeepSeek-Coder-6.7B | RTX 3090 | 4.20 | 8.71 | 13.21 | 0.00 | 0.00 | 0.00 |
| DeepSeek-Coder-33B | RTX 3090 | 9.01 | 18.65 | 28.30 | 0.00 | 0.00 | 0.00 |
| DeepSeek-Coder-V2-16B | RTX 3090 | 7.01 | 14.51 | 22.01 | 0.00 | 0.00 | 0.00 |
| DeepSeek-Coder-V2-236B | RTX 3090 | 14.01 | 29.02 | 44.02 | 0.00 | 0.00 | 0.00 |

#### 2.1.1 Graph — Model Margins

![Model margins per 1M tokens](charts/model_margins_per_1m.png)

---

### 2.2 Observations

- Models with **negative margin** at median provider prices → move to **Private Tap only**.  
- Models with **stable positive margin** → safe to include in Public Tap.  
- Provider cost ranges already include FX buffer of 5.0%.  
- This table is the **core justification** that the Public Tap can be run profitably.  

---

## 3) Public Tap — Monthly Projection Scenarios

The following scenarios assume:

- All revenue is prepaid.  
- Costs scale linearly with tokens served.  
- Provider costs use **median GPU rental prices** (FX buffer applied).  
- Marketing allocation: 20% of inflow.  

---

### 3.1 Scenario Table (per month)

| Case      | Tokens Sold (M) | Revenue (€) | COGS (€) | Gross Margin (€) | Gross Margin % | Fixed+Loan (€) | Marketing (€) | Net Result (€) |
|-----------|----------------:|------------:|---------:|-----------------:|---------------:|---------------:|--------------:|---------------:|
| Worst     | 1.0 | 0.0 | 17.39 | -17.39 | 0.0 | 3748.75 | 0.0 | **-3766.14** |
| Baseline  | 5.0 | 0.0 | 86.94 | -86.94 | 0.0 | 3748.75 | 0.0 | **-3835.69** |
| Best      | 15.0 | 0.0 | 260.82 | -260.82 | 0.0 | 3748.75 | 0.0 | **-4009.57** |

#### 3.1.1 Chart — Scenario Components

![Public scenarios (stacked)](charts/public_scenarios_stack.png)

#### 3.1.2 Mermaid — Baseline Components (Pie)

```mermaid
pie title Baseline Monthly Components
  "Revenue €0.0" : 0.0
  "COGS €86.94" : 86.94
  "Marketing €0.0" : 0.0
  "Fixed+Loan €3748.75" : 3748.75
```

---

### 3.2 Break-even

- **Total fixed monthly costs (personal + business + loan):** €3748.75  
- **Required margin to break even (fixed + marketing):** €None  
- **Required prepaid inflow:** €None  

#### 3.2.1 Chart — Break-even

![Break-even inflow](charts/break_even.png)

---

### 3.3 Notes

- Public Tap scales with demand; no idle GPU rentals.  
- Negative net result in **worst case** only reduces profit — not cash runway, since all inflows are prepaid.  
- Best case shows upside potential if adoption is strong.  

---

## 4) Private Tap — Profitability Rules

Private Tap clients prepay for **dedicated GPU-hours** plus a **management fee**.  
Python calculates profitability per GPU as follows:

- **Provider cost €/hr (median)** from `gpu_rentals.csv`  
- **Markup target** from `price_sheet.csv` (% over provider cost)  
- **Sell price €/hr** = Provider cost + Markup  
- **Gross margin €/hr** = Sell price − Provider cost  
- **Management fee €/mo** = fixed fee added to each client  

---

### 4.1 Table — GPU Economics (per hour)

| GPU Model | Provider Cost €/hr (median) | Markup % | Sell Price €/hr | Gross Margin €/hr |
|-----------|----------------------------:|---------:|----------------:|------------------:|
| A10 | 0.95 | 50.00 | 1.42 | 0.47 |
| A100 40GB (PCIe) | 0.94 | 50.00 | 1.41 | 0.47 |
| A100 80GB (SXM/PCIe) | 1.70 | 50.00 | 2.55 | 0.85 |
| H100 80GB (PCIe/SXM) | 2.64 | 50.00 | 3.97 | 1.32 |
| H200 141GB | 6.08 | 50.00 | 9.11 | 3.04 |
| L4 | 0.63 | 50.00 | 0.95 | 0.32 |
| L40S | 0.59 | 50.00 | 0.88 | 0.29 |
| RTX 3090 | 0.75 | 50.00 | 1.13 | 0.38 |
| RTX 4090 | 0.52 | 50.00 | 0.78 | 0.26 |

#### 4.1.1 Chart — GPU Economics

![Private Tap GPU economics](charts/private-gpu-economics.png)

---

### 4.2 Example Client Pack

| Hours Prepaid | GPU Model | Revenue (€) | Provider Cost (€) | Gross Margin (€) | Management Fee (€) | Total Gross Margin (€) |
|---------------|-----------|------------:|-----------------:|-----------------:|-------------------:|-----------------------:|


---

### 4.3 Notes

- **Prepaid only** → no unpaid usage risk.  
- GPUs rented only after payment → no idle cost.  
- Management fee ensures **baseline profitability** even with low GPU usage.  
- Larger prepaid packs amplify gross margin.  

---

## 5) Worst/Best Case Projections

Scenarios combine **Public Tap** and **Private Tap** economics.  
All revenue is prepaid; no refunds. Costs scale linearly with demand.

---

### 5.1 Monthly Scenarios (snapshot)

| Case     | Public Revenue (€) | Private Revenue (€) | Total Revenue (€) | Total COGS (€) | Gross Margin (€) | Fixed+Loan (€) | Marketing (€) | Net (€) |
|----------|-------------------:|--------------------:|------------------:|---------------:|-----------------:|---------------:|--------------:|--------:|
| Worst    | 0.0 | 0.0 | 0.0 | 17.39 | -17.39 | 3748.75 | 0.0 | **-3766.14** |
| Baseline | 0.0 | 0.0 | 0.0 | 86.94 | -86.94 | 3748.75 | 0.0 | **-3835.69** |
| Best     | 0.0 | 0.0 | 0.0 | 260.82 | -260.82 | 3748.75 | 0.0 | **-4009.57** |

---

### 5.2 Yearly Projections (12 months)

| Case     | Total Revenue (€) | Total COGS (€) | Gross Margin (€) | Fixed+Loan (€) | Marketing (€) | Net (€) |
|----------|------------------:|---------------:|-----------------:|---------------:|--------------:|--------:|
| Worst    |  | 208.68 | -208.68 | 44985.0 | 0.0 | **-45193.68** |
| Baseline |  | 1043.28 | -1043.28 | 44985.0 | 0.0 | **-46028.28** |
| Best     |  | 3129.84 | -3129.84 | 44985.0 | 0.0 | **-48114.840000000004** |

---

### 5.3 Loan-Term Projection (60 months)

| Case     | Total Revenue (€) | Total COGS (€) | Gross Margin (€) | Fixed+Loan (€) | Marketing (€) | Net (€) |
|----------|------------------:|---------------:|-----------------:|---------------:|--------------:|--------:|
| Worst    |  | 1043.4 | -1043.4 | 224925.0 | 0.0 | **-225968.4** |
| Baseline |  | 5216.4 | -5216.4 | 224925.0 | 0.0 | **-230141.4** |
| Best     |  | 15649.199999999999 | -15649.199999999999 | 224925.0 | 0.0 | **-240574.2** |

---

### 5.4 Notes

- Fixed+Loan already includes **monthly loan repayment €748.75** × 60 months.  
- Marketing allocation: 20% of inflows reserved each period.  
- Net values reflect all obligations — showing repayment ability across full loan term.  

---

## 6) Loan Schedule (60 Months)

Loan request: €None  
Interest: None% flat, term None months  
Monthly payment: **€748.75**  
Total repayment: **€44925.0**  
Total interest: **€14925.0**

---

### 6.1 Repayment Table

| Month | Opening Balance (€) | Interest (€) | Principal (€) | Payment (€) | Closing Balance (€) |
|------:|--------------------:|-------------:|--------------:|------------:|--------------------:|


#### 6.1.1 Chart — Loan Balance Over Time

![Loan balance over time](charts/loan_balance_over_time.png)

---

### 6.2 Notes

- Flat interest = equal monthly payments of €748.75.  
- Payment is included in **fixed monthly costs** in all scenarios.  
- Repayment is funded from **prepaid revenue margin** (no credit risk).  
- Closing balance reaches **€0** at month 60.  

---

## 7) Taxes & VAT Set-Aside

All sales are subject to VAT rules in the Netherlands/EU.

---

### 7.1 VAT Policy

- **Standard VAT rate:** 21%  
- **EU B2B reverse-charge:** True  
- **Stripe Tax:** True  
- **Revenue recognition:** prepaid liability until consumed (prepaid liability until consumed)

---

### 7.2 VAT Set-Aside Table (examples)

| Gross Revenue (€) | VAT Rate % | VAT Set-Aside (€) | Net Revenue (€) |
|------------------:|-----------:|-----------------:|----------------:|
| 1000 | 21 | 210.0 | 790.0 |
| 10000 | 21 | 2100.0 | 7900.0 |
| 100000 | 21 | 21000.0 | 79000.0 |

---

### 7.3 Notes

- VAT is automatically collected via Stripe and reserved in a separate account.  
- Net revenue (after VAT) is what funds costs, loan repayment, and margin.  
- EU B2B reverse-charge invoices show €0 VAT but still appear in returns.  
- No risk of “spending VAT by accident” since funds are earmarked.  

---

## 8) Assurances for Lender

This business model is designed to minimize financial risk:

- **Prepaid only** — no service without an active balance.  
- **Non-refundable credits** — all inflows are final (except where legally required).  
- **No idle GPUs** — rentals only occur after prepaid demand is confirmed.  
- **Linear scaling** — each €1 prepaid corresponds to profitable capacity; no over-extension.  
- **Loan repayment embedded in fixed costs** — €748.75 per month for 60 months is always budgeted.  
- **FX buffer applied** — protects against USD/EUR currency swings on GPU rentals.  
- **Marketing spend capped as % of inflow** — prevents runaway acquisition costs.  
- **VAT separated** — reserved at collection, ensuring compliance.  

---

### Why this matters

- No credit risk from customers.  
- No exposure to hardware depreciation (no owned GPUs).  
- No cashflow gaps: inflows always precede outflows.  
- Clear, predictable 60-month repayment plan.  

---

## 9) Appendices

### 9.1 Data Sources

- **Configuration:** `config.yaml` (policies, limits, finance controls)  
- **Costs:** `costs.yaml` (fixed monthly overhead)  
- **Loan:** `lending_plan.yaml` (amount, term, interest, repayment plan)  
- **Pricing:** `price_sheet.csv` (per-model Public Tap prices, Private Tap fees, services)  
- **Models:** `oss_models.csv` (open-source models with parameters, context sizes, licenses)  
- **GPUs:** `gpu_rentals.csv` (provider prices, VRAM, sources)  

---

### 9.2 Engine Outputs (Generated by Python)

- **Model economics:** `model_price_per_1m_tokens.csv`  
- **Scenario tables:** `public_tap_scenarios.csv`, `private-tap-economics.csv`  
- **Break-even targets:** `break_even_targets.csv`  
- **Loan schedule:** `loan_schedule.csv`  
- **VAT set-aside:** `vat_set_aside.csv`  
- **Risk buffers:** `fx_sensitivity.csv`, `provider_price_drift.csv`  

---

### 9.3 Engine Version

- Finance Engine: vv1.0.0  
- Last generated: 2025-09-27T11:10:15Z  

---

### 9.4 Notes

- All monetary values expressed in **EUR** unless otherwise stated.  
- Provider GPU prices in **USD/hour** converted with FX rate 1.08 and buffer 5.0%.  
- Throughput (tokens/sec) is assumed until measured by **llama-orch** telemetry.  
- Template designed to show **inputs, outputs, and safeguards** clearly to lenders.  