# 12 — Operator Variables (CSV Spec & Overlay)

 Status: Draft
 Version: 0.1.0

## 1. Doel & Scope

- **Doel.** Definieert het variabelenmodel (instelbare knoppen) voor D3. Variabelen worden geleverd als CSV en worden door de engine over de constante operator‑YAML heen gelegd.
- **Scope.** CSV‑schema, validatieregels, allowed scopes en paden, overlay‑precedence, treatments (fixed/low_to_high/random), determinisme met seed.

## 2. CSV‑schema (MUST)

 Elke variabelen‑CSV (per scope) MUST de volgende kolommen hebben:

- `variable_id` (string, uniek binnen en liefst over alle CSV’s)
- `scope` (enum: `general` | `public_tap` | `private_tap`)
- `path` (dot‑pad binnen de operatorconfig volgens scope, zie §3)
- `type` (enum: `numeric` | `discrete`)
- `unit` (bijv. `percent` | `fraction` | `EUR` | `EUR_per_month` | `tokens` | `months` | `enum`)
- `min` (leeg voor `discrete`; numeriek voor `numeric`)
- `max` (leeg voor `discrete`; numeriek voor `numeric`)
- `step` (leeg voor `discrete`; numeriek >0 voor `numeric`)
- `default` (MUST binnen [min,max] en op step‑rooster voor `numeric`; of geldige enumwaarde voor `discrete`)
- `treatment` (enum: `fixed` | `low_to_high` | `random`)
- `notes` (vrije tekst; voor `discrete` MUST een `values=a|b|c` hint bevatten)

 Extra kolommen zijn toegestaan maar worden genegeerd door de engine.

### 2.1 Units (normatief)

- `percent` = 0..100 als getal (geen %‑teken in CSV).
- `fraction` = 0..1.
- Monetary: `EUR`, `EUR_per_month` (float ≥ 0).
- Tijd/hoeveelheid: `months`, `tokens` (float ≥ 0).
- `enum` uitsluitend met `type=discrete` en `notes` met `values=...`.

## 3. Scope → Allowed path roots (MUST)

- `general` → `finance.*`, `insurances.*`, `tax.*`, `reserves.*`.
- `public_tap` → `pricing_policy.public_tap.*`, `prepaid_policy.credits.*`, `acquisition.*`, `autoscaling.*`.
- `private_tap` → `pricing_policy.private_tap.*`, `pricing_policy.private_tap.vendor_weights.*`, `pricing_policy.private_tap.management_fee.*`.

Variabelen MOGEN geen paden buiten bovenstaande roots aanwijzen. Pogingen → **ERROR**.

## 4. Validatie (MUST)

{{ ... }}
- `variable_id` uniek per CSV; duplicates over meerdere CSV’s SHOULD worden vermeden (WARNING).
- `scope` en `type` en `treatment` waarden exact uit de toegestane sets.
- `numeric`: `min <= default <= max`, `step > 0`, en `(default - min) % step == 0` na kwantisatie.
- `discrete`: `default` MUST in de `values`‑lijst uit `notes` staan.
- `path` MUST een bestaand/ondersteund pad in de operator‑YAML zijn (zie inputs/operator/). Onbekend → **ERROR**.

## 5. Overlay & Precedence (MUST)

- Operator‑YAML (constants) vormen de basis.
- Variabelen worden per run over die basis gelegd volgens `path`.
- Bij conflict (constante key én variabele op hetzelfde pad) **wint de variabele**. Dit wordt gelogd als `variable_override` met CSV‑bron en pad.
- `simulation.yaml → run.fail_on_warning: true` MAY escaleren naar **ERROR** bij overrides.

## 6. Treatments & sampling (MUST)

- `fixed`: gebruik `default` exact, geen sampling.
- `low_to_high` (grid): genereer roosterwaarden `min, min+step, …, max`. De engine maakt de cartesische combinatie over alle `low_to_high` variabelen binnen dezelfde scope.
- `random`: sample uniform in `[min, max]` en kwantiseer naar dichtstbijzijnde step‑tik (offset `min`). Aantal onafhankelijke draws wordt gestuurd door `simulation.yaml → run.random_runs_per_simulation`.
- `discrete` + `random`: trek met gelijke kans uit de `values`‑lijst.
- Alle randomisatie MUST deterministisch zijn o.b.v. de seed (zie §7).

 Opmerking: combinatorische explosie SHOULD worden beheerst via kleine roosters en door `random` in te zetten; de engine MAY een limiet afdwingen en anders **ERROR** geven met hint.

## 7. Determinisme & Seed (MUST)

- Seed resolutie:
   1) `simulation.stochastic.random_seed` (indien gezet)
   2) `simulation.run.random_seed` (indien gezet)
   3) Pijplijnseed: `inputs/operator/public_tap.yaml: meta.seed` of `private_tap.yaml: meta.seed`
   4) Geen seed → **ERROR**
- Identieke inputs + seed geven byte‑gelijke outputs.

## 8. Voorbeelden (samenvattend)

 Uit `inputs/variables/public_tap.csv`:

 ```csv
 variable_id,scope,path,type,unit,min,max,step,default,treatment,notes
 target_margin_pct,public_tap,pricing_policy.public_tap.target_margin_pct,numeric,percent,20,80,5,55,low_to_high,"Target gross margin % over GPU cost"
 acq_budget,public_tap,acquisition.budget_monthly_eur,numeric,EUR,0,10000,100,500,low_to_high,"Monthly acquisition budget"
 acq_google_ads_share,public_tap,acquisition.channel_allocation.google_ads,numeric,fraction,0,1,0.1,0.5,random,"Channel share"
 ```

 Aanvullende variabelen voor groei & churn (Public):

 ```csv
 variable_id,scope,path,type,unit,min,max,step,default,treatment,notes
 acq_budget_month0,public_tap,acquisition.budget_month0_eur,numeric,EUR,0,20000,100,1000,low_to_high,"Month 0 acquisition budget"
 acq_budget_growth_mom,public_tap,acquisition.budget_growth_pct_mom,numeric,percent,0,50,1,8,fixed,"MoM budget growth %"
 churn_pct_mom_public,public_tap,acquisition.churn_pct_mom,numeric,percent,0,50,0.5,3,fixed,"Monthly churn % for active customers (public)"
 tokens_per_conversion,public_tap,acquisition.tokens_per_conversion_mean,numeric,tokens,0,200000,1000,50000,random,"Average tokens per new customer conversion"
 cac_fallback_public,public_tap,acquisition.cac_base_eur,numeric,EUR,0,500,5,50,fixed,"Fallback CAC if channel-driven CAC not configured"
 ```

 Autoscaling (Public):

 ```csv
 variable_id,scope,path,type,unit,min,max,step,default,treatment,notes
 autoscale_target_util,public_tap,autoscaling.target_utilization_pct,numeric,percent,1,100,1,75,fixed,"Target sustained utilization %"
 autoscale_peak_factor,public_tap,autoscaling.peak_factor,numeric,fraction,1,3,0.05,1.2,fixed,"Peak multiplier over average demand (p95)"
 autoscale_min_instances,public_tap,autoscaling.min_instances_per_model,numeric,count,0,100,1,0,fixed,"Minimum instances per model"
 autoscale_max_instances,public_tap,autoscaling.max_instances_per_model,numeric,count,1,1000,1,100,fixed,"Maximum instances per model"
 ```

 Uit `inputs/variables/private_tap.csv`:

 ```csv
 variable_id,scope,path,type,unit,min,max,step,default,treatment,notes
 default_markup_pct,private_tap,pricing_policy.private_tap.default_markup_over_provider_cost_pct,numeric,percent,0,100,5,40,low_to_high,"Markup % over provider GPU cost"
 management_fee,private_tap,pricing_policy.private_tap.management_fee_eur_per_month,numeric,EUR,0,1000,50,199,low_to_high,"Monthly management fee"
 vendor_weight_cost,private_tap,pricing_policy.private_tap.vendor_weights.cost,numeric,fraction,0,1,0.1,0.5,low_to_high,"Vendor score weight: cost"
 ```

 Aanvullende variabelen voor groei & churn (Private):

 ```csv
 variable_id,scope,path,type,unit,min,max,step,default,treatment,notes
 private_budget_month0,private_tap,acquisition.budget_month0_eur,numeric,EUR,0,20000,100,500,low_to_high,"Month 0 acquisition budget (private)"
 private_budget_growth_mom,private_tap,acquisition.budget_growth_pct_mom,numeric,percent,0,50,1,6,fixed,"MoM budget growth % (private)"
 churn_pct_mom_private,private_tap,acquisition.churn_pct_mom,numeric,percent,0,50,0.5,3,fixed,"Monthly churn % for active clients (private)"
 hours_per_client_month,private_tap,acquisition.hours_per_client_month_mean,numeric,hours,0,200,1,10,random,"Average monthly hours per new private client"
 cac_fallback_private,private_tap,acquisition.cac_base_eur,numeric,EUR,0,1000,10,200,fixed,"Fallback CAC if channel-driven CAC not configured (private)"
 ```

 Uit `inputs/variables/general.csv`:

 ```csv
 variable_id,scope,path,type,unit,min,max,step,default,treatment,notes
 marketing_allocation,general,finance.marketing_allocation_pct_of_inflow,numeric,percent,0,100,5,20,low_to_high,"% of inflow allocated to marketing"
 vat_buffer_policy,general,tax.vat_buffer_policy,numeric,percent,0,100,10,100,fixed,"Percent of VAT collected that you park aside"
 founder_draw_policy,general,reserves.founder_draw_policy,discrete,enum,,,,flat,fixed,"values=flat|percent_of_profit"
 ```

## 9. Afbakening

- Variabelen mogen geen `facts/` datasets overschrijven.
- Geen externe bronnen of andere formaten dan CSV voor variabelen.
- Geen backwards compatibility‑garanties pre‑1.0.
