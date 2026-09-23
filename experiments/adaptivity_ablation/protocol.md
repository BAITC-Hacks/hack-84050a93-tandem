# Frozen adaptivity ablation protocol

Base: `26864b1760e794134af0b7ebf69c109891e7f71d` (fresh origin/main).
One candidate only, `FixedSurveyAgent`. No production integration. This protocol,
source, input hashes, and seed-0–4 reproduction are committed before any seed-50–59
comparison. The comparison report records that source commit. No retuning afterward.

## Exact intervention and ranking

`build_candidate.py` transforms a copy of base `agent.py`. Imports/class name and
the history path change to support its new directory. Before the unchanged pilot
selection loop, save pristine beliefs and substitute a public-data-only resource
ledger. Run the original selection loop in this ledger to produce the entire
schedule. Each fantasy response equals that offer's **initial prior mean**, scaled
from reference channel to the scheduled channel. Counts/costs are deterministic
from public cells and channel prices. Fantasies reduce planning variance only;
they never enter the separate real beliefs or final plan. Restore real environment
and pristine beliefs, execute the saved schedule, update from real replies only,
then execute the byte-identical final planning/validation block.

No actual environment call occurs while scheduling. The candidate does not import
fixtures, scorer, model truth, scenario names, or evaluator internals. Its own ledger
contains only public inputs. Original Agent does not import the experiment.

Exact ranking inherited from the frozen base, with hypothetical posterior mean `m`
and standard deviation `s`, cell audience `N`, cell ARPU sum `M`, reference multiplier
`r`, action multiplier `u`, per-contact cost `c`:

1. Sort current tariff, ARPU stratum and target lexicographically. Preserve the same
   historical priors and duplicate-offer filter. Try requested sizes **100 then 200**,
   shrink to public cell size and affordable contacts; reject sizes below 10.
2. First up to six actions alternate HIGH/MID/LOW. The second round prefers another
   current tariff in that stratum. Coverage length is exactly
   `min(2 * number_of_present_strata, max(1, pilot_limit // 2))`.
3. Per cell, threshold `t = max(0, max_observed(m - 0.7*s))`; breadth
   `b = 1 / (1 + 0.45 * previous_cell_pilots)`. Per stratum, attempts count distinct
   tested offers and promising count uses `m > 1.65*s`; weight
   `w = max(0.15, (promising+1)/(attempts+2))`, or 1 during coverage/confirmation.
4. Observation variance `v = 0.804²/n * (r/u)²`; next variance
   `s_next² = 1 / (1/s² + 1/v)`; future-mean spread `f = sqrt(s² - s_next²)`.
   `K = max(0, E[max(Normal(m-t,f),0)] - current_value)`, where current_value is
   `max(m-t,0)` for previously tested offers and 0 otherwise. During coverage use
   `K = max(m-t,0) + 0.25*K`.
5. Confirmation bonus `C = max(0,s-s_next) * 0.65` during confirmation, otherwise
   multiply by 0.25; set C to zero for untested offers or `m < t`.
   Rank by `M*(K*b*w + C) + min(n*M/N*m*u/r,0) - n*c - 0.35*n*max(t,0)*M/N`.
   Ties retain the first sorted offer and first size. The existing untested-offer
   cell residual shift remains `clip(0.4*median(residuals), -0.12, 0.12)` after two
   tested offers; with prior-mean fantasies it is zero apart from rounding.
6. Repeats are assigned by this same deterministic score. During the last five
   slots (step >= max(1, pilot_limit-5)), prefer already scheduled offers with
   1–2 prior measurements and `m-0.7*s > 0`. Otherwise use the full pool.
   Stop schedule creation if no feasible action or best score <= 0 after slot 1.
7. Preserve pilot budget `min(18% initial budget,18000)`, final contact reserve
   `min(2500,max(1,initial_contacts//3))`, <=20 pilots. Reference channel and free
   channel fallback are unchanged. During real replay only resource availability
   may shrink/skip/stop actions; observed effects cannot alter remaining actions.

All final risk correction (1.65), option generation, posterior update and portfolio
selection are retained. This removes outcome-dependent exploration, including
outcome-dependent repeats/stopping, while preserving the same planning heuristic.
It does not isolate each individual adaptive feature from other adaptive features.
Fixed scheduling has additional compute overhead, which is measured in agent time.

## Measurements selected before comparison

- Reproduce Agent at mock seeds 0–4. Match net, total cost/contacts, pilot count,
  preflight/status against `benchmark_review_agent.json`; pilot cost/contacts also
  against saved `pilot_research/dev_baseline_*` journals. Absolute net tolerance 1e-6.
- New series: seeds **50–59**, both policies, **mock, mixed_1, negative_1, rare_1,
  mixed_2, negative_2, rare_2** from existing pilot_research fixtures. 140 attempts.
  Both variants are included up front, not selected after seeing results.
- Sequential bounded subprocesses, 45 seconds per attempt, alternating policy order.
  Persist each attempt immediately to append-only JSONL; retain exceptions/timeouts
  and invalid outcomes. Never overwrite earlier attempts.
- Use unchanged `scoring_core.score_campaigns` and sanitizer, scoring paid pilots
  even on invalid/error final output. Preflight/caps/dropped campaigns independently
  determine validity. No synthetic truth enters either candidate.
- Per scenario/policy: median/q10/min net among scored attempts, valid-positive
  fraction over **all attempts**, status counts, expense/contact/pilot/time means.
  All raw per-pair records retain these metrics and preflight details. Paired delta
  is **adaptive minus fixed**, only when both valid; retain unmatched failures.
  Report median/q10/min/max delta, win/tie counts; min delta is worst adaptive
  deterioration, max delta is worst fixed deterioration. Never sum across models.
- Rare discovery means at least one true-positive tariff/stratum pair was probed;
  deployment means a true-positive pair appears in the final plan. Record whether
  a positive noisy observation occurred as a separate diagnostic. Discovery does
  not require total net to be positive. Truth is read by the evaluator only.

## Decision criteria (descriptive, no significance claim)

Safety gate: all candidate attempts must pass preflight, no exceptions/timeouts,
no discarded or capped campaigns. To recommend **consider candidate**, fixed must
have non-worse median and q10 net, non-worse valid-positive count in every scenario,
and non-worse discovery/deployment counts in both rare scenarios, with at least
one strict improvement. If adaptive satisfies the analogous dominance conditions,
recommend **keep current**, with scenario-specific effect sizes. Otherwise results
are **mixed / insufficient for superiority**; keep production unchanged. Always
show minima and worst paired deterioration even if a recommendation gate passes.
These are conservative engineering gates, not statistical proof or permission to
automatically integrate a candidate.

New seeds change noise on already known synthetic models. Equal seed does not
imply identical sampled clients after different actions. Net is synthetic and is
not real revenue. Ten repeats per scenario are a small study, not hidden judging.

## Reproduction

From repository root with the existing requirements-dev environment:

```sh
python -m pytest -q experiments/adaptivity_ablation/test_ablation.py
python experiments/adaptivity_ablation/measure.py --mode reproduce --out reports/adaptivity_ablation/reproduction_rerun.json
python experiments/adaptivity_ablation/measure.py --mode compare --out reports/adaptivity_ablation/comparison_rerun.json
```

Frozen input hashes use UTF-8 with LF normalization for Windows Git portability.
Full comparison needs a new output basename on each invocation.
