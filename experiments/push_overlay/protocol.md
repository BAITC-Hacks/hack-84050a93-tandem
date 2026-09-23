# Frozen free-Push overlay candidate

Base: production `agent:Agent` at `75716aa`; candidate entry point
`experiments.push_overlay.agent:PushOverlayAgent` (also aliased as `Agent`).
One candidate. No parameter search or ranking changes after comparison starts.
The measurement runner records source hashes before evaluating either policy.

## Exact policy frozen before outcome measurements

1. Execute `super().act(env)` exactly once. Keep its paid pilots and returned
   campaigns in their original order, with every original campaign field intact.
2. Continue only if public channel `push` has a finite, numeric, exactly zero
   cost and finite conversion multiplier in (0, 1.2]. Require a valid reference
   channel multiplier in (0, 1]. Use latest finite posterior mean/std from each
   actually executed pilot in the baseline trace; never read model truth.
3. Enumerate each **original** final campaign and each target actually piloted in
   that campaign's exact current-tariff/ARPU cell. Copy all original audience
   filters exactly and change only target, channel (`push`), and campaign name.
   An overlay therefore selects exactly one original campaign's audience.
   Do not enumerate from appended overlays. Skip an identical existing Push
   target/filter combination. Derive audience membership/size from the public
   profile, including all data/call filters; do not estimate it from trace totals.
4. For ordering only, use normal reference-effect posteriors `(mu, sigma)`.
   Scale by public channel multiplier/reference multiplier. Candidate difference
   mean is `mu_push - mu_base`. For the same target, errors share a latent value:
   `sigma_difference = abs(push_scale - base_scale) * sigma_base` (correlation 1).
   For different targets, use `hypot(push_scale*sigma_target,
   base_scale*sigma_base)` (independence approximation, not an established fact).
   Rank by audience nonnegative finite ARPU mass times
   `clip(E[max(Normal(mean_difference, sigma_difference), 0)], 0, 2)`.
   Channel saturation and pilot coverage are omitted from this **ranking proxy**.
   The clip limits proxy ratios, not true model effects or measured score.
5. Sort all candidates once by descending proxy, then original campaign index,
   then target string. No re-ranking after an append. Append every feasible
   candidate in this order, including a zero-proxy candidate if capacity remains.
   Overlapping proxies are never added or reported as portfolio profit.
6. Keep at most 10 final campaigns and 5000 customers per campaign. Require final
   contacts <= public remaining contacts and <=15000 minus all public pilot
   contacts. Every append has zero cost; verify the entire plan against the
   remaining budget and contact limit after each prospective append. Skip an
   infeasible candidate; retain the unmodified base when no additions fit.
7. Record the preserved base trace separately. The experimental outer trace has
   actual new final campaigns, costs, total/unique contacts, and preflight.
   Its schema explicitly declares intentional overlap and nonadditive ranking.
   Do not pass it to the production renderer/exporter during this experiment.

## Mechanism and limitation

Under the unchanged public scorer, gain for an already contacted customer is the
maximum over their campaigns. Adding a zero-cost campaign on an already covered
audience cannot reduce that maximum. Preserving the complete prefix and staying
within all caps prevents appended contacts from displacing previous campaigns.
This is a conditional weak-dominance argument under the supplied score mechanics,
not proof of real-world causal uplift or of a strict improvement on hidden data.
A ranking proxy selects among valid additions; its accuracy is not needed for
the conditional non-decrease argument. Other scoring/channel rules may invalidate
that argument. The experiment must independently check caps and audience subsets.

## Focused implementation checks

Verify baseline prefix and pilots preserved, super called once, exact subset
audiences (including data/call filters), all resource limits, duplicates/no-free
Push/no-room cases, deterministic output, same-target shared uncertainty, actual
trace accounting, and absence of hidden-model imports/access in policy source.
Scorer imports belong only in tests/measurement code. Root submission and
production policy stay unchanged while the experiment is evaluated.

## Outcome measurement contract

Use the unchanged scorer for baseline and candidate, including all executed
pilots. Preserve each attempt and failure. Compare both existing scenarios and
fresh preregistered model configurations/noise seeds; freeze the runner's exact
models, seeds, hashes, and decision gates before those comparisons. Report paired
net difference and strict improvements, median/q10/min, campaign/contact costs,
validity/caps, prefix equality and pilot equality. Retain every deterioration
for diagnosis; do not silently discard errors or tune the candidate afterward.
No scenario/model IDs, effect tables, scorer, or evaluator internals enter policy.

### Preregistered series and integration gate

- Development: all 12 existing `pilot_research` scenarios plus the four
  `NEW_DEVELOPMENT` configurations in `scenarios.py`, noise seeds 60–62.
- Holdout, without changing candidate/ranking after development: the 12
  existing scenarios plus four different `NEW_HOLDOUT` configurations,
  noise seeds 70–79. Existing-model noise repetitions are not new models.
- Every pair records official score including pilots, preflight, dropped/capped
  campaigns, original-prefix identity, actual pilot-ID/observation hashes,
  recomputed audience inclusion, unchanged costs and unique audience, and limits.
- A measured deterioration below -1e-6, a changed pilot trajectory/prefix,
  or any invalid plan blocks integration pending diagnosis. Failures remain
  in reported attempts. Hypotheses cannot be silently retuned on the holdout.
- Require strict gains above 1e-6 on at least two holdout scenario families,
  including at least one new configuration, before recommending the added
  implementation/reporting complexity. Non-decrease without substantive gains
  is insufficient. This is an engineering gate, not a significance test.
- Production integration additionally requires trace, CSV, validator, report,
  channel explanations, documentation and repeatability to agree about the
  intentional overlapping suffix. An experiment passing this gate does not
  automatically make the production package ready.

Original production code remains `75716aa` during this experiment. A later
signed-coverage candidate is a separate change: any combined candidate needs
a new paired comparison against its own fixed base.
