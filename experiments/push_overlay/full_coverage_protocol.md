# FullCoveragePushAgent: second candidate, before opening holdout

This is a separately frozen development extension of PushOverlayAgent. Development
outcomes of the first candidate were already observed. Its source/protocol remain
unchanged. Do not present the second candidate as chosen before development.
Holdout outcomes must remain unopened until this source and comparison rules are
frozen. Entry point: `experiments.push_overlay.full_coverage_agent:FullCoveragePushAgent`.

## Single additional hypothesis

A baseline pilot sometimes samples every member of a public current-tariff/ARPU
cell. In that case, the publicly returned actual sample count equals the complete
public cell size, proving that every member was already contacted. A free Push
over that exact cell introduces no newly contacted customers even when the cell
does not occur in a baseline final campaign. The unchanged scorer's per-customer
maximum can therefore weakly increase, with no new contact cost, provided every
resource limit is respected and the original campaign prefix is preserved.

## Exact candidate and evidence

1. Call production `Agent.act` exactly once. Preserve all its pilots and all final
   campaigns, in their original order and with every original field unchanged.
2. Build ordered anchors: first the original final campaigns; then individual
   executed baseline pilots in chronological order when that **one** pilot's
   actual `n_customers` exactly equals the public size of its current-tariff/ARPU
   cell (positive and <=5000). Production pilots use whole cells before sampling.
   Reject any trace entry with an explicit nonempty data/call pilot filter.
   Missing data/call fields are interpreted only under this production trace
   contract. Do not sum partial counts, infer coverage from expectations, infer
   it from repeated positive responses, or read private sampled IDs.
3. Each final anchor has exact original filters and original target/channel.
   Each full-pilot anchor has current/ARPU filters, data/call=None, and its source
   pilot's target/channel. Record source, 1-based source pilot step, actual sample
   count, exact public cell size, and the equality evidence in the trace.
4. Enumerate all targets actually piloted in each anchor's current/ARPU cell.
   Copy that anchor's filters exactly; set Push, target, and a unique name. Use
   the latest reference posterior for each target from real baseline pilots.
5. Import the **unchanged v1 ranking helper**: public audience ARPU mass times
   clipped [0,2] Gaussian expected positive part of Push minus anchor effect.
   Same target uses correlation 1; different targets use the disclosed independence
   approximation. Channel saturation/pilot coverage remain omitted from ranking.
   Sort once by descending proxy, anchor order, target string. Final anchors
   precede pilot anchors on tied scores. Do not sum proxies into final profit.
6. Append feasible candidates, including zero proxies if room remains. Deduplicate
   identical Push target/filter combinations against both prefix and accepted
   suffix, regardless of which anchor generated them. Multiple full pilots of
   one cell may provide separate ranking references; they do not establish new
   coverage through a union of partial pilots.
7. Require finite public Push cost exactly zero, valid channel multipliers,
   <=10 final campaigns, <=5000 customers each, final contacts <= public remaining
   and <=15000 minus all pilot contacts, and unchanged remaining-budget checks.
   Validate the complete prospective plan at every append. Return the original
   prefix unchanged when no candidate fits. Keep the base trace separately and
   report actual final contacts/cost/preflight plus anchor evidence.

## Evaluation boundary

Neither policy nor helper imports scorer, effects, scenarios, environment
internals, or reads sampled IDs. Focused checks use public synthetic tables and
one actual baseline seed 42 for prefix/pilot/resource equivalence without net
scoring. The paired measurement harness independently verifies each appended
audience is within the union of customers actually contacted by baseline pilots
and final campaigns. Private sampled IDs are available only to that evaluator.
Record development and unopened holdout separately. Publish all candidates and
failures; do not pool their results or silently replace the first candidate.

## Frozen measurement schedule and decision

Development repeats the same 12 known models and four development configurations,
noise seeds 60–62. This is development data already inspected for v1. Then, with
this candidate unchanged, evaluate the 12 known models and four unopened holdout
configurations at noise seeds 70–79. Existing-model noise changes are not new
models. Use `measure.py --candidate experiments.push_overlay.full_coverage_agent:FullCoveragePushAgent`.

All pairs must retain identical pilot trajectories, original final prefix,
total communication cost and unique audience INCLUDING PILOTS, obey all limits,
and have no measured decline below -1e-6. Final-only unique audience may grow.
Any violation blocks integration. Require strict holdout gains in at least two
scenario families, including at least one new configuration, and report the
size and frequency rather than only their existence. No hidden-score guarantee.
Integration additionally requires the complete product/trace/report checks.

The initial v1 runner typo (official unique-audience field name) and all its
failed attempts remain recorded. It was fixed before valid v1 outcome comparisons.
Extending the audience proof to include complete pilots is a new policy, not a
retroactive correction to v1; both development records remain separate.
