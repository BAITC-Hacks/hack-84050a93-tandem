# One compaction-then-fill candidate

Incumbent: accepted `agent:Agent` from main b13f997. The old development and
holdout traces informed this hypothesis; they are no longer an unopened holdout
for this candidate. Freeze this source before any new outcome measurement.

Call incumbent once; pilots and its initial BaselineAgent prefix stay untouched.
Group only its free Push suffix by identical target, ARPU, data, and call filters.
Merge a group only when every current tariff is a distinct explicit singleton,
all audiences are nonempty, their sum is <=5000, and the semicolon-current filter
selects exactly their disjoint union. Retain the first group's campaign name and
position; sort currents lexically. Skip a whole group if it cannot fit. There is
one compaction pass, without iterative grouping or optimisation.

Traverse incumbent trace.overlay.candidates in its existing ranking order. Skip
selected candidates; reconstruct each remaining offer from its exact proven
anchor filters and target. Deduplicate by atomic current/target/channel/ARPU/data/
call keys, so merged campaigns cover their original offers semantically. Require
a real target pilot in the same cell. Append only whole nonempty anchors <=5000,
within 10 campaigns, remaining budget, remaining contacts and 15000 minus pilots.
New names are deterministic packing_push_NN, avoiding existing names. Never
modify the ranking, introduce smaller audiences, or compact newly appended offers.

If no new offer is appended, return the EXACT incumbent plan and keep the EXACT
incumbent trace, undoing pointless compaction. On changes, preserve incumbent
plan/trace separately and report actual packing/preflight/contact/cost details.
Before returning a changed plan, call independent `validate_packing` and raise
on any failed incumbent-preservation, coverage, target-evidence, or resource proof.
Do not add proxy gains or use a production renderer for this experimental schema.

The public scorer's max-per-customer rule makes exact impact-preserving repacking
neutral; zero-cost already-covered additions can weakly improve it. The evaluator
must independently check preservation of every incumbent customer/target/channel
impact, identical pilots/cost, covered added audiences, and absence of caps. This
is a conditional scorer argument, not a claim of real-world revenue or strict
hidden-score gain. No model truth, sampled pilot IDs, or scorer enters the policy.

## Frozen measurement and decision gates

Development: the 12 previously known scenarios plus the four previous overlay
development configurations, seeds 60:63 (48 pairs). These are reused development
data. Holdout: the same 12 known models with new observation seeds, plus four
new model/profile configurations fixed in scenarios.py, seeds 90:95 (80 pairs).
Only the latter four configurations test unseen model/profile configurations;
new seeds on the old 12 models test observation noise. Neither predicts judging.

Freeze candidate, validator, harness, generator registration and this protocol
in Git before measuring outcomes. Retain every attempt. No tuning after opening
the holdout. A shipping candidate needs zero invalid plans/errors/timeouts,
zero preservation or resource violations and zero paired net regressions in
both sets; at least eight strict holdout wins across at least two scenario
families, including one new model configuration. A changed production policy
also requires coherent trace/validator/report integration and a fresh acceptance
check. If that cannot be completed by 17:40 local on September 23, keep accepted
main b13f997 and retain this as isolated research. The user submits the project.
