# Signed pilot coverage: preregistered protocol

Base: `75716aa8b4dec4bcb9eeb471709c3caa19529211`. One candidate; no parameter search or tuning after measurement. This file, candidate, tests, measurement wrappers and `reports/signed_pilot_coverage/freeze.json` are committed before the first paired evaluation. The freeze commit is recovered with `git log -1 --format=%H -- reports/signed_pilot_coverage/freeze.json`.

## Intervention

For all pilots in the same current-tariff/ARPU cell (including other targets), use the latest posterior mean of their offer multiplied by that pilot's channel/reference scale, retaining its sign: `s_i`. Set `q_i=min(1,n_i/N)`, sort descending by `s_i`, and use `w_i=q_i*product(1-q_j for j<i)`, `p0=product(1-q_i)`.

`increment(r) = p0*r + sum(w_i*max(0,r-s_i))`.

Apply this to both mean and risk-adjusted gains, subtracting the entire final contact cost. Preserve all pilot policy, priors, posteriors, risk 1.65, channel scaling, candidate audiences, portfolio search, limits, official environment and scorer. The copied Agent has only the isolated portfolio import and the corrected path to root `data/change_tariff.csv`. No hidden state enters either policy; synthetic truth is used only by the unchanged measurement harness.

## Before measurement

- Run exhaustive small-subset formula tests, boundary cases, mixed signs and equivalence to the old formula for nonnegative inputs.
- Review `minimal.diff`; freeze normalized-LF source/data hashes, including the shared production sources and root artifacts. No source changes during development or holdout.
- Baseline `agent:Agent`; candidate `experiments.signed_pilot_coverage.agent:Agent`.

## Measurements

- Use `experiments.pilot_research.measure.bounded_run` and its existing 12 scenarios, unchanged official scorer. Development seeds 60–64; holdout seeds 70–79. These are 180 paired observations / 360 attempts, not 180 distinct models. No early selection or tuning between phases.
- Fixed per-attempt timeout 60 seconds; record every attempt, errors, timeouts and stdout/stderr in append-only JSONL. If the deadline forces truncation, report the exact completed volume and keep all partial attempts.
- Compare complete pilot diagnostic records, trace pilot actions, historical-prior status, reference channel, pilot net, pilot costs, pilot contacts and number of pilots exactly. Compare model hashes. This tests pilot budgets; final expenditures are allowed to differ.
- Save preflight, sanitizer drops, scorer caps, net including pilots, all expenses/contacts, final campaign count/sizes and elapsed time. Failed/invalid attempts remain in denominators; they never count as wins or valid positive runs.
- Cross-check the two policies using existing `tools.benchmark._run_row` on mock seed60 and `tools.stress_benchmark.run_scenario` on all five original stress scenarios seed60. Compare their scores with the diagnostic harness; keep these duplicate checks separate from inferential counts.
- Export seed42 CSV twice per policy through existing `tools.benchmark._run_worker('submission', ...)`; save CSVs here, never overwrite root `submission.csv`.
- Summarize each model and phase separately: median, linear-interpolated q10, minimum net, pair differences, wins/losses/ties, valid-positive count and rate. Numerical ties use absolute tolerance `1e-6`; do not sum money across models.

## Fixed decision gates

1. All attempts finish with valid preflight, no errors/timeouts/drops/caps or limit violations.
2. Exact pilot trajectories and pilot resources match in every pair; root history is actually loaded.
3. Positive median paired net change on at least two stress models, including at least one of `devin_all_bad`, `negative_1`, `negative_2` (predominantly negative effects).
4. No decrease in median, q10 or valid-positive count on protected models `mock`, `devin_mixed`, `mixed_1`, `mixed_2`. Apply the checks separately to development and holdout; holdout is decisive, and development regressions are also disclosed.
5. Disclose every other deterioration, especially a lower minimum or a lost profitable run. Mixed or zero results mean KEEP BASELINE under the deadline. These are conservative engineering gates, not a significance test or a prediction of hidden judging.

No changes to production or main. A potential future transfer must also update the duplicated formula and source-hash guard in `tools/channel_explanations.py`, regenerate fresh output manifests and undergo independent review. That transfer is outside this experiment.
