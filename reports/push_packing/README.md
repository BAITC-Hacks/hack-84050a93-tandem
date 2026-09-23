# Push packing: KEEP ACCEPTED MAIN

Accepted product remains `b13f9974f12411418088e40d22bca2e060fed96a` (v1.3).
This isolated experiment did not meet its prespecified shipping gate.
No production change or platform submission was made.

| Set | Paired runs | Strict wins | Ties | Losses | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 48 | 7 | 41 | 0 | 0 |
| Holdout | 80 | 7 | 73 | 0 | 0 |

The [protocol](../../experiments/push_packing/protocol.md), committed before
outcomes, required at least eight strict holdout wins, including an unseen model
configuration. Observed seven: the threshold failed. We do not lower it after
seeing results. Median paired improvement is zero in both sets. There were no
observed invalid plans, timeouts, capped/dropped campaigns, effect-preservation
violations, or negative deltas. This is a modest synthetic extension, not evidence
of a competition-winning breakthrough.

## Frozen procedure

Policy and independent validator: `5781670`. Before any outcome measurement,
`a87690d` fixed a reporting issue: a failed nondecrease assertion must not exclude
negative deltas from the scored-pair distribution. Every attempt is retained in
JSONL. Both complete series report unchanged source hashes.

Development reused 12 known models and four prior overlay development models,
seeds 60–62. Holdout used new observation seeds 90–94 on the 12 known models plus
four new profile/model configurations, fixed in
[prespecified_configurations.json](prespecified_configurations.json) before any
measurement. Only the four new configurations test new model/profile data;
new seeds on old models test noise. Previous overlay holdout traces informed the
hypothesis and are not called fresh holdout here.

The candidate merges compatible, disjoint, free Push audiences, preserving every
incumbent customer/target/channel effect, then fills available campaign slots
with whole already-covered audiences in the old ranking order. It leaves the
exact incumbent plan/trace unchanged if it cannot add an offer. Coverage uses
public profiles and one complete pilot, never sums of partial pilots. Hidden
pilot IDs and model truth are used only in the independent evaluator.

## Evidence and reproduction

- [Decision and aggregate results](decision.json)
- [Development summary](development.json), [all development pairs](development.jsonl)
- [Holdout summary](holdout.json), [all holdout pairs](holdout.jsonl)
- [Frozen protocol](../../experiments/push_packing/protocol.md)
- 34 focused candidate/validator tests passed before outcomes (3.51 seconds).

From this research branch after installing requirements-dev.txt:

```bash
python experiments/push_packing/measure.py --phase development --seeds 60:63 --out reports/push_packing/rerun_development.json
python experiments/push_packing/measure.py --phase holdout --seeds 90:95 --out reports/push_packing/rerun_holdout.json
python -m pytest -q experiments/push_packing/test_packing.py experiments/push_packing/test_validation.py
```

An isolated production/report adapter was prototyped in a separate worktree while
measurement ran. It was not integrated after the outcome gate failed. The current
product and its independently accepted browser behavior remain v1.3. No claim
about hidden judging, a real-world deployment, or guaranteed placement follows
from these synthetic results.
