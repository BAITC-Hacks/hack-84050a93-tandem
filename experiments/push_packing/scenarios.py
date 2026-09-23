"""Four prespecified, unopened packing holdout configurations; harness only.

The previous overlay holdout has informed the packing hypothesis and is no
longer a holdout for it. These model/profile seeds are fixed before outcomes.
Reuse the original generator, including effect-independent positive placement.
"""
from experiments.push_overlay import scenarios as original

TARIFFS = ("tariff_1", "tariff_6", "tariff_10", "tariff_13", "tariff_18", "tariff_20")
NEW_HOLDOUT = ("packing_holdout_mixed", "packing_holdout_negative",
               "packing_holdout_rare", "packing_holdout_high_conversion")
NEW_CONFIGS = {
    NEW_HOLDOUT[0]: original.Scenario("packing_holdout", "mixed", 2026100101,
        TARIFFS, 2700, 36, (0.04, 0.55), (-0.30, -0.010), (0.20, 0.90)),
    NEW_HOLDOUT[1]: original.Scenario("packing_holdout", "negative", 2026100102,
        TARIFFS, 2700, 0, (0.04, 0.55), (-0.30, -0.010), (0.30, 0.90)),
    NEW_HOLDOUT[2]: original.Scenario("packing_holdout", "rare", 2026100103,
        TARIFFS, 2700, 1, (0.45, 0.95), (-0.20, -0.015), (0.35, 0.85)),
    NEW_HOLDOUT[3]: original.Scenario("packing_holdout", "high_conversion", 2026100104,
        TARIFFS, 2700, 36, (0.03, 0.60), (-0.30, -0.010), (0.90, 0.999)),
}
# Harness registration only. Agent modules never import fixtures or effect data.
original.CONFIGS.update(NEW_CONFIGS)
metadata = original.metadata
fixture = original.fixture
