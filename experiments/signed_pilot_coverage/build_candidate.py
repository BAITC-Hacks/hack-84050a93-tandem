"""Reproduce the frozen candidate from unchanged root production sources."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source occurrence: {old!r}")
    return text.replace(old, new, 1)


def build():
    agent = (ROOT / "agent.py").read_text(encoding="utf-8")
    agent = replace_once(agent, "from strategy.portfolio import choose_portfolio, make_options",
                         "from experiments.signed_pilot_coverage.portfolio import choose_portfolio, make_options")
    agent = replace_once(agent, 'Path(__file__).resolve().parent / "data" / "change_tariff.csv"',
                         'Path(__file__).resolve().parents[2] / "data" / "change_tariff.csv"')
    portfolio = (ROOT / "strategy" / "portfolio.py").read_text(encoding="utf-8")
    portfolio = replace_once(portfolio, "pilot_effects.append((max(0.0, pilot_mean * scale), min(1.0, n / len(cell))))",
                             "pilot_effects.append((pilot_mean * scale, min(1.0, n / len(cell))))")
    old_formula = '''            # Individual pilot IDs are deliberately unknown. Under the public
            # uniform sampling rule, estimate prior coverage probabilistically.
            # Never count a pilot's gain again as incremental final-campaign gain.
            uncovered, sunk = 1.0, 0.0
            for pilot_ratio, probability in pilot_effects:
                sunk += uncovered * probability * min(max(0.0, ratio), pilot_ratio)
                uncovered *= 1.0 - probability
            return ratio - sunk'''
    new_formula = '''            # Individual pilot IDs are unknown; retain independent uniform coverage.
            # Descending signed effects assign each best prior contact its weight.
            # Uncontacted customers receive ratio; contacted customers can recover
            # a negative prior effect as well as improve a positive prior effect.
            uncovered, increment = 1.0, 0.0
            for pilot_ratio, probability in pilot_effects:
                increment += uncovered * probability * max(0.0, ratio - pilot_ratio)
                uncovered *= 1.0 - probability
            return uncovered * ratio + increment'''
    portfolio = replace_once(portfolio, old_formula, new_formula)
    (DEST / "agent.py").write_text(agent, encoding="utf-8", newline="\n")
    (DEST / "portfolio.py").write_text(portfolio, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    build()
