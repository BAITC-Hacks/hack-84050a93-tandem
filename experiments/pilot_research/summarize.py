"""Build comparison tables from preserved diagnostic journals."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.pilot_research.measure import summarize

OUT = ROOT / "reports/pilot_research"


def load(prefix, development=False):
    files = [OUT / (prefix + "_diagnostics.jsonl")]
    if development:
        files.append(OUT / (prefix + "_remaining.jsonl"))
    rows = [json.loads(line) for file in files for line in file.read_text(encoding="utf-8").splitlines()]
    keys = [(r["scenario"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate attempts in selected comparison")
    return rows


def paired(base, candidate):
    left = {(r["scenario"], r["seed"]): r for r in base}
    right = {(r["scenario"], r["seed"]): r for r in candidate}
    if left.keys() != right.keys():
        raise ValueError("unpaired seeds/scenarios")
    result = {}
    for scenario in sorted({r["scenario"] for r in base}):
        keys = [key for key in left if key[0] == scenario]
        differences = [right[k]["net"] - left[k]["net"] for k in keys
                       if left[k].get("net") is not None and right[k].get("net") is not None]
        result[scenario] = {
            "paired_scored": len(differences), "runs": len(keys),
            "mean_delta": float(np.mean(differences)) if differences else None,
            "median_delta": float(np.median(differences)) if differences else None,
            "wins": sum(d > 1e-8 for d in differences),
            "ties": sum(abs(d) <= 1e-8 for d in differences),
            "losses": sum(d < -1e-8 for d in differences),
        }
    return result


def number(x):
    return "NA" if x is None else f"{x:,.0f}".replace(",", " ")


def main():
    development = {tag: load("dev_" + tag, True) for tag in ("baseline", "sized", "stop")}
    holdout = {tag: load("holdout_" + tag) for tag in ("baseline", "stop")}
    result = {"development": {k: summarize(v) for k, v in development.items()},
              "holdout": {k: summarize(v) for k, v in holdout.items()},
              "paired_development": {k: paired(development["baseline"], development[k]) for k in ("sized", "stop")},
              "paired_holdout": paired(holdout["baseline"], holdout["stop"])}
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Сравнение пилотных политик", "", "Все значения — синтетические у.е. и локальные модели.", ""]
    for phase in ("development", "holdout"):
        stats = result[phase]
        tags = ["baseline", "sized", "stop"] if phase == "development" else ["baseline", "stop"]
        lines += [f"## {phase}: медиана net", "", "| Сценарий | " + " | ".join(tags) + " |",
                  "| --- | " + " | ".join("---:" for _ in tags) + " |"]
        for scenario in stats["baseline"]:
            lines.append("| " + scenario + " | " + " | ".join(number(stats[t][scenario]["median_net"]) for t in tags) + " |")
        lines += [""]
    stats = result["holdout"]
    lines += ["## Holdout: нижний хвост и корректные положительные запуски", "",
              "| Сценарий | q10 B → C | min B → C | positive B → C | выигрыш / равно / хуже |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for scenario, b in stats["baseline"].items():
        c = stats["stop"][scenario]
        p = result["paired_holdout"][scenario]
        lines.append(f"| {scenario} | {number(b['q10_net'])} → {number(c['q10_net'])} | "
                     f"{number(b['min_net'])} → {number(c['min_net'])} | "
                     f"{b['valid_positive']}/{b['runs']} → {c['valid_positive']}/{c['runs']} | "
                     f"{p['wins']} / {p['ties']} / {p['losses']} |")
    lines += ["", "## Holdout: средние расходы на пилоты", "",
              "| Сценарий | Число B → C | Контакты B → C | Стоимость B → C | Потери B → C |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for scenario, b in stats["baseline"].items():
        c = stats["stop"][scenario]
        lines.append(f"| {scenario} | {b['mean_n_pilots']:.1f} → {c['mean_n_pilots']:.1f} | "
                     f"{number(b['mean_pilot_contacts'])} → {number(c['mean_pilot_contacts'])} | "
                     f"{number(b['mean_pilot_cost'])} → {number(c['mean_pilot_cost'])} | "
                     f"{number(b['mean_loss'])} → {number(c['mean_loss'])} |")
    lines += ["", "Потери = mean(max(-net, 0)). Pilot net считается отдельно с дедупликацией; "
              "он не равен предельному вкладу пилотов в финальный net при пересечениях.", "",
              "## Holdout: обнаружение и время", "",
              "| Сценарий | Проверен хороший offer B → C | Выбран в финале B → C | Среднее act, с B → C |",
              "| --- | ---: | ---: | ---: |"]
    for scenario, b in stats["baseline"].items():
        c = stats["stop"][scenario]
        lines.append(f"| {scenario} | {b['positive_discovery_runs']} → {c['positive_discovery_runs']} | "
                     f"{b['positive_deployment_runs']} → {c['positive_deployment_runs']} | "
                     f"{b['mean_agent_seconds']:.3f} → {c['mean_agent_seconds']:.3f} |")
    lines += ["", "B = baseline, C = StopAgent. Проверен хороший offer означает, что пилот "
              "затронул сочетание с положительным истинным gross-эффектом; это устанавливает "
              "только harness после действия. Это не гарантия положительного наблюдения, "
              "статистически достоверного обнаружения или положительного net.", "",
              "Все ошибки, таймауты, preflight, pilot net, суммарные расходы/контакты и "
              "wall-time доступны в comparison.json и исходных JSONL. Время измерено на "
              "одном компьютере при параллельных прогонах и не является контролируемым "
              "сравнением производительности.", ""]
    (OUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
