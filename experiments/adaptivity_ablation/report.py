"""Render frozen-series evidence without changing either policy or its inputs."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.adaptivity_ablation.verify import verify_frozen


def number(value):
    return f"{value:,.2f}".replace(",", " ") if value is not None else "—"


def main():
    frozen = verify_frozen()
    folder = ROOT / "reports/adaptivity_ablation"
    report = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / report["journal"]).read_text(encoding="utf-8").splitlines()]
    summary = report["summary"]
    audit = {"frozen_inputs_match": True, "attempts": len(rows), "per_scenario": {}}
    all_pairs = []
    for scenario in report["scenarios"]:
        fixed = [r for r in rows if r["scenario"] == scenario and r["policy"] == "fixed"]
        schedules = [r.get("agent_trace", {}).get("fixed_schedule") for r in fixed]
        hashes = {hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for s in schedules}
        def matches(row):
            schedule = row.get("agent_trace", {}).get("fixed_schedule", [])
            actual = row.get("pilot_diagnostics", [])
            return len(schedule) == len(actual) and all(
                (p["filter_current_tariff"], p["filter_arpu_segment"], p["target_tariff"], p["channel"], p["n_customers"])
                == (a["current"], a["segment"], a["target"], a["channel"], a["n_customers"])
                for p, a in zip(schedule, actual))
        audit["per_scenario"][scenario] = dict(fixed_runs=len(fixed), unique_schedule_hashes=len(hashes),
            all_fixed_actions_match_schedule=all(matches(r) for r in fixed),
            planned_pilots=len(schedules[0]) if schedules and schedules[0] else 0,
            planned_strata=sorted({p["filter_arpu_segment"] for p in (schedules[0] or [])}),
            planned_current_tariffs=sorted({p["filter_current_tariff"] for p in (schedules[0] or [])}),
            same_model_each_pair=all(p["same_model"] for p in summary[scenario]["pairs"]))
        all_pairs.extend(dict(scenario=scenario, **p) for p in summary[scenario]["pairs"])
    audit["all_valid"] = all(r["status"] == "ok" for r in rows)
    audit["no_drops_or_caps"] = all(r.get("dropped_campaigns") == r.get("capped_campaigns") == 0 for r in rows)
    audit["fixed_schedule_invariance"] = all(s["unique_schedule_hashes"] == 1 and s["all_fixed_actions_match_schedule"] for s in audit["per_scenario"].values())
    audit["expected_attempts_present"] = len(rows) == 140 and len({(r["policy"], r["scenario"], r["seed"]) for r in rows}) == 140

    def dominates(first, second):
        strict = False
        for scenario, result in summary.items():
            a, b = result["policies"][first], result["policies"][second]
            if a["statuses"]["ok"] != a["attempts"]:
                return False
            fields = ["median_net", "q10_net", "valid_positive"]
            if scenario.startswith("rare_"):
                fields += ["positive_discovery", "positive_deployment"]
            for field in fields:
                if a[field] is None or b[field] is None or a[field] < b[field]:
                    return False
                strict |= a[field] > b[field]
        return strict

    fixed_dominates, adaptive_dominates = dominates("fixed", "adaptive"), dominates("adaptive", "fixed")
    decision = "рассмотреть кандидата" if fixed_dominates else "сохранить текущую" if adaptive_dominates else "данных недостаточно для общего превосходства; результаты смешанные"
    audit["decision_gates"] = dict(fixed_dominates=fixed_dominates, adaptive_dominates=adaptive_dominates, decision=decision)
    (folder / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, data in (("pairs.csv", all_pairs), ("policy_summary.csv", [dict(scenario=scenario, policy=policy, **{k: v for k, v in stats.items() if k != "statuses"}, **stats["statuses"])
                        for scenario, result in summary.items() for policy, stats in result["policies"].items()])):
        with (folder / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    metrics = ["policy", "scenario", "seed", "status", "net", "pilot_net", "pilot_cost", "pilot_contacts", "n_pilots", "total_cost", "total_contacts", "preflight_valid", "dropped_campaigns", "capped_campaigns", "positive_discovered", "positive_deployed", "positive_observation", "agent_seconds", "wall_seconds", "exception"]
    with (folder / "attempts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in metrics} for r in rows)

    lines = ["# Польза адаптивной разведки: FixedSurveyAgent против Agent", "",
        f"**Решение: {decision}.** Production не меняется. Один фиксированный соперник; перенастройки после открытия результатов не было.", "",
        f"База origin/main: `{frozen['base_sha']}`. Исходники и протокол до новой серии: `{report['git']['sha']}`.",
        f"SHA256 кандидата (UTF-8/LF): `{frozen['sha256_lf_utf8']['experiments/adaptivity_ablation/fixed_agent.py']}`.", "",
        "## Что сравниваем", "",
        "FixedSurveyAgent строит весь план разведки до первого реального пилота. Он использует исходные priors, публичную ценность/размер аудитории и то же правило ценности информации, что Agent. Планировочные наблюдения равны prior mean; они уменьшают неопределённость только в отдельной копии beliefs. Первые шесть действий покрывают HIGH/MID/LOW и разные текущие тарифы. Повторы распределяются тем же скорингом; последние пять слотов предпочитают подтверждение перспективных по priors предложений. При равенстве выбирается первый вариант в исходном детерминированном порядке. Реальные ответы не меняют расписание, но обновляют чистые реальные beliefs для финала.", "",
        "Сохранены история, публичные фильтры, размеры 100/200 с уменьшением по ресурсам, риск-поправка 1.65, генератор вариантов, финальный планировщик и валидатор. Полный блок финала побайтово совпадает с базой. Изменение включает зависимость выбора, повторов и остановки разведки от наблюдений; отдельные компоненты адаптации здесь не разделены. Лимиты одинаковы, фактические расходы могут различаться.", "",
        "Точное правило и заранее принятые критерии: [protocol.md](../experiments/adaptivity_ablation/protocol.md). Копия получена проверяемым преобразованием [build_candidate.py](../experiments/adaptivity_ablation/build_candidate.py). Никаких моделей эффектов, сценариев или приватных данных среды в коде кандидата нет.", "",
        "## Контроль базы и объём", "",
        "Seed 0–4 воспроизведены: net (допуск 1e-6), общие расходы/контакты, число пилотов и preflight совпали с сохранённым benchmark; стоимость/контакты пилотов — с прежними диагностическими журналами. Четыре проверки механизма прошли. Начальный тест крайне малого бюджета ошибочно ожидал допустимый финал при 80 контактах и неделимых ячейках по 240; оба агента явно отказывают. Исправлено ожидание теста, политика не менялась.", "",
        f"Новая серия: **{len(rows)} попыток, 70 пар**, seed 50–59 на mock и заранее выбранных mixed_1/2, negative_1/2, rare_1/2. Все семь моделей уже существовали в pilot_research. Ошибок, timeout, invalid_plan, отброшенных и обрезанных кампаний: **{sum(r['status'] != 'ok' for r in rows)}**. Версии: Python {report['versions']['python']}, NumPy {report['versions']['numpy']}, pandas {report['versions']['pandas']}.", "",
        "Все 10 расписаний FixedSurveyAgent внутри каждой модели идентичны; выполненные действия совпадают с расписанием. Хеши scorer, моделей, данных и обоих алгоритмов после серии совпали с замороженными. Подробности: [audit.json](../reports/adaptivity_ablation/audit.json).", "",
        "## Net по моделям", "",
        "Условные единицы; A — адаптивный Agent, F — FixedSurveyAgent. Медиана разностей не равна разности медиан. Доля плюсов учитывает все попытки, включая потенциальные сбои.", "",
        "| Модель | Медиана A | Медиана F | q10 A | q10 F | Минимум A | Минимум F | Корректных плюсов A/F |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scenario in report["scenarios"]:
        a, f = (summary[scenario]["policies"][p] for p in ("adaptive", "fixed"))
        lines.append(f"| {scenario} | {number(a['median_net'])} | {number(f['median_net'])} | {number(a['q10_net'])} | {number(f['q10_net'])} | {number(a['min_net'])} | {number(f['min_net'])} | {a['valid_positive']}/10 · {f['valid_positive']}/10 |")
    lines += ["", "## Попарные разницы A − F", "", "Положительное значение означает преимущество адаптации. Минимум — худшее ухудшение A относительно F; максимум — худшее ухудшение F относительно A. Все отдельные пары: [pairs.csv](../reports/adaptivity_ablation/pairs.csv).", "", "| Модель | Медиана Δ | q10 Δ | Минимум Δ | Максимум Δ | Побед A/F/ничьих |", "|---|---:|---:|---:|---:|---:|"]
    for scenario in report["scenarios"]:
        p = summary[scenario]["paired"]
        lines.append(f"| {scenario} | {number(p['median_adaptive_minus_fixed'])} | {number(p['q10_adaptive_minus_fixed'])} | {number(p['min_adaptive_minus_fixed'])} | {number(p['max_adaptive_minus_fixed'])} | {p['adaptive_wins']}/{p['fixed_wins']}/{p['ties']} |")
    lines += ["", "## Обнаружение редкого положительного сочетания", "", "Обнаружение означает хотя бы один пилот истинно положительного сочетания; положительный зашумлённый ответ учитывается отдельно. Выбор в финале не означает положительный общий net. Истина доступна только оценщику.", "", "| Модель | Проверили хорошее A/F | Получили положительный ответ на хорошем A/F | Выбрали в финал A/F |", "|---|---:|---:|---:|"]
    for scenario in ("rare_1", "rare_2"):
        a, f = (summary[scenario]["policies"][p] for p in ("adaptive", "fixed"))
        lines.append(f"| {scenario} | {a['positive_discovery']}/10 · {f['positive_discovery']}/10 | {a['positive_observation']}/10 · {f['positive_observation']}/10 | {a['positive_deployment']}/10 · {f['positive_deployment']}/10 |")
    lines += ["", "## Ресурсы и время", "", "Средние на один запуск. Контакты включают повторы; это не число уникальных людей. Время — выполнение act, включая составление расписания F, без скоринга и запуска процесса. Полное wall time также сохранено в CSV/JSONL.", "", "| Модель | Агент | Пилотов | Контакты пилотов | Стоимость пилотов | Все контакты | Общий расход | act, сек |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in report["scenarios"]:
        for policy in ("adaptive", "fixed"):
            s = summary[scenario]["policies"][policy]
            lines.append(f"| {scenario} | {policy} | {number(s['mean_n_pilots'])} | {number(s['mean_pilot_contacts'])} | {number(s['mean_pilot_cost'])} | {number(s['mean_total_contacts'])} | {number(s['mean_total_cost'])} | {number(s['mean_agent_seconds'])} |")
    lines += ["", "## Интерпретация и решение", "",
        "- **mixed_2:** адаптация выигрывает все 10 пар; медиана попарного выигрыша 397 005.76 у.е., даже худшая пара даёт +147 215.01. Положительных запусков 10/10 против 2/10. Здесь польза адаптивной разведки подтверждена на данной синтетической модели при одинаковом финальном планировщике.",
        "- **negative_1:** адаптация выигрывает 9/10 пар, медиана попарного выигрыша 127 203.07. Оба агента остаются отрицательными во всех запусках. **negative_2:** 8/10 побед и +37 747.74 по медиане разностей, но q10 адаптивного агента хуже: −517 635.77 против −503 476.65.",
        "- **mixed_1:** медианы net почти равны (905 425.62 против 905 860.59), адаптация выигрывает 6/10 пар и заметно улучшает q10, однако её минимум хуже: 120 005.89 против 222 143.28. Улучшение одного показателя нижнего хвоста не означает улучшения всех.",
        "- **mock:** более высокая отдельная медиана Agent не означает типичного попарного выигрыша: медиана A−F равна −13 313.51, побед 5/5. Худшее ухудшение адаптации — **−436 043.47 на seed 58**. Минимум Agent 685 689.43 против 1 027 410.56 у FixedSurveyAgent.",
        "- **rare_1:** фиксированное расписание выигрывает 10/10 пар и имеет меньшие потери; медиана A−F = −50 426.56. **rare_2:** 5/5 побед, медиана разностей всего +873.75, но q10 и минимум адаптации хуже. В обеих rare-моделях оба агента нашли и выбрали хороший вариант **0/10** раз. Преимущество discovery не показано; способность находить редкое предложение здесь не подтверждена ни у одного.",
        "Ни один агент не прошёл заранее установленный критерий превосходства по всем моделям. **Данных недостаточно для общего вывода о превосходстве; результаты смешанные. Сохраняем текущий production, FixedSurveyAgent остаётся исследовательским кандидатом.** Оснований менять production автоматически или заявлять победу на скрытой модели нет. Повторного подбора политики по этим seed не проводилось.", "",
        "## Ограничения", "",
        "Это семь известных синтетических моделей с новыми seed шума, не скрытое судейство и не реальные эффекты Beeline. Равный seed при разных действиях не даёт одинаковые выборки клиентов. Всего десять повторов на модель: q10 и минимум нестабильны. Денежные результаты разных моделей не складываются в общую прибыль. Одна заранее определённая fixed-политика не представляет все возможные неадаптивные методы. Public mock связан с выданной историей; независимость от истории здесь не проверяется.", "",
        "## Воспроизводимость и файлы", "", "```sh",
        "python -m pytest -q experiments/adaptivity_ablation/test_ablation.py",
        "python experiments/adaptivity_ablation/measure.py --mode reproduce --out reports/adaptivity_ablation/reproduction_rerun.json",
        "python experiments/adaptivity_ablation/measure.py --mode compare --out reports/adaptivity_ablation/comparison_rerun.json", "```", "",
        "Команды выполняются из корня репозитория в окружении с requirements-dev.txt. Новое имя выхода обязательно: старые попытки не перезаписываются. Скрипт report.py оформляет именно сохранённую исходную comparison.json; для повторной серии сводка уже находится в её JSON.", "",
        "- [comparison.json](../reports/adaptivity_ablation/comparison.json): сводки, пары, SHA и версии.",
        "- [comparison.jsonl](../reports/adaptivity_ablation/comparison.jsonl): все попытки, полные traces, ответы пилотов, final, preflight, ошибки и время.",
        "- [attempts.csv](../reports/adaptivity_ablation/attempts.csv): компактные метрики всех попыток.",
        "- [policy_summary.csv](../reports/adaptivity_ablation/policy_summary.csv): сводки по моделям и политикам.",
        "- [reproduction.json](../reports/adaptivity_ablation/reproduction.json): контроль seed 0–4; полный журнал рядом.",
        "- [frozen.json](../experiments/adaptivity_ablation/frozen.json): контрольные суммы политики, исходников моделей, scorer и входных данных.", ""]
    (ROOT / "docs/adaptivity_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
