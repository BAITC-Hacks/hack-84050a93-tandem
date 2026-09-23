# Исследование распределения пилотов

База: `c65003c5ec8811b829b583bc98e9fe79954b6538`. Реализация изолирована
в `experiments/pilot_research/`; основной агент, его стратегия, данные,
валидатор и официальная среда не изменены.

## Два варианта

- `experiments.pilot_research.agent:Agent` (sized): первый пилот каждой
  гипотезы ограничен 40 клиентами и четвертью ячейки, минимум 10;
  повторные измерения сохраняют исходный выбор 100/200.
- `experiments.pilot_research.agent:StopAgent` (stop): исходные размеры
  пилотов; после минимум шести шагов и покрытия всех доступных ARPU-страт
  разведка прекращается, если не менее 80% исследованных гипотез имеют
  отрицательное posterior mean, нет уверенно положительной гипотезы
  (`mean > 1.65 * std`) и взвешенное среднее плюс 1.65 стандартной ошибки
  отрицательно. Это эвристика; она не доказывает отсутствие хороших
  неисследованных предложений и предполагает возможность объединять
  наблюдения из разных ячеек.

Размеры, приоры, финальное ранжирование и портфель в stop совпадают с базой.
Скопированный алгоритм использует исторический CSV из корня репозитория.
Агент получает только публичный env; истинные эффекты доступны только harness.

## Зафиксированный результат разработки

Seed 0–9: исходный benchmark воспроизведён по net, расходам, контактам,
числу кампаний, пилотов и preflight. Проверка —
`reports/pilot_research/baseline_check.json`.

SIZED снижает медиану mock с 1 434 332 до 1 088 793 (примерно −24%).
STOP сохраняет результаты mock и смешанных моделей и уменьшает потери,
но в исходном rare_good находит хороший сегмент в 0/10 вместо 8/10 запусков.
Отрицательный итог базы здесь не означает, что сегмент не был найден.

В `reports/pilot_research/selection.json` до проверки новых seed зафиксирован
STOP и SHA-256 его исходника. Оба варианта и все результаты сохраняются.
Новые seed 30–49 проверяют шум тех же моделей. В каждой семье дополнительных
синтетических сценариев меняются расположение и величины положительных эффектов.
Одинаковый seed не гарантирует одинаковые выборки при разных действиях агента.

## Воспроизведение

После установки существующих `requirements-dev.txt` из корня репозитория:

```bash
python experiments/pilot_research/run_suite.py --agent agent:Agent --seeds 0:10 --prefix rerun_dev_baseline
python experiments/pilot_research/run_suite.py --agent experiments.pilot_research.agent:Agent --seeds 0:10 --prefix rerun_dev_sized
python experiments/pilot_research/run_suite.py --agent experiments.pilot_research.agent:StopAgent --seeds 0:10 --prefix rerun_dev_stop
python experiments/pilot_research/run_suite.py --agent agent:Agent --seeds 30:50 --prefix rerun_holdout_baseline
python experiments/pilot_research/run_suite.py --agent experiments.pilot_research.agent:StopAgent --seeds 30:50 --prefix rerun_holdout_stop
python -m pytest -q tests experiments/pilot_research/test_research.py
```

Runner вызывает неизменённые `tools/benchmark.py` и `tools/stress_benchmark.py`
через `--agent`, затем дополнительный harness. Префикс должен быть новым:
результаты предыдущих попыток не перезаписываются. Все выходы находятся
только в `reports/pilot_research/`.
