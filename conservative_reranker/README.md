# Conservative reranker

This directory implements the train-only experiment described in
`protocol.md`. It does not alter `mycode`, `tlogic_temporal_validity`, or
`hierarchical_temporal_ranker`.

The pipeline first creates an early `train.txt` prefix and mines TLogic rules
only from that prefix. The builder then generates TLogic and direction-level
history candidates for later pseudo-future training queries. Candidate
truncation is score-based and never keeps a candidate because it is the answer.
The trainer fits query-gated bounded residual models for four equal-capacity
feature views and evaluates them once on the final internal time block.
Both rule mining and candidate construction use a dedicated train-only graph
loader that never opens `valid.txt` or `test.txt`. The runner limits BLAS/OpenMP
threads inside multiprocessing stages to avoid CPU oversubscription and shows
an overall four-stage progress bar plus query/worker progress.

On a shared Linux host, constrain only this experiment with `--cpu-affinity`
and `--nice-level`. `--gpu-index` exposes one physical GPU to the tiny training
stage; with `--device auto`, the runner falls back to CPU if that GPU exceeds
`--max-gpu-utilization`. The chosen resources are recorded in
`run_resources.json`. Candidate progress is reported every 16 completed
queries across workers, rather than only when a whole worker chunk finishes.

## P0 candidate-cap sensitivity

`run_cap_sensitivity.py` builds the complete `512/256/768` candidate pool once,
then derives three smaller configurations offline and trains all four treatments:

- `baseline_256_128_384`
- `base_512_128_640`
- `external_256_256_512`
- `combined_512_256_768`

The large archive stores float64 source priorities and exact per-source ranks.
Consequently, offline cap slicing is exactly equivalent to rebuilding the same
smaller cap directly; it is not an approximate filter over float32 model
features. The final comparison is written to `cap_sensitivity_summary.json`.

On the current 96-physical-core shared host, the polite 48-worker command is:

```bash
cd /path/to/chatTKG
python conservative_reranker/run_cap_sensitivity.py \
  --dataset icews18 \
  --rules output_external_cameo/conservative_reranker_icews18_s13/prefix_rules.json \
  --output-dir output_external_cameo/cap_sensitivity_icews18_s13 \
  --queries-per-relation 64 \
  --num-processes 48 \
  --epochs 40 \
  --patience 6 \
  --device cpu \
  --python-executable /usr/bin/python \
  --cpu-affinity 24-71 \
  --nice-level 10 \
  --seed 13 \
  --rebuild
```

This uses 48 physical cores (twice the prior run), avoids the GPUs already used
by other jobs, and gives normal-priority processes scheduling precedence.

## Smoke test

```powershell
python -B -m unittest conservative_reranker/test_pipeline.py

python conservative_reranker/run_experiment.py `
  --dataset icews14 `
  --output-dir output_external_cameo/conservative_reranker_prefix_smoke_s13 `
  --num-walks 5 `
  --rule-processes 2 `
  --max-queries 80 `
  --queries-per-relation 4 `
  --num-processes 1 `
  --epochs 3 `
  --patience 2 `
  --device cpu `
  --seed 13
```

The smoke run validates the code path only. It is not evidence of predictive
benefit.

## Full exploratory run

```powershell
python conservative_reranker/run_experiment.py `
  --dataset icews14 `
  --output-dir output_external_cameo/conservative_reranker_icews14_s13 `
  --num-walks 200 `
  --rule-processes 8 `
  --queries-per-relation 64 `
  --num-processes 8 `
  --epochs 40 `
  --patience 6 `
  --device auto `
  --seed 13
```

Outputs include the immutable feature archive and metadata, one checkpoint per
learned treatment, and `report.json` with selection/confirmation metrics,
paired time-block uncertainty, preservation/rescue counts, and the predeclared
progression gate.

Supplying `--rules` is supported only for a rule file with the matching
`.meta.json` produced by `mine_prefix_rules.py`. The
`--allow-unverified-rules` switch exists solely for code-path smoke tests; such
runs are labeled invalid for progression and can never pass the gate.
