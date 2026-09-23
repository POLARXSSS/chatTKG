# Conservative reranker

This directory implements the train-only experiment described in
`protocol.md`. It does not alter `mycode`, `tlogic_temporal_validity`, or
`hierarchical_temporal_ranker`.

Install the experiment dependencies with:

```powershell
python -m pip install -r conservative_reranker/requirements.txt
```

The CAMEO manual is intentionally not vendored. Download the source identified
by `SOURCE_URL` in `external_cameo/run_study.py` and save it as
`external_cameo/CAMEO.Manual.1.1b3.tex`, or pass its path with `--source`.

The pipeline first creates an early `train.txt` prefix and mines TLogic rules
only from that prefix. The builder then generates TLogic and direction-level
history candidates for later pseudo-future training queries. Candidate
truncation is score-based and never keeps a candidate because it is the answer.
The trainer fits query-gated bounded residual models for four equal-capacity
feature views and evaluates them once on the final internal time block.

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
