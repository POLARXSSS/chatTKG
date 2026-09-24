# ICEWS18 candidate-cap sensitivity results

This note records the train-only pseudo-future P0 experiment with seed 13. No
validation or test facts were read. All four treatments use the same temporal
split, rules, model, optimizer, seed, and CPU training runtime. A complete
`512/256/768` pool was built once; exact source ranks and float64 priorities
were then used to derive the smaller pools without recomputing candidates.

## Confirmation results

The cap notation is `base/external/union`.

| Caps | Candidate rows | Union coverage | learned_all MRR | learned_official MRR |
|---|---:|---:|---:|---:|
| 256/128/384 | 3,528,427 | 0.795566 | **0.372147** | 0.370102 |
| 512/128/640 | 5,876,763 | 0.810564 | 0.370622 | **0.371824** |
| 256/256/512 | 3,964,165 | 0.800783 | 0.371438 | 0.371541 |
| 512/256/768 | 6,206,554 | **0.815455** | 0.367646 | 0.370606 |

Increasing the base cap from 256 to 512 raised TLogic coverage by 2.02
percentage points but changed TLogic MRR by only +0.000057; Hits@1, Hits@3,
and Hits@10 were unchanged. The newly recovered answers were therefore almost
entirely in the low-ranked tail.

Paired moving-block comparisons against `256/128/384` found:

- `512/128/640`, learned_base: delta MRR +0.001812, 95% CI
  [0.000191, 0.003408].
- `512/128/640`, learned_official: +0.001722, CI
  [0.000733, 0.002700].
- `512/128/640`, learned_all: -0.001525, CI
  [-0.004148, 0.001107].
- `256/256/512`, learned_official: +0.001440, CI
  [-0.000586, 0.003680].
- `512/256/768`, learned_all: -0.004501, CI
  [-0.010203, 0.001419].

The recommended recall-aware configuration for the next ranking experiment is
`512/128/640`, with `256/128/384` retained as the efficiency control. Increasing
the external cap to 256 or increasing both sources is not supported.

## Attribution limits

At `512/128/640`, learned_official minus learned_all was +0.001202 MRR with CI
[-0.000995, 0.003617], and learned_official minus learned_random was +0.000719
with CI [-0.002360, 0.003401]. Thus the run supports limited base-tail recovery,
not a unique benefit from official CAMEO semantics. Every configuration also
failed the predeclared 99% base-top-1 preservation gate.

The next priority is ranking-objective and conservative-gate work, rather than
another increase in candidate limits.

## Reproduction

```bash
python conservative_reranker/run_cap_sensitivity.py \
  --python-executable /usr/bin/python \
  --dataset icews18 \
  --rules output_external_cameo/conservative_reranker_icews18_s13/prefix_rules.json \
  --output-dir output_external_cameo/cap_sensitivity_icews18_s13 \
  --queries-per-relation 64 \
  --num-processes 48 \
  --epochs 40 \
  --patience 6 \
  --device cpu \
  --cpu-affinity 24-71 \
  --nice-level 10 \
  --seed 13 \
  --rebuild
```

Generated `.npz`, `.pt`, full report, dataset, and rule artifacts are excluded
from version control.
