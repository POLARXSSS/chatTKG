# Conservative candidate reranker protocol

Status: exploratory train-only mechanism experiment. The implementation must not
read validation or test facts.

## Question and hypothesis

The primary question is whether a query-conditioned residual reranker can use
history candidates without destroying correct TLogic ordering. The primary
estimand is the paired MRR difference on the final, chronologically held-out
part of the pseudo-future queries constructed from `train.txt`.

The mechanism hypothesis is that the learned official-CAMEO reranker improves
over original TLogic and over a fixed residual. The semantic hypothesis is
stronger: the official grouping must also improve over both an equal-capacity
all-history model and an equal-capacity size-preserving random grouping.

## Leakage boundary

- Source facts: `train.txt` only, including the synthetic inverse views created
  by `Grapher`.
- TLogic rule structures and confidence/support values are mined only from the
  early prefix with timestamps below the pseudo-future cutoff. Full-train rules
  are rejected unless an explicit smoke-only override is supplied.
- Every query at time `t` uses only facts with timestamp `< t`.
- Facts at the query timestamp are inserted only after every query at that
  timestamp has been featurized.
- The selected pseudo-future timestamps are divided chronologically into model
  training (60%), checkpoint selection (20%), and internal confirmation (20%).
- ICEWS validation and test files are not accepted by the builder or trainer.

This internal confirmation is still exploratory because the research design
was developed using earlier ICEWS14 validation diagnostics. It is not a final
generalization test.

## Candidate set and treatments

For each query the union is built without looking at its answer:

`TLogic rule candidates UNION direction-level subject history`.

The following treatments are evaluated on the same confirmation queries:

1. `tlogic`: original TLogic candidates and scores only.
2. `fixed_all`: fixed 0.95 TLogic + 0.05 all-history score.
3. `fixed_official`: the existing frozen fixed blend with official CAMEO roots.
4. `fixed_random`: the same fixed blend with a size-preserving random grouping.
5. `learned_base`: learned conservative residual using TLogic features only.
6. `learned_all`: equal-capacity residual using TLogic and all-history features.
7. `learned_official`: equal-capacity residual with official-root history.
8. `learned_random`: equal-capacity residual with randomized-root history.

The learned score is

`final_score = tlogic_score + sigmoid(query_gate) * max_residual * tanh(delta)`.

Training uses query-level pairwise ranking loss. A distillation penalty is
applied only when the original TLogic top-1 is correct, and a gate penalty
encourages the default behavior to preserve the base ordering.

## Outcomes

Primary: confirmation MRR and paired delta versus `tlogic`.

Secondary: Hits@1/3/10, answer coverage, MRR conditional on coverage, original
correct-top-1 preservation rate, harmed original-top-1 count, rescued query
count, direction-specific delta, and a 7-timestamp moving-block bootstrap
interval.

Candidate truncation is score-based and label-blind. The report records how
many positive queries are lost by truncation. Queries without a positive
candidate receive entity-count rank and are retained in aggregate metrics.

## Predeclared decision gate

Proceed to a fresh external validation only if all are true on the internal
confirmation block:

- `learned_official - tlogic >= 0.001` MRR;
- the 95% 7-timestamp moving-block interval has a positive lower endpoint;
- both forward and inverse mean deltas are positive;
- learned official exceeds learned all-history and learned random grouping;
- original correct-top-1 preservation is at least 99%.

Failure of the semantic conditions does not invalidate reranking as an
engineering result, but it prevents a knowledge-driven contribution claim.
