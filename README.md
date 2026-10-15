# AuditFL

Implementation of audit-constrained federated aggregation and dependency-aware counterfactual recovery.

Install Python 3.10+ and run `python -m pip install -e .`.

```bash
python -m auditfl prepare
python -m auditfl train --invalid invalid.json
python -m auditfl recover --invalid invalid.json --deadline 60
```

`prepare` downloads [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), builds Dirichlet client partitions and writes `data/federated.npz`. The NPZ contains float NCHW `x`, integer `y`, integer `client`, and `split` (`train`, `validation`, `test`). For FEMNIST, export authorized LEAF writer examples to this schema, preserving writers as clients; set channels/classes accordingly. Supply a JSON list such as `["client:185:0"]` as `invalid.json`; the node must actually participate in that round. Training injects a declared additive update corruption only at the supplied nodes. Recovery excludes those contributions and renormalizes the remaining sample weights.

The two-convolution classifier uses persistent client SGD momentum, a cosine schedule and optional server momentum. Full checkpoints retain model, server transition state, every participating client's optimizer state and round. Coupled minibatch permutations depend on seed/round/client; recomputation uses the corrected parent. The dependency manifest records the sequential model chain and client-state dependencies. Under this synchronous protocol every later round is in the affected closure.

The recovery command compares corrected re-execution, edited stored-update replay, omitted-momentum re-execution, no recovery and full counterfactual training. It uses an independently recomputed reference with the same initial state, schedule, random order and removal semantics. Unsupported requests return `supported:false` when checkpoint retention, byte/work budget or client availability prevents the complete plan. Use `--unavailable 0` to represent unavailable historical clients; availability does not silently trigger another tier.

`config.json` selects rounds, client participation, checkpoint gap, retention, local optimizer, heterogeneity and corruption size. `--work-budget` limits client-round recomputations; `--byte-budget` limits point-to-point model/update/metadata traffic. `--deadline` is measured against each executed recovery; reference generation is reported separately. Retention applies to historical checkpoint and update artifacts; `run.pt` is the retained run bundle and is not a deployable model download. Normal storage reports actual checkpoint bytes separately from the paper's accounting formula, whose equal-size model/client-state convention is retained in `declared_retained_bytes`.

Outputs include `manifest.json`, normal accounting, `recovery.json`, and `fidelity.png`. Correctness metrics compare complete state and normalized parameters; prediction disagreement and test accuracy describe utility. Client steps, communication and elapsed time describe different resources. Failures remain recovery requests with zero timely success. These commands execute optimization only when invoked; no result values are distributed.

## Scope of the audit

`--invalid` supplies the audit scope, separating reconstruction from detection. The training run also records the frozen median-plus-three-MAD norm and development-loss-rise rule with two-round persistence. Use `python -m auditfl recover --audit-mode frozen` for its detected scope; `audit.json` retains missed nodes, false positives and processing time. FedProx, SCAFFOLD, robust aggregation and calibrated historical reuse have different state/transition contracts and are not mislabeled as the included FedAvg controls. A nonexistent or unscheduled invalid node is rejected before reconstruction. Recovery tiers and missing-state controls share the same declared removal reference.
