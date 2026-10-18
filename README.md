# AuditFL

Implementation of audit-constrained federated aggregation and dependency-aware counterfactual recovery.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

`prepare` downloads [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) with torchvision and makes the seeded client partition. `config.json` holds seed 11, client/training/retention settings. Create `invalid.json` as a JSON list of computation IDs such as `["client:0:0"]` before training; `[]` selects no injected invalid updates. Reuse the same config/data/output paths for the workflow.

```bash
python -m auditfl prepare --config config.json --data data
```

## Run and evaluate

```bash
python -m auditfl train --config config.json --data data --output outputs --invalid invalid.json
python -m auditfl recover --config config.json --data data --output outputs --invalid invalid.json --deadline 60
```

The example uses oracle audit mode. `--audit-mode frozen` reuses frozen audit decisions.

## Outputs

The output folder holds round checkpoints, model/manifest, `normal.json`, `audit.json` and recovery state. Recovery writes `recovery.json` and, where applicable, `fidelity.png`.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
