import argparse
import copy
import json
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def json_ready(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


class CNN(nn.Module):

    def __init__(self, channels, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


def initial(cfg):
    torch.manual_seed(cfg["seed"])
    model = CNN(cfg["channels"], cfg["classes"])
    return {
        "model": copy.deepcopy(model.state_dict()),
        "server": {k: torch.zeros_like(v) for k, v in model.state_dict().items()},
        "clients": {},
        "round": 0,
    }


def local_update(state, client, round_id, data, cfg, omit_momentum=False):
    model = CNN(cfg["channels"], cfg["classes"])
    model.load_state_dict(state["model"])
    lr = cfg["learning_rate"] * 0.5 * (1 + math.cos(math.pi * round_id / cfg["rounds"]))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    key = str(client)
    if key in state["clients"] and (not omit_momentum):
        optimizer.load_state_dict(copy.deepcopy(state["clients"][key]))
        for group in optimizer.param_groups:
            group["lr"] = lr
    indices = np.flatnonzero((data["client"] == client) & (data["split"] == "train"))
    if not len(indices):
        raise ValueError(f"Client {client} has no training data")
    generator = torch.Generator().manual_seed(cfg["seed"] + 100003 * round_id + client)
    steps = 0
    for _ in range(cfg["local_epochs"]):
        order = torch.randperm(len(indices), generator=generator).numpy()
        for batch in np.array_split(
            order, max(1, math.ceil(len(order) / cfg["batch_size"]))
        ):
            ix = indices[batch]
            x = torch.tensor(data["x"][ix], dtype=torch.float32)
            y = torch.tensor(data["y"][ix], dtype=torch.long)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            steps += 1
    delta = {
        k: v.detach().clone() - state["model"][k] for k, v in model.state_dict().items()
    }
    state["clients"][key] = copy.deepcopy(optimizer.state_dict())
    return (delta, len(indices), steps)


def advance(
    state,
    round_id,
    participants,
    invalid,
    data,
    cfg,
    tier="exact",
    historical=None,
    inject=False,
):
    updates = []
    weights = []
    records = []
    steps = 0
    for client in participants:
        node = f"client:{round_id}:{client}"
        if node in invalid and (not inject):
            continue
        if tier == "edited":
            delta = historical[node]["delta"]
            weight = historical[node]["weight"]
            local_steps = 0
        else:
            delta, weight, local_steps = local_update(
                state, client, round_id, data, cfg, tier == "no-momentum"
            )
            if inject and node in invalid or node in cfg.get("fault_nodes", []):
                delta = {k: v + cfg["injection_magnitude"] for k, v in delta.items()}
        updates.append(delta)
        weights.append(weight)
        steps += local_steps
        records.append(
            {
                "id": node,
                "parent": f"model:{round_id}",
                "local_parent": f"client-state:{client}:{round_id}",
                "weight": weight,
                "delta": delta,
            }
        )
    if not updates:
        raise ValueError(f"No admitted participant remains at round {round_id}")
    total = sum(weights)
    for key in state["model"]:
        mean = sum((w * u[key] for w, u in zip(weights, updates))) / total
        state["server"][key] = cfg["server_momentum"] * state["server"][key] + mean
        state["model"][key] += state["server"][key]
    state["round"] = round_id + 1
    return (records, steps)


def utility(state, data, cfg):
    model = CNN(cfg["channels"], cfg["classes"])
    model.load_state_dict(state["model"])
    model.eval()
    ids = np.flatnonzero(data["split"] == "test")
    if not len(ids):
        return (None, [])
    predictions = []
    with torch.no_grad():
        for batch in np.array_split(
            ids, max(1, math.ceil(len(ids) / cfg["batch_size"]))
        ):
            predictions.extend(
                model(torch.tensor(data["x"][batch], dtype=torch.float32))
                .argmax(1)
                .tolist()
            )
    return (float(np.mean(np.array(predictions) == data["y"][ids])), predictions)


def audit_loss(parameters, data, cfg):
    ids = np.flatnonzero(data["split"] == "validation")
    if not len(ids):
        raise ValueError("Frozen audit requires held-out development examples")
    model = CNN(cfg["channels"], cfg["classes"])
    model.load_state_dict(parameters)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in np.array_split(
            ids, max(1, math.ceil(len(ids) / cfg["batch_size"]))
        ):
            total += F.cross_entropy(
                model(torch.tensor(data["x"][batch], dtype=torch.float32)),
                torch.tensor(data["y"][batch], dtype=torch.long),
                reduction="sum",
            ).item()
    return total / len(ids)


def tensor_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all((tensor_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all((tensor_equal(x, y) for x, y in zip(a, b)))
    return a == b


def prepare(args, cfg):
    from torchvision.datasets import CIFAR10

    rng = np.random.default_rng(cfg["seed"])
    arrays = []
    for training in [True, False]:
        source = CIFAR10(args.data, train=training, download=True)
        x = np.asarray(source.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255
        y = np.asarray(source.targets)
        client = np.full(len(y), -1)
        for label in range(cfg["classes"]):
            ids = np.flatnonzero(y == label)
            rng.shuffle(ids)
            counts = rng.multinomial(
                len(ids), rng.dirichlet(np.full(cfg["clients"], cfg["dirichlet_alpha"]))
            )
            for c, group in enumerate(np.split(ids, np.cumsum(counts)[:-1])):
                client[group] = c
        split = np.full(len(y), "train" if training else "test", dtype="<U10")
        if training:
            for c in range(cfg["clients"]):
                ids = np.flatnonzero(client == c)
                rng.shuffle(ids)
                split[ids[: max(1, int(0.1 * len(ids)))]] = "validation"
        arrays.append((x, y, client, split))
    np.savez_compressed(
        Path(args.data) / "federated.npz",
        **{
            name: np.concatenate([r[i] for r in arrays])
            for i, name in enumerate(["x", "y", "client", "split"])
        },
    )


def train(args, cfg, data):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    state = initial(cfg)
    rng = np.random.default_rng(cfg["seed"])
    invalid = set(json.loads(Path(args.invalid).read_text())) if args.invalid else set()
    schedule = [
        rng.choice(cfg["clients"], cfg["participants"], replace=False).tolist()
        for _ in range(cfg["rounds"])
    ]
    scheduled = {
        f"client:{t}:{client}"
        for t, clients in enumerate(schedule)
        for client in clients
    }
    if not invalid.issubset(scheduled):
        raise ValueError(
            "Invalid scope includes a nonexistent or unscheduled client computation"
        )
    history = {}
    manifest = [{"id": "model:0", "parents": [], "kind": "initial-state"}]
    local_steps = 0
    audit_records = []
    previous_flags = {}
    detected = set()
    audit_seconds = 0.0
    client_parent = {str(c): f"client-state:{c}:initial" for c in range(cfg["clients"])}
    manifest.extend(
        (
            {"id": value, "parents": [], "kind": "initial-client-state"}
            for value in client_parent.values()
        )
    )
    for t, participants in enumerate(schedule):
        if (
            t % cfg["checkpoint_gap"] == 0
            and t >= cfg["rounds"] - cfg["retention"] - cfg["checkpoint_gap"]
        ):
            torch.save(copy.deepcopy(state), out / f"checkpoint_{t}.pt")
        parent = copy.deepcopy(state["model"])
        records, steps = advance(
            state, t, participants, invalid, data, cfg, inject=True
        )
        local_steps += steps
        audit_start = time.perf_counter()
        base_loss = audit_loss(parent, data, cfg)
        norms = np.array(
            [
                float(torch.sqrt(sum((v.square().sum() for v in r["delta"].values()))))
                for r in records
            ]
        )
        median = np.median(norms)
        cutoff = median + 3 * np.median(np.abs(norms - median))
        for record, norm in zip(records, norms):
            client = record["id"].split(":")[2]
            candidate = {key: parent[key] + record["delta"][key] for key in parent}
            increase = audit_loss(candidate, data, cfg) - base_loss
            flag = bool(norm > cutoff and increase > 0.1)
            if (
                flag
                and previous_flags.get(client, (-2, False))[0] == t - 1
                and previous_flags[client][1]
            ):
                detected.update([record["id"], f"client:{t - 1}:{client}"])
            previous_flags[client] = (t, flag)
            audit_records.append(
                {
                    "node": record["id"],
                    "norm": float(norm),
                    "loss_increase": increase,
                    "flag": flag,
                }
            )
        audit_seconds += time.perf_counter() - audit_start
        for record in records:
            client = record["id"].split(":")[2]
            record["local_parent"] = client_parent[client]
            manifest.append(
                {
                    "id": record["id"],
                    "parents": [record["parent"], record["local_parent"]],
                    "kind": "client-computation",
                    "weight": record["weight"],
                    "data_client": client,
                    "random_seed": cfg["seed"] + 100003 * t + int(client),
                }
            )
            client_parent[client] = f"client-state:{client}:{t + 1}"
            manifest.append(
                {
                    "id": client_parent[client],
                    "parents": [record["id"]],
                    "kind": "client-state",
                }
            )
            if t >= cfg["rounds"] - cfg["retention"]:
                history[record["id"]] = record
        manifest.append(
            {
                "id": f"model:{t + 1}",
                "parents": [f"model:{t}"] + [r["id"] for r in records],
                "kind": "aggregation",
                "round": t,
            }
        )
    torch.save(
        {
            "state": state,
            "history": history,
            "schedule": schedule,
            "invalid": sorted(invalid),
            "detected": sorted(detected),
            "config": cfg,
        },
        out / "run.pt",
    )
    (out / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, allow_nan=False)
    )
    (out / "normal.json").write_text(
        json.dumps(
            json_ready(
                {
                    "client_steps": local_steps,
                    "accuracy": utility(state, data, cfg)[0],
                    "checkpoint_bytes": sum(
                        (f.stat().st_size for f in out.glob("checkpoint_*.pt"))
                    ),
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    (out / "audit.json").write_text(
        json.dumps(
            json_ready(
                {
                    "records": audit_records,
                    "scope": sorted(detected),
                    "true_positive_nodes": len(detected & invalid),
                    "missed_invalid_nodes": len(invalid - detected),
                    "false_positive_nodes": len(detected - invalid),
                    "audit_processing_s": audit_seconds,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )


def recover(args, cfg, data):
    root = Path(args.output)
    run = torch.load(root / "run.pt", map_location="cpu", weights_only=False)
    cfg = run["config"]
    cfg["fault_nodes"] = run["invalid"]
    invalid = (
        set(json.loads(Path(args.invalid).read_text()))
        if args.invalid
        else set(run["detected"] if args.audit_mode == "frozen" else run["invalid"])
    )
    if not invalid:
        (root / "recovery.json").write_text(
            json.dumps(
                json_ready(
                    {
                        "supported": False,
                        "reason": "No invalid computation identified by the selected audit",
                        "audit_mode": args.audit_mode,
                        "request_count": 1,
                        "timely_success": 0,
                    }
                ),
                indent=2,
                allow_nan=False,
            )
        )
        return
    scheduled = {
        f"client:{t}:{client}"
        for t, clients in enumerate(run["schedule"])
        for client in clients
    }
    if not invalid.issubset(scheduled):
        raise ValueError("Invalid scope includes a nonexistent or unscheduled node")
    manifest = json.loads((root / "manifest.json").read_text())
    children = {}
    for node in manifest:
        for parent in node["parents"]:
            children.setdefault(parent, []).append(node["id"])
    closure = set(invalid)
    queue = list(invalid)
    while queue:
        node = queue.pop()
        for child in children.get(node, []):
            if child not in closure:
                closure.add(child)
                queue.append(child)
    earliest = min((int(s.split(":")[1]) for s in invalid))
    checkpoint = earliest // cfg["checkpoint_gap"] * cfg["checkpoint_gap"]
    path = root / f"checkpoint_{checkpoint}.pt"
    rounds = cfg["rounds"] - checkpoint
    d = sum((v.numel() * v.element_size() for v in initial(cfg)["model"].values()))
    network = rounds * cfg["participants"] * (2 * d + cfg["metadata_bytes"])
    admitted = (
        path.exists()
        and rounds <= cfg["retention"]
        and (network <= args.byte_budget)
        and (rounds * cfg["participants"] <= args.work_budget)
    )
    required = {
        c
        for t in range(checkpoint, cfg["rounds"])
        for c in run["schedule"][t]
        if f"client:{t}:{c}" not in invalid
    }
    unavailable = sorted(set(args.unavailable) & required)
    if unavailable:
        admitted = False
    if not admitted:
        (root / "recovery.json").write_text(
            json.dumps(
                json_ready(
                    {
                        "supported": False,
                        "checkpoint_available": path.exists(),
                        "required_rounds": rounds,
                        "required_network_bytes": network,
                        "unavailable_clients": args.unavailable,
                        "request_count": 1,
                        "timely_success": 0,
                    }
                ),
                indent=2,
                allow_nan=False,
            )
        )
        return
    start = time.perf_counter()
    reference = initial(cfg)
    for t, participants in enumerate(run["schedule"]):
        advance(reference, t, participants, invalid, data, cfg)
    target_params = torch.cat([v.flatten() for v in reference["model"].values()])
    _, target_predictions = utility(reference, data, cfg)
    rows = []
    for tier in ["exact", "edited", "no-momentum", "no-recovery", "full-retraining"]:
        timer = time.perf_counter()
        state = torch.load(path, map_location="cpu", weights_only=False)
        steps = 0
        if tier == "no-recovery":
            state = copy.deepcopy(run["state"])
        elif tier == "full-retraining":
            state = initial(cfg)
            for t, participants in enumerate(run["schedule"]):
                _, count = advance(state, t, participants, invalid, data, cfg)
                steps += count
        else:
            for t in range(checkpoint, cfg["rounds"]):
                _, count = advance(
                    state,
                    t,
                    run["schedule"][t],
                    invalid,
                    data,
                    cfg,
                    tier,
                    run["history"],
                )
                steps += count
        elapsed = time.perf_counter() - timer
        params = torch.cat([v.flatten() for v in state["model"].values()])
        accuracy, pred = utility(state, data, cfg)
        rows.append(
            {
                "tier": tier,
                "parameter_deviation": float(
                    torch.linalg.vector_norm(params - target_params)
                    / max(float(torch.linalg.vector_norm(target_params)), 1e-12)
                ),
                "complete_state_equal": tensor_equal(state, reference),
                "prediction_disagreement": (
                    float(np.mean(np.array(pred) != target_predictions))
                    if pred
                    else None
                ),
                "accuracy": accuracy,
                "client_steps": steps,
                "recovery_s": elapsed,
                "network_bytes": (
                    0
                    if tier in ["edited", "no-recovery"]
                    else (cfg["rounds"] if tier == "full-retraining" else rounds)
                    * cfg["participants"]
                    * (2 * d + cfg["metadata_bytes"])
                ),
                "timely_success": int(elapsed <= args.deadline),
            }
        )
    result = {
        "supported": True,
        "request_count": 1,
        "affected_nodes": len(closure),
        "rows": rows,
        "reference_and_recovery_total_s": time.perf_counter() - start,
        "normal_network_bytes_per_round": cfg["participants"]
        * (2 * d + cfg["metadata_bytes"]),
        "declared_retained_bytes": (d + cfg["clients"] * d)
        * (math.ceil(cfg["retention"] / cfg["checkpoint_gap"]) + 1)
        + cfg["retention"] * cfg["participants"] * (d + cfg["metadata_bytes"]),
    }
    (root / "recovery.json").write_text(
        json.dumps(json_ready(result), indent=2, allow_nan=False)
    )
    plt.bar([r["tier"] for r in rows], [r["parameter_deviation"] for r in rows])
    plt.ylabel("Normalized parameter deviation")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(root / "fidelity.png")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "train", "recover"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="outputs")
    p.add_argument("--invalid")
    p.add_argument("--audit-mode", choices=["oracle", "frozen"], default="oracle")
    p.add_argument("--unavailable", nargs="*", type=int, default=[])
    p.add_argument("--byte-budget", type=float, default=float("inf"))
    p.add_argument("--work-budget", type=float, default=float("inf"))
    p.add_argument("--deadline", type=float, default=60)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "prepare":
        prepare(args, cfg)
        return
    data = np.load(Path(args.data) / "federated.npz")
    if args.command == "train":
        train(args, cfg, data)
    else:
        recover(args, cfg, data)


if __name__ == "__main__":
    main()
