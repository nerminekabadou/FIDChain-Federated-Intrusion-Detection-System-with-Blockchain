"""Off-chain benchmark of FIDChain's aggregation strategies.

The blockchain and IPFS only transport the weights; they do not change the
learning. This script reproduces the federated rounds in memory with the
project's own model (utils_manager.create_model), hospital split
(utils_simulation.createHospitals) and aggregation functions, and compares:

  * centralized training on all data (upper bound, no privacy)
  * each hospital training alone (no collaboration)
  * federated: plain average, FedAvg (weighted by dataset size),
    similarity-weighted ("single" and "multiple" factors)

Every setup sees the same total number of local epochs.

Usage (from the project root, after `python scripts/load_dataset.py`):
    python scripts/offchain_benchmark.py [--rounds 5] [--epochs 10]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from utils_manager import (
    HEIGHT, WIDTH, DEPTH, NUM_CLASSES, compile_info, create_model,
    similarity_factor_multiple, similarity_factor_single,
)
from utils_simulation import HOSPITAL_SPLIT, RANDOM_SEED, createHospitals, set_reproducibility

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def new_model():
    model = create_model((HEIGHT, WIDTH, DEPTH), NUM_CLASSES)
    model.compile(**compile_info)
    return model


def evaluate(model, X, y_onehot):
    pred = np.argmax(model.predict(X, verbose=0), axis=1)
    true = np.argmax(y_onehot, axis=1)
    return {"accuracy": round(float(accuracy_score(true, pred)), 4),
            "macro_f1": round(float(f1_score(true, pred, average="macro")), 4)}


def aggregate(strategy, weights, sizes):
    names = list(weights)
    n_layers = len(weights[names[0]])
    average = [sum(weights[h][i] for h in names) / len(names) for i in range(n_layers)]
    if strategy == "average":
        return average
    if strategy == "fedavg":
        total = sum(sizes.values())
        return [sum(weights[h][i] * sizes[h] / total for h in names) for i in range(n_layers)]
    if strategy == "similarity_single":
        factors = {h: similarity_factor_single(h, weights, average, names) for h in names}
        return [sum(factors[h] * weights[h][i] for h in names) for i in range(n_layers)]
    if strategy == "similarity_multiple":
        factors = {h: similarity_factor_multiple(h, weights, average, names) for h in names}
        return [sum(factors[h][i] * weights[h][i] for h in names) for i in range(n_layers)]
    raise ValueError(strategy)


def run_federated(strategy, hospitals, X_test, y_test, rounds, epochs):
    set_reproducibility(RANDOM_SEED)
    global_weights = new_model().get_weights()
    models = {h: new_model() for h in hospitals}
    sizes = {h: len(hospitals[h].dataset["X_train"]) for h in hospitals}
    evaluator = new_model()
    history = []
    for r in range(rounds):
        local = {}
        for h, model in models.items():
            model.set_weights(global_weights)
            d = hospitals[h].dataset
            model.fit(d["X_train"], d["y_train"], epochs=epochs, verbose=0)
            local[h] = model.get_weights()
        global_weights = [np.array(w, dtype=np.float32) for w in aggregate(strategy, local, sizes)]
        evaluator.set_weights(global_weights)
        history.append(evaluate(evaluator, X_test, y_test))
        print(f"  {strategy} round {r + 1}: {history[-1]}", flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)

    set_reproducibility(RANDOM_SEED)
    hospitals = createHospitals()
    X_test = np.concatenate([hospitals[h].dataset["X_test"] for h in HOSPITAL_SPLIT])
    y_test = np.concatenate([hospitals[h].dataset["y_test"] for h in HOSPITAL_SPLIT])
    total_epochs = args.rounds * args.epochs
    results = {"setup": {
        "rounds": args.rounds, "local_epochs_per_round": args.epochs,
        "train_images": {h: int(len(hospitals[h].dataset["X_train"])) for h in HOSPITAL_SPLIT},
        "test_images": int(len(X_test)),
    }}

    t = time.time()
    set_reproducibility(RANDOM_SEED)
    central = new_model()
    central.fit(np.concatenate([hospitals[h].dataset["X_train"] for h in HOSPITAL_SPLIT]),
                np.concatenate([hospitals[h].dataset["y_train"] for h in HOSPITAL_SPLIT]),
                epochs=total_epochs, verbose=0)
    results["centralized"] = {**evaluate(central, X_test, y_test), "seconds": round(time.time() - t)}
    print("centralized", results["centralized"], flush=True)

    results["local_only"] = {}
    for h in HOSPITAL_SPLIT:
        set_reproducibility(RANDOM_SEED)
        model = new_model()
        d = hospitals[h].dataset
        model.fit(d["X_train"], d["y_train"], epochs=total_epochs, verbose=0)
        results["local_only"][h] = evaluate(model, X_test, y_test)
        print("local", h, results["local_only"][h], flush=True)

    results["federated"] = {}
    for strategy in ["average", "fedavg", "similarity_single", "similarity_multiple"]:
        t = time.time()
        history = run_federated(strategy, hospitals, X_test, y_test, args.rounds, args.epochs)
        results["federated"][strategy] = {"per_round": history, "final": history[-1],
                                          "seconds": round(time.time() - t)}

    with open(os.path.join(OUT, "fl_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    labels = {"average": "Plain average", "fedavg": "FedAvg (size-weighted)",
              "similarity_single": "Similarity-weighted (single)",
              "similarity_multiple": "Similarity-weighted (per layer)"}
    plt.figure(figsize=(6.4, 4))
    x = range(1, args.rounds + 1)
    for s, label in labels.items():
        plt.plot(x, [r["accuracy"] for r in results["federated"][s]["per_round"]], marker="o", label=label)
    plt.axhline(results["centralized"]["accuracy"], color="black", ls="--", lw=1, label="Centralized (upper bound)")
    best_local = max(v["accuracy"] for v in results["local_only"].values())
    plt.axhline(best_local, color="grey", ls=":", lw=1, label="Best single hospital alone")
    plt.xlabel("Federated round")
    plt.ylabel("Test accuracy (all hospitals)")
    plt.xticks(list(x))
    plt.legend(fontsize=8)
    plt.title("FIDChain aggregation strategies - Alzheimer MRI (4 classes)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fl_accuracy.png"), dpi=150)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
