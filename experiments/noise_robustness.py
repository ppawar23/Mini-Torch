"""
Noise-robustness experiment for the Mini-Torch 3-layer MLP.

Trains a 784 -> hidden -> 10 sigmoid MLP on MNIST or Fashion-MNIST, then
evaluates it on copies of the test set perturbed by additive Gaussian noise
at a sweep of sigma values. Outputs:

    results/<dataset>/training_history.csv  (epoch, train_loss, train_acc,
                                             test_loss, test_acc)
    results/<dataset>/noise_sweep.csv       (sigma, seed, accuracy)
    results/<dataset>/weights.npz           (final model parameters)
    results/<dataset>/confusion_sigma_<s>.npy  (confusion at sigmas in
                                                CONFUSION_SIGMAS)
    results/<dataset>/sample_images.npz     (clean test images for the
                                             qualitative figure)

The trained model lives in memory only between train and eval; the weights are
also serialised so the notebook can load them without retraining.

Usage:
    python -m experiments.noise_robustness --dataset mnist
    python -m experiments.noise_robustness --dataset fashion --epochs 20
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MLP import (
    ArrayDataset,
    build_model,
    evaluate,
    load_images_labels,
    load_test_labels,
    one_hot,
)
from DataLoader import DataLoader
from MSELoss import MSELoss
from SGD import SGD


DEFAULT_SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
CONFUSION_SIGMAS = [0.0, 0.3]
DATASET_DIRS = {
    "mnist": os.path.join("dataset", "mnist"),
    "fashion": os.path.join("dataset", "fashion-mnist"),
}


def add_gaussian_noise(x, sigma, rng):
    """Add N(0, sigma^2) noise to inputs already normalised to [0, 1]."""
    if sigma <= 0.0:
        return x
    noisy = x + rng.normal(0.0, sigma, x.shape).astype(x.dtype)
    return np.clip(noisy, 0.0, 1.0, out=noisy)


def evaluate_on_noisy(model, x, y_onehot, sigma, seed, batch_size=256):
    """Evaluate top-1 accuracy on a noised copy of (x, y_onehot)."""
    rng = np.random.default_rng(seed)
    x_noisy = add_gaussian_noise(x, sigma, rng)
    correct = 0
    total = x.shape[0]
    for start in range(0, total, batch_size):
        out = model.forward(x_noisy[start : start + batch_size])
        pred = np.argmax(out, axis=1)
        true = np.argmax(y_onehot[start : start + batch_size], axis=1)
        correct += int(np.sum(pred == true))
    return correct / total


def confusion_matrix(model, x, labels, sigma, seed, num_classes=10, batch_size=256):
    rng = np.random.default_rng(seed)
    x_noisy = add_gaussian_noise(x, sigma, rng)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for start in range(0, x.shape[0], batch_size):
        out = model.forward(x_noisy[start : start + batch_size])
        pred = np.argmax(out, axis=1)
        true = labels[start : start + batch_size]
        for t, p in zip(true, pred):
            cm[t, p] += 1
    return cm


def parse_sigmas(spec):
    if not spec:
        return DEFAULT_SIGMAS
    return [float(s) for s in spec.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description="Noise-robustness sweep for Mini-Torch MLP")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override the default dataset directory.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmas", type=str, default="",
                        help="Comma-separated sigma values; default is the full sweep.")
    parser.add_argument("--eval-seeds", type=int, default=5,
                        help="Number of noise RNG seeds per sigma (for error bars).")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    data_dir = args.data_dir or DATASET_DIRS[args.dataset]
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    test_sol_csv = os.path.join(data_dir, "test-solutions.csv")

    print(f"[{args.dataset}] loading from {data_dir}")
    train_x, train_labels = load_images_labels(train_csv, expect_labels=True)
    test_x, _ = load_images_labels(test_csv, expect_labels=False)
    test_labels = load_test_labels(test_sol_csv)

    train_y = one_hot(train_labels, num_classes=10)
    test_y = one_hot(test_labels, num_classes=10)

    train_loader = DataLoader(ArrayDataset(train_x, train_y),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(ArrayDataset(test_x, test_y),
                             batch_size=args.batch_size, shuffle=False)

    np.random.seed(args.seed)
    model = build_model(hidden_size=args.hidden_size, seed=args.seed)
    loss_fn = MSELoss()
    optimizer = SGD(modules=[model], lr=args.lr)

    print(f"[{args.dataset}] training 784 -> {args.hidden_size} -> 10 for "
          f"{args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")

    history_path = os.path.join(out_dir, "training_history.csv")
    t_start = time.time()
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                         "elapsed_seconds"])

        for epoch in range(args.epochs):
            for x_batch, y_batch in train_loader:
                out = model.forward(x_batch)
                _ = loss_fn.forward(out, y_batch)
                optimizer.zero_grad()
                grad = loss_fn.backward()
                model.backward(grad)
                optimizer.step()

            train_loss, train_acc = evaluate(model, train_loader, loss_fn)
            test_loss, test_acc = evaluate(model, test_loader, loss_fn)
            elapsed = time.time() - t_start
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                             f"{test_loss:.6f}", f"{test_acc:.6f}", f"{elapsed:.2f}"])
            print(f"  epoch {epoch + 1:2d}/{args.epochs} | "
                  f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | "
                  f"elapsed={elapsed:.1f}s")

    weights = {}
    for i, layer in enumerate(model.modules):
        for j, p in enumerate(layer.parameters()):
            weights[f"layer{i}_param{j}"] = p
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)

    print(f"[{args.dataset}] noise sweep over sigmas={sigmas} with "
          f"{args.eval_seeds} seeds each")
    sweep_path = os.path.join(out_dir, "noise_sweep.csv")
    with open(sweep_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "seed", "accuracy"])
        for sigma in sigmas:
            row_accs = []
            for s in range(args.eval_seeds):
                eval_seed = 1000 + s
                acc = evaluate_on_noisy(model, test_x, test_y, sigma, eval_seed)
                writer.writerow([sigma, eval_seed, f"{acc:.6f}"])
                row_accs.append(acc)
            mean = float(np.mean(row_accs))
            std = float(np.std(row_accs))
            print(f"  sigma={sigma:.2f}  acc={mean:.4f} +/- {std:.4f}")

    for sigma in CONFUSION_SIGMAS:
        cm = confusion_matrix(model, test_x, test_labels, sigma, seed=2000)
        cm_path = os.path.join(out_dir, f"confusion_sigma_{sigma:.2f}.npy")
        np.save(cm_path, cm)

    sample_idx = np.arange(min(8, test_x.shape[0]))
    np.savez(os.path.join(out_dir, "sample_images.npz"),
             images=test_x[sample_idx], labels=test_labels[sample_idx])

    print(f"[{args.dataset}] done. outputs in {out_dir}")


if __name__ == "__main__":
    main()
