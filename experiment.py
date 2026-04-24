"""
Basic MNIST experiment using a classic three-layer MLP.

Architecture:
- Input layer: 784 features (28x28 flattened image)
- Hidden layer: configurable width with Sigmoid activation
- Output layer: 10 logits (digit classes)

Training:
- Loss: Mean Squared Error (MSE)
- Optimizer: Stochastic Gradient Descent (SGD)
"""

import argparse
import csv
import os

import numpy as np

from DataLoader import DataLoader
from Linear import Linear
from MSELoss import MSELoss
from SGD import SGD
from Sequential import Sequential
from Sigmoid import Sigmoid


def one_hot(labels, num_classes=10):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded
# end function


def _detect_layout(header, first_row, expect_labels):
    header_l = [h.strip().lower() for h in header]

    if expect_labels:
        if "label" in header_l:
            label_idx = header_l.index("label")
            pixel_indices = [
                i for i in range(len(header)) if header_l[i] != "label" and header_l[i] != "id"
            ]
            return label_idx, pixel_indices
        # end if

        if len(first_row) == 786:
            return 1, list(range(2, 786))
        # end if

        if len(first_row) == 785:
            return 0, list(range(1, 785))
        # end if
    else:
        if len(first_row) == 785:
            return None, list(range(1, 785))
        # end if

        if len(first_row) == 784:
            return None, list(range(0, 784))
        # end if
    # end if

    raise ValueError("Could not infer CSV column layout from header/row shape.")
# end function


def load_images_labels(csv_path, expect_labels):
    images = []
    labels = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)
        label_idx, pixel_indices = _detect_layout(header, first_row, expect_labels)

        rows = [first_row]
        rows.extend(list(reader))

        for row in rows:
            if expect_labels and label_idx is not None:
                labels.append(int(row[label_idx]))
            # end if

            pixels = [float(row[idx]) / 255.0 for idx in pixel_indices]
            images.append(pixels)
        # end for
    # end with

    x = np.asarray(images, dtype=np.float32)
    y = None if not expect_labels else np.asarray(labels, dtype=int)
    return x, y
# end function


def load_test_labels(solution_csv):
    labels = []
    with open(solution_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_l = [h.strip().lower() for h in header]

        if "label" in header_l:
            label_idx = header_l.index("label")
        else:
            label_idx = 1 if len(header) > 1 else 0
        # end if

        for row in reader:
            labels.append(int(row[label_idx]))
        # end for
    # end with

    return np.asarray(labels, dtype=int)
# end function


class ArrayDataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # end method

    def __len__(self):
        return self.x.shape[0]
    # end method

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    # end method



def build_model(hidden_size, seed=42):
    np.random.seed(seed)
    return Sequential([
        Linear(in_features=784, out_features=hidden_size),
        Sigmoid(),
        Linear(in_features=hidden_size, out_features=10),
    ])
# end function


def evaluate(model, dataloader, loss_fn):
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in dataloader:
        out = model.forward(x)
        loss = loss_fn.forward(out, y)

        total_loss += loss * x.shape[0]
        total_correct += int(np.sum(np.argmax(out, axis=1) == np.argmax(y, axis=1)))
        total_count += x.shape[0]
    # end for

    return total_loss / total_count, total_correct / total_count
# end function


def parse_args():
    parser = argparse.ArgumentParser(description="Basic 3-layer MLP experiment on MNIST")
    parser.add_argument(
        "--mnist-dir",
        type=str,
        default=os.path.join("dataset", "mnist"),
        help="Directory containing train.csv, test.csv, and test-solutions.csv",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
# end function


def main():
    args = parse_args()

    train_csv = os.path.join(args.mnist_dir, "train.csv")
    test_csv = os.path.join(args.mnist_dir, "test.csv")
    test_sol_csv = os.path.join(args.mnist_dir, "test-solutions.csv")

    print("Loading MNIST data...")
    train_x_raw, train_labels = load_images_labels(train_csv, expect_labels=True)
    test_x_raw, _ = load_images_labels(test_csv, expect_labels=False)
    test_labels = load_test_labels(test_sol_csv)

    train_y = one_hot(train_labels, num_classes=10)
    test_y = one_hot(test_labels, num_classes=10)

    train_loader = DataLoader(
        ArrayDataset(train_x_raw, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ArrayDataset(test_x_raw, test_y),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(hidden_size=args.hidden_size, seed=args.seed)
    loss_fn = MSELoss()
    optimizer = SGD(modules=[model], lr=args.lr)

    print(
        f"\nTraining 3-layer MLP: 784 -> {args.hidden_size} -> 10 | "
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}"
    )

    for epoch in range(args.epochs):
        for x, y in train_loader:
            out = model.forward(x)
            _ = loss_fn.forward(out, y)

            optimizer.zero_grad()
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()
        # end for

        train_loss, train_acc = evaluate(model, train_loader, loss_fn)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
        )
    # end for

    print("\nDone.")
# end function


if __name__ == "__main__":
    main()
