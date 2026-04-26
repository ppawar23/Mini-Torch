"""
Homework 2: Multi-Layer Perceptron Experiments
Runs four investigations on both MNIST and Fashion MNIST:
  1. Hidden layer size vs performance
  2. Batch size effects + early stopping
  3. Noise robustness
  4. Multiple hidden layers

Usage:
  python3 run_experiments.py

Place this in your Mini-Torch root directory.
Expects data in:
  dataset/mnist/train.csv, test.csv, test-solutions.csv
  dataset/fashion/train.csv, test.csv, test-solutions.csv
"""

import os
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Linear import Linear
from Sigmoid import Sigmoid
from Sequential import Sequential
from MSELoss import MSELoss
from SGD import SGD
from DataLoader import DataLoader


# ====================================================================
# Data Loading
# ====================================================================

def one_hot(labels, num_classes=10):
    encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def load_dataset(data_dir):
    print(f"  Loading from {data_dir}...")
    t0 = time.time()

    images, labels = [], []
    with open(os.path.join(data_dir, 'train.csv'), 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        label_idx = header.index('label')
        pixel_indices = [i for i, h in enumerate(header) if h not in ('id', 'label')]
        for row in reader:
            labels.append(int(row[label_idx]))
            images.append([float(row[i]) / 255.0 for i in pixel_indices])
    train_x = np.array(images, dtype=np.float32)
    train_y = one_hot(np.array(labels, dtype=int))

    images2 = []
    with open(os.path.join(data_dir, 'test.csv'), 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        pixel_indices = [i for i, h in enumerate(header) if h not in ('id', 'label')]
        for row in reader:
            images2.append([float(row[i]) / 255.0 for i in pixel_indices])
    test_x = np.array(images2, dtype=np.float32)

    test_labels = []
    with open(os.path.join(data_dir, 'test-solutions.csv'), 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        li = header.index('label') if 'label' in header else 1
        for row in reader:
            test_labels.append(int(row[li]))
    test_labels = np.array(test_labels[:len(test_x)], dtype=int)
    test_y = one_hot(test_labels)

    print(f"    Train: {train_x.shape[0]}, Test: {test_x.shape[0]} ({time.time()-t0:.0f}s)")
    return train_x, train_y, test_x, test_y


# ====================================================================
# Helpers
# ====================================================================

class ArrayDataset:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_model(layer_sizes, seed=42):
    np.random.seed(seed)
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:  # no Sigmoid on last layer
            layers.append(Sigmoid())
    return Sequential(layers)

def train_one_epoch(model, loader, loss_fn, optimizer):
    total_loss, count = 0.0, 0
    for x, y in loader:
        out = model.forward(x)
        loss = loss_fn.forward(out, y)
        optimizer.zero_grad()
        grad = loss_fn.backward()
        model.backward(grad)
        optimizer.step()
        total_loss += loss * x.shape[0]
        count += x.shape[0]
    return total_loss / count


def evaluate(model, loader, loss_fn):
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        out = model.forward(x)
        loss = loss_fn.forward(out, y)
        total_loss += loss * x.shape[0]
        correct += int(np.sum(np.argmax(out, axis=1) == np.argmax(y, axis=1)))
        count += x.shape[0]
    return total_loss / count, correct / count


def compute_confusion(model, loader):
    cm = np.zeros((10, 10), dtype=int)
    for x, y in loader:
        out = model.forward(x)
        pred = np.argmax(out, axis=1)
        actual = np.argmax(y, axis=1)
        for p, a in zip(pred, actual):
            cm[a, p] += 1
    return cm


def plot_confusion(cm, title, filepath):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=7)
    plt.colorbar(ax.images[0])
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# ====================================================================
# Experiment 1: Hidden Layer Size vs Performance
# ====================================================================

def exp1_hidden_size(train_x, train_y, test_x, test_y, name, out_dir):
    print(f"\n{'='*60}\nEXP 1: Hidden Layer Size ({name})\n{'='*60}")

    sizes = [32, 64, 128, 256]
    epochs, lr, bs = 10, 0.1, 64
    results = {}

    for hs in sizes:
        print(f"\n  Hidden size = {hs}")
        model = build_model([784, hs, 10])
        loss_fn = MSELoss()
        opt = SGD([model], lr=lr)
        trl = DataLoader(ArrayDataset(train_x, train_y), batch_size=bs, shuffle=True)
        tel = DataLoader(ArrayDataset(test_x, test_y), batch_size=bs, shuffle=False)
        history = []
        t0 = time.time()
        for ep in range(epochs):
            tl = train_one_epoch(model, trl, loss_fn, opt)
            _, ta = evaluate(model, trl, loss_fn)
            vl, va = evaluate(model, tel, loss_fn)
            history.append({'ep': ep+1, 'tl': tl, 'ta': ta, 'vl': vl, 'va': va})
            print(f"    Epoch {ep+1}/{epochs} | train={ta:.4f} | test={va:.4f}")
        results[hs] = {'history': history, 'time': time.time()-t0}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for hs in sizes:
        axes[0].plot(range(1, epochs+1), [h['va']*100 for h in results[hs]['history']],
                     '-o', label=f'H={hs}', markersize=3)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title(f'{name}: Test Accuracy vs Hidden Layer Size')
    axes[0].legend(); axes[0].grid(True)

    axes[1].bar([str(s) for s in sizes],
                [results[s]['history'][-1]['va']*100 for s in sizes], color='steelblue')
    ax2 = axes[1].twinx()
    ax2.plot([str(s) for s in sizes], [results[s]['time'] for s in sizes], 'r-s')
    ax2.set_ylabel('Time (s)', color='red')
    axes[1].set_xlabel('Hidden Size'); axes[1].set_ylabel('Final Acc (%)')
    axes[1].set_title(f'{name}: Final Accuracy and Training Time')
    axes[1].grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'exp1_{name}.png'), dpi=150)
    plt.close()
    return results


# ====================================================================
# Experiment 2: Batch Size + Early Stopping
# ====================================================================

def exp2_batch_size(train_x, train_y, test_x, test_y, name, out_dir):
    print(f"\n{'='*60}\nEXP 2: Batch Size + Early Stopping ({name})\n{'='*60}")

    batch_sizes = [10, 32, 64, 128]
    max_epochs, patience, lr, hs = 20, 3, 0.1, 128
    results = {}

    for bs in batch_sizes:
        print(f"\n  Batch size = {bs}")
        model = build_model([784, hs, 10])
        loss_fn = MSELoss()
        opt = SGD([model], lr=lr)
        trl = DataLoader(ArrayDataset(train_x, train_y), batch_size=bs, shuffle=True)
        tel = DataLoader(ArrayDataset(test_x, test_y), batch_size=bs, shuffle=False)
        history = []
        best_vl, wait = float('inf'), 0
        t0 = time.time()

        for ep in range(max_epochs):
            et = time.time()
            tl = train_one_epoch(model, trl, loss_fn, opt)
            epoch_time = time.time() - et
            _, ta = evaluate(model, trl, loss_fn)
            vl, va = evaluate(model, tel, loss_fn)
            history.append({'ep': ep+1, 'ta': ta, 'va': va, 'vl': vl, 'et': epoch_time})
            print(f"    Epoch {ep+1} | test={va:.4f} | val_loss={vl:.5f} | {epoch_time:.1f}s")
            if vl < best_vl - 1e-5:
                best_vl = vl; wait = 0
            else:
                wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {ep+1}")
                break

        results[bs] = {'history': history, 'time': time.time()-t0}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for bs in batch_sizes:
        axes[0].plot(range(1, len(results[bs]['history'])+1),
                     [h['va']*100 for h in results[bs]['history']],
                     '-o', label=f'BS={bs}', markersize=3)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title(f'{name}: Accuracy by Batch Size'); axes[0].legend(); axes[0].grid(True)

    axes[1].bar([str(bs) for bs in batch_sizes],
                [np.mean([h['et'] for h in results[bs]['history']]) for bs in batch_sizes],
                color='coral')
    axes[1].set_xlabel('Batch Size'); axes[1].set_ylabel('Avg Epoch Time (s)')
    axes[1].set_title(f'{name}: Training Speed'); axes[1].grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'exp2_{name}.png'), dpi=150)
    plt.close()
    return results


# ====================================================================
# Experiment 3: Noise Robustness
# ====================================================================

def exp3_noise(train_x, train_y, test_x, test_y, name, out_dir):
    print(f"\n{'='*60}\nEXP 3: Noise Robustness ({name})\n{'='*60}")

    model = build_model([784, 128, 10])
    loss_fn = MSELoss()
    opt = SGD([model], lr=0.1)
    trl = DataLoader(ArrayDataset(train_x, train_y), batch_size=64, shuffle=True)
    print("  Training base model...")
    for ep in range(10):
        train_one_epoch(model, trl, loss_fn, opt)
    tel = DataLoader(ArrayDataset(test_x, test_y), batch_size=64, shuffle=False)
    _, base_acc = evaluate(model, tel, loss_fn)
    print(f"  Base accuracy: {base_acc*100:.2f}%")

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = {}
    for sigma in noise_levels:
        np.random.seed(99)
        noisy_x = np.clip(test_x + np.random.normal(0, sigma, test_x.shape).astype(np.float32), 0, 1)
        nl = DataLoader(ArrayDataset(noisy_x, test_y), batch_size=64, shuffle=False)
        _, acc = evaluate(model, nl, loss_fn)
        results[sigma] = acc
        print(f"  sigma={sigma:.2f} | acc={acc*100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(noise_levels, [results[s]*100 for s in noise_levels], 'b-o')
    axes[0].set_xlabel('Noise (sigma)'); axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'{name}: Accuracy vs Noise'); axes[0].grid(True)
    axes[1].bar([str(s) for s in noise_levels], [results[s]*100 for s in noise_levels], color='steelblue')
    axes[1].set_xlabel('Noise Level'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{name}: Degradation'); axes[1].grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'exp3_{name}.png'), dpi=150)
    plt.close()

    fig2, ax2 = plt.subplots(1, 4, figsize=(10, 3))
    for i, sigma in enumerate([0.0, 0.1, 0.3, 0.5]):
        np.random.seed(99)
        noisy = np.clip(test_x[0] + np.random.normal(0, sigma, 784).astype(np.float32), 0, 1)
        ax2[i].imshow(noisy.reshape(28, 28), cmap='gray')
        ax2[i].set_title(f'sigma={sigma}'); ax2[i].axis('off')
    plt.suptitle(f'{name}: Noise Samples')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'exp3_samples_{name}.png'), dpi=150)
    plt.close()
    return results


# ====================================================================
# Experiment 4: Multiple Hidden Layers
# ====================================================================

def exp4_depth(train_x, train_y, test_x, test_y, name, out_dir):
    print(f"\n{'='*60}\nEXP 4: Multiple Hidden Layers ({name})\n{'='*60}")

    configs = {
        '1-layer (784-128-10)': [784, 128, 10],
        '2-layer (784-128-64-10)': [784, 128, 64, 10],
        '3-layer (784-256-128-64-10)': [784, 256, 128, 64, 10],
    }
    epochs, lr, bs = 10, 0.1, 64
    results = {}

    for cfg_name, arch in configs.items():
        print(f"\n  {cfg_name}")
        model = build_model(arch)
        loss_fn = MSELoss()
        opt = SGD([model], lr=lr)
        trl = DataLoader(ArrayDataset(train_x, train_y), batch_size=bs, shuffle=True)
        tel = DataLoader(ArrayDataset(test_x, test_y), batch_size=bs, shuffle=False)
        history = []
        t0 = time.time()
        for ep in range(epochs):
            tl = train_one_epoch(model, trl, loss_fn, opt)
            _, ta = evaluate(model, trl, loss_fn)
            vl, va = evaluate(model, tel, loss_fn)
            history.append({'ep': ep+1, 'tl': tl, 'ta': ta, 'vl': vl, 'va': va})
            print(f"    Epoch {ep+1}/{epochs} | train={ta:.4f} | test={va:.4f}")
        cm = compute_confusion(model, tel)
        results[cfg_name] = {'history': history, 'time': time.time()-t0, 'cm': cm}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cfg_name in configs:
        axes[0].plot(range(1, epochs+1), [h['va']*100 for h in results[cfg_name]['history']],
                     '-o', label=cfg_name, markersize=3)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title(f'{name}: Depth Comparison'); axes[0].legend(fontsize=8); axes[0].grid(True)
    for cfg_name in configs:
        axes[1].plot(range(1, epochs+1), [h['tl'] for h in results[cfg_name]['history']],
                     '-o', label=cfg_name, markersize=3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Training Loss')
    axes[1].set_title(f'{name}: Loss by Depth'); axes[1].legend(fontsize=8); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'exp4_{name}.png'), dpi=150)
    plt.close()

    best = max(results, key=lambda n: results[n]['history'][-1]['va'])
    plot_confusion(results[best]['cm'], f'{name}: Confusion ({best})',
                   os.path.join(out_dir, f'exp4_cm_{name}.png'))
    return results


# ====================================================================
# Main
# ====================================================================

def main():
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)

    for dname, ddir in [('MNIST', 'dataset/mnist'), ('FashionMNIST', 'dataset/fashion')]:
        print(f"\n{'#'*60}\n  DATASET: {dname}\n{'#'*60}")
        train_x, train_y, test_x, test_y = load_dataset(ddir)
        exp1_hidden_size(train_x, train_y, test_x, test_y, dname, out_dir)
        exp2_batch_size(train_x, train_y, test_x, test_y, dname, out_dir)
        exp3_noise(train_x, train_y, test_x, test_y, dname, out_dir)
        exp4_depth(train_x, train_y, test_x, test_y, dname, out_dir)

    print(f"\n{'='*60}\nALL EXPERIMENTS COMPLETE\nPlots saved in '{out_dir}/'\n{'='*60}")


if __name__ == '__main__':
    main()
