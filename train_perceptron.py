"""
Perceptron Learning Rule on MNIST
Uses the Mini-Torch framework to classify handwritten digits.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from examples.MNISTDataset import MNISTDataset
from DataLoader import DataLoader
from Linear import Linear
from Threshold import Threshold
from Error import Error


def compute_accuracy(linear, threshold, dataloader):
    """Compute classification accuracy over a dataset."""
    correct = 0
    total = 0
    for x, y in dataloader:
        raw = linear.forward(x)
        pred = threshold.forward(raw)
        correct += np.sum(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
        total += x.shape[0]
    return correct / total
# end function


def main():
    # --- Configuration ---
    DATA_DIR = "data"
    NUM_EPOCHS = 10
    BATCH_SIZE = 1
    np.random.seed(42)

    # --- Load Data ---
    print("Loading training data...")
    train_dataset = MNISTDataset(f"{DATA_DIR}/train.csv", is_train=True)
    print(f"  Training samples: {len(train_dataset)}")

    print("Loading test data...")
    test_dataset = MNISTDataset(f"{DATA_DIR}/test.csv", is_train=False)
    # Load test labels from solutions file
    import csv
    with open(f"{DATA_DIR}/test-solutions.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        test_labels = []
        for row in reader:
            test_labels.append(int(row[1]))
    test_dataset.labels = np.array(test_labels, dtype=int)
    test_dataset.is_train = True  # so __getitem__ returns one-hot labels
    print(f"  Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Build Model ---
    linear = Linear(in_features=784, out_features=10)
    threshold = Threshold(threshold=0.5)
    loss_fn = Error()

    # --- Training Loop ---
    epoch_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            # Forward pass
            raw = linear.forward(x)
            pred = threshold.forward(raw)

            # Compute loss
            loss_val = loss_fn.forward(pred, y)
            running_loss += loss_val
            num_batches += 1

            # Backward pass (perceptron learning rule updates weights)
            error = loss_fn.backward()
            threshold.backward(error)
            linear.backward(error)
        # end for

        avg_loss = running_loss / num_batches
        epoch_losses.append(avg_loss)

        # Evaluate accuracy
        train_acc = compute_accuracy(linear, threshold, train_loader)
        test_acc = compute_accuracy(linear, threshold, test_loader)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Test Acc: {test_acc*100:.2f}%")
    # end for

    # --- Plot Results ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    axes[0].plot(range(1, NUM_EPOCHS+1), epoch_losses, 'b-o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Loss')
    axes[0].set_title('Training Loss over Epochs')
    axes[0].grid(True)

    # Accuracy curves
    axes[1].plot(range(1, NUM_EPOCHS+1), [a*100 for a in train_accuracies], 'b-o', label='Train')
    axes[1].plot(range(1, NUM_EPOCHS+1), [a*100 for a in test_accuracies], 'r-s', label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Classification Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_results.pdf', dpi=150)
    plt.savefig('training_results.png', dpi=150)
    print("\nPlots saved to training_results.pdf and training_results.png")

    # --- Confusion Matrix ---
    from collections import defaultdict
    confusion = np.zeros((10, 10), dtype=int)
    for x, y in test_loader:
        raw = linear.forward(x)
        pred = threshold.forward(raw)
        p = np.argmax(pred, axis=1)
        t = np.argmax(y, axis=1)
        for pi, ti in zip(p, t):
            confusion[ti, pi] += 1

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im = ax2.imshow(confusion, cmap='Blues')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix')
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax2.text(j, i, str(confusion[i, j]),
                     ha='center', va='center',
                     color='white' if confusion[i, j] > confusion.max()/2 else 'black',
                     fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('confusion_matrix.pdf', dpi=150)
    plt.savefig('confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to confusion_matrix.pdf and confusion_matrix.png")

    # --- Sample Predictions ---
    fig3, axes3 = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes3.flat):
        x, y = test_dataset[i]
        img = x.reshape(28, 28)
        raw = linear.forward(x.reshape(1, -1))
        pred = threshold.forward(raw)
        predicted_digit = np.argmax(pred)
        actual_digit = np.argmax(y)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {predicted_digit}, True: {actual_digit}")
        ax.axis('off')
    plt.suptitle('Sample Test Predictions')
    plt.tight_layout()
    plt.savefig('sample_predictions.pdf', dpi=150)
    plt.savefig('sample_predictions.png', dpi=150)
    print("Sample predictions saved to sample_predictions.pdf and sample_predictions.png")

# end function

if __name__ == "__main__":
    main()
