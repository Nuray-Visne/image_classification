import os
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 4
TARGET_CLASSES = ["Egg (Food)", "Chicken", "Balloon"]
MAX_IMAGES_PER_CLASS = 500
MAX_SOURCE_SAMPLES = 2000


def detect_env() -> str:
    if Path("/kaggle").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"


def resolve_paths():
    env = detect_env()
    repo_root = Path.cwd()
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))

    if env == "kaggle":
        base_data_dir = Path("/kaggle/working/data")
    elif env == "colab":
        base_data_dir = Path("/content/data")
    else:
        base_data_dir = repo_root / "data"

    data_root = base_data_dir / "openimages_subset" / "classification"
    split_root = base_data_dir / "openimages_subset_split"
    results_root = repo_root / "results"

    base_data_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    return env, repo_root, data_root, split_root, results_root


def get_class_counts(data_root: Path, class_names):
    counts = {}
    for class_name in class_names:
        class_dir = data_root / class_name
        if class_dir.exists():
            counts[class_name] = len([p for p in class_dir.iterdir() if p.is_file()])
        else:
            counts[class_name] = 0
    return counts


def ensure_dataset(data_root: Path, class_names, max_images_per_class: int, max_samples: int):
    if data_root.exists():
        class_counts = get_class_counts(data_root, class_names)
        if all(count == max_images_per_class for count in class_counts.values()):
            print("Using existing dataset at", data_root)
            print("Existing class counts:", class_counts)
            return

        print("Existing dataset does not exactly match requested sample size. Rebuilding...")
        print("Existing class counts:", class_counts)
        shutil.rmtree(data_root)

    from download_images import create_classification_dataset_from_openimages

    create_classification_dataset_from_openimages(
        class_names=class_names,
        output_dir=str(data_root),
        max_images_per_class=max_images_per_class,
        max_samples=max_samples,
    )


def create_train_test_split(source_dir: Path, split_root: Path, train_ratio: float = 0.8, seed: int = 42):
    rng = random.Random(seed)

    if split_root.exists():
        shutil.rmtree(split_root)

    train_dir = split_root / "train"
    test_dir = split_root / "test"
    summary = {}

    for class_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        files = sorted([p for p in class_dir.iterdir() if p.is_file()])
        if not files:
            continue

        rng.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        class_train_dir = train_dir / class_dir.name
        class_test_dir = test_dir / class_dir.name
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_test_dir.mkdir(parents=True, exist_ok=True)

        for src in train_files:
            shutil.copy2(src, class_train_dir / src.name)

        for src in test_files:
            shutil.copy2(src, class_test_dir / src.name)

        summary[class_dir.name] = {"train": len(train_files), "test": len(test_files)}

    return train_dir, test_dir, summary


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_examples += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_examples
    avg_acc = accuracy_score(all_labels, all_preds)
    return avg_loss, avg_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_examples += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_examples
    avg_acc = accuracy_score(all_labels, all_preds)
    return avg_loss, avg_acc, all_labels, all_preds


def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, repo_root, data_root, split_root, results_root = resolve_paths()

    print("Environment:", env)
    print("Data root:", data_root)
    print("Results root:", results_root)

    ensure_dataset(
        data_root,
        class_names=TARGET_CLASSES,
        max_images_per_class=MAX_IMAGES_PER_CLASS,
        max_samples=MAX_SOURCE_SAMPLES,
    )

    train_dir, test_dir, split_summary = create_train_test_split(
        data_root, split_root, train_ratio=0.8, seed=SEED
    )
    print("Split summary:", split_summary)

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Classes:", train_dataset.classes)
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_parameters = int(sum(p.numel() for p in model.parameters()))
    print("Number of parameters:", num_parameters)

    history = {
        "epoch_time_seconds": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    training_start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        epoch_time_seconds = time.time() - epoch_start_time

        history["epoch_time_seconds"].append(epoch_time_seconds)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"Time: {epoch_time_seconds:.2f}s"
        )
    total_training_time_seconds = time.time() - training_start_time

    final_test_loss, final_test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    final_train_acc = history["train_acc"][-1]
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Final test accuracy: {final_test_acc * 100:.2f}%")
    print(f"Total training time: {total_training_time_seconds:.2f}s")

    results_summary = pd.DataFrame(
        [
            {
                "model": "resnet50_scratch",
                "seed": SEED,
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "classes": ", ".join(train_dataset.classes),
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "final_train_accuracy": final_train_acc,
                "final_test_accuracy": final_test_acc,
                "final_test_loss": final_test_loss,
                "num_parameters": num_parameters,
                "total_training_time_seconds": total_training_time_seconds,
                "average_epoch_time_seconds": sum(history["epoch_time_seconds"]) / len(history["epoch_time_seconds"]),
            }
        ]
    )

    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, NUM_EPOCHS + 1)),
            "epoch_time_seconds": history["epoch_time_seconds"],
            "train_loss": history["train_loss"],
            "train_acc": history["train_acc"],
            "test_loss": history["test_loss"],
            "test_acc": history["test_acc"],
        }
    )

    summary_path = results_root / "resnet50_scratch_summary.csv"
    history_path = results_root / "resnet50_scratch_history.csv"
    figure_path = results_root / "resnet50_scratch_curves.png"

    results_summary.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)

    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss over {NUM_EPOCHS} epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over {NUM_EPOCHS} epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    print("Saved:", summary_path)
    print("Saved:", history_path)
    print("Saved:", figure_path)


if __name__ == "__main__":
    main()
