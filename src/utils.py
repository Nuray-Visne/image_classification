import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def ensure_dataset(data_root: Path, class_names, max_images_per_class: int, max_samples: int, exact_match: bool = True):
    if data_root.exists():
        class_counts = get_class_counts(data_root, class_names)
        if exact_match:
            matches = all(count == max_images_per_class for count in class_counts.values())
        else:
            matches = all(count >= max_images_per_class for count in class_counts.values())

        if matches:
            print("Using existing dataset at", data_root)
            print("Existing class counts:", class_counts)
            return

        print("Existing dataset does not match requested sample size. Rebuilding...")
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


def count_parameters(model):
    total_parameters = int(sum(p.numel() for p in model.parameters()))
    trainable_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total_parameters, trainable_parameters


def measure_inference_time(model, loader, device, runs: int = 20):
    model.eval()
    images, _ = next(iter(loader))
    images = images[:1].to(device)
    times = []

    with torch.no_grad():
        _ = model(images)

        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start_time)

    avg_seconds = sum(times) / len(times)
    return avg_seconds, avg_seconds * 1000.0


def build_results_summary(rows):
    return pd.DataFrame([rows])


def build_history_dataframe(history, num_epochs: int):
    data = {"epoch": list(range(1, num_epochs + 1))}
    data.update(history)
    return pd.DataFrame(data)


def save_training_curves(history, num_epochs: int, figure_path: Path, title_prefix: str):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_figure(y_true, y_pred, class_names, figure_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(title)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
