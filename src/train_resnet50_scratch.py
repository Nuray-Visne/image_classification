"""Train a scratch ResNet50 classifier on the Open Images subset.

Requirements:
- Install dependencies from `requirements.txt`
- Run from the project root so `src/` and `results/` resolve correctly


Outputs:
- `results/resnet50_scratch.pth`
- `results/resnet50_scratch_summary.csv`
- `results/resnet50_scratch_history.csv`
- `results/resnet50_scratch_curves.png`
- `results/resnet50_scratch_confusion_matrix.png`

Run:
- `python src/train_resnet50_scratch.py`
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from utils import (
    build_history_dataframe,
    build_results_summary,
    count_parameters,
    create_train_test_split,
    ensure_dataset,
    evaluate,
    measure_inference_time,
    resolve_paths,
    save_confusion_matrix_figure,
    save_training_curves,
    set_seed,
    train_one_epoch,
)


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 4
TARGET_CLASSES = ["Egg (Food)", "Chicken", "Balloon"]
MAX_IMAGES_PER_CLASS = 500
MAX_SOURCE_SAMPLES = 2000

def build_scratch_resnet50(num_classes: int):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    set_seed(SEED)

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

    model = build_scratch_resnet50(num_classes=len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_parameters, trainable_parameters = count_parameters(model)
    print("Number of parameters:", num_parameters)
    print("Trainable parameters:", trainable_parameters)

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

    final_test_loss, final_test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    final_train_acc = history["train_acc"][-1]
    inference_time_seconds, inference_time_ms = measure_inference_time(model, test_loader, device)
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Final test accuracy: {final_test_acc * 100:.2f}%")
    print(f"Total training time: {total_training_time_seconds:.2f}s")
    print(f"Average inference time: {inference_time_ms:.2f} ms per image")

    results_summary = build_results_summary(
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
            "trainable_parameters": trainable_parameters,
            "total_training_time_seconds": total_training_time_seconds,
            "average_epoch_time_seconds": sum(history["epoch_time_seconds"]) / len(history["epoch_time_seconds"]),
            "average_inference_time_seconds": inference_time_seconds,
            "average_inference_time_ms": inference_time_ms,
        }
    )

    history_df = build_history_dataframe(history, NUM_EPOCHS)

    summary_path = results_root / "resnet50_scratch_summary.csv"
    history_path = results_root / "resnet50_scratch_history.csv"
    figure_path = results_root / "resnet50_scratch_curves.png"
    confusion_matrix_path = results_root / "resnet50_scratch_confusion_matrix.png"
    checkpoint_path = results_root / "resnet50_scratch.pth"

    results_summary.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)
    torch.save(model.state_dict(), checkpoint_path)

    save_training_curves(history, NUM_EPOCHS, figure_path, "ResNet50 Scratch")
    save_confusion_matrix_figure(
        y_true,
        y_pred,
        test_dataset.classes,
        confusion_matrix_path,
        "ResNet50 Scratch Confusion Matrix",
    )
    print("Saved:", summary_path)
    print("Saved:", history_path)
    print("Saved:", figure_path)
    print("Saved:", confusion_matrix_path)
    print("Saved:", checkpoint_path)


if __name__ == "__main__":
    main()
