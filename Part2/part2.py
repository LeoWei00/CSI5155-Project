import csv
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import get_token, login
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

# 1. Experiment settings.
SEED = 42
EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 32
NUM_WORKERS = 4 
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 128
DATASET_NAME = "Shanmuk4622/ai-image-detection-dataset"
CLASS_NAMES = ["Real", "AI"]  # Label 1 (AI) is the positive class.
OUTPUT_DIR = Path(__file__).resolve().parent / Path("part2_results")

# Each tuple is (FC learning rate, unfrozen backbone learning rate).
LR_OPTIONS = {
    "fc_only":       [(1e-3, None), (3e-4, None)],
    "last_block":    [(1e-3, 1e-4), (3e-4, 3e-5)],
    "last_2_blocks": [(1e-3, 1e-4), (3e-4, 3e-5)],
    "full":          [(1e-3, 1e-5), (3e-4, 3e-5)],
}
STRATEGY_NAMES = {
    "fc_only": "FC only",
    "last_block": "Layer 4 + FC",
    "last_2_blocks": "Layers 3-4 + FC",
    "full": "Full fine-tuning",
}

# Exact RGB mean/std used for ImageNet-pretrained torchvision ResNet18.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 2. Same dataset/splits as Part 1.
def load_sd15_task(split):
    from datasets import concatenate_datasets, load_dataset

    real = load_dataset(DATASET_NAME, "real", split=split)
    sd15 = load_dataset(DATASET_NAME, "sd15", split=split)
    return concatenate_datasets([real, sd15]).shuffle(seed=42)


def downsample(img):
    return img.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
    )


def prepare_batch(batch):
    images = np.stack([
        np.asarray(downsample(example["image"]), dtype=np.float32) / 255.0
        for example in batch
    ])
    images = (images - IMAGENET_MEAN) / IMAGENET_STD
    # TorchVision models require channels first: (batch, 3, height, width).
    images = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
    labels = torch.tensor([example["label"] for example in batch], dtype=torch.long)
    return torch.from_numpy(images), labels


def make_loader(data, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=prepare_batch,
        generator=generator,
    )


# 3. Pretrained architecture.
def create_resnet(strategy):
    # Load an independent copy of the specified ImageNet-1K pretrained model.
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze pretrained weights first, then specify trainable stages below.
    for param in model.parameters():
        param.requires_grad = False

    # The original 1000-class FC layer is replaced with a trainable 2-class head.
    model.fc = nn.Linear(model.fc.in_features, 2)

    if strategy in ("last_block", "last_2_blocks"):
        for param in model.layer4.parameters():
            param.requires_grad = True

    if strategy == "last_2_blocks":
        for param in model.layer3.parameters():
            param.requires_grad = True

    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True

    if strategy not in LR_OPTIONS:
        raise ValueError(f"Unknown fine-tuning strategy: {strategy}")

    return model


def set_training_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):  # noqa: SIM102
            if not any(p.requires_grad for p in module.parameters(recurse=False)):
                module.eval()


def make_optimizer(model, fc_lr, backbone_lr):
    parameter_groups = [{"params": model.fc.parameters(), "lr": fc_lr}]
    backbone_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("fc.") and p.requires_grad
    ]
    if backbone_params:
        if backbone_lr is None:
            raise ValueError("Trainable backbone requires a backbone learning rate")
        parameter_groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(parameter_groups, weight_decay=WEIGHT_DECAY)


def classification_metrics(true_labels, predictions):
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(true_labels, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(true_labels, predictions, pos_label=1, zero_division=0),
    }


# 4. Training and validation.
def train_epoch(model, loader, criterion, optimizer, device):
    set_training_mode(model)
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    true_labels, predictions = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        true_labels.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(true_labels, predictions)
    return total_loss / len(loader.dataset), metrics, true_labels, predictions


def fit_trial(strategy, fc_lr, backbone_lr, train_data, val_loader, device):
    set_seed(SEED)
    model = create_resnet(strategy).to(device)
    train_loader = make_loader(train_data, shuffle=True, seed=SEED)
    optimizer = make_optimizer(model, fc_lr, backbone_lr)
    criterion = nn.CrossEntropyLoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    history = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    bad_epochs = 0

    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
        })

        print(
            f"  Epoch {epoch:02d}/{EPOCHS}: "
            f"train loss={train_loss:.4f} | val loss={val_loss:.4f} | "
            f"val F1={val_metrics['f1']:.4f} | val acc={val_metrics['accuracy']:.4f}"
        )

        # Maximize validation F1. Break equal-F1 ties using lower validation loss.
        improved = (
            val_metrics["f1"] > best_f1 + 1e-8
            or (
                abs(val_metrics["f1"] - best_f1) <= 1e-8
                and val_loss < best_loss - 1e-8
            )
        )
        if improved:
            best_f1 = val_metrics["f1"]
            best_loss = val_loss
            best_epoch = epoch
            # CPU copy includes BatchNorm buffers. No checkpoint file is written.
            best_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  Early stopping after {epoch} epochs")
                break

    elapsed = time.perf_counter() - start
    del model, optimizer, train_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "history": history,
        "weights": best_weights,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_val_loss": best_loss,
        "training_seconds": elapsed,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "fc_lr": fc_lr,
        "backbone_lr": backbone_lr,
    }


def is_better(candidate, incumbent):
    return (
        incumbent is None
        or candidate["best_val_f1"] > incumbent["best_val_f1"] + 1e-8
        or (
            abs(candidate["best_val_f1"] - incumbent["best_val_f1"]) <= 1e-8
            and candidate["best_val_loss"] < incumbent["best_val_loss"]
        )
    )


# 5. Figures and CSV outputs.
def save_learning_curve(history, best_epoch, path, title, fc_lr, backbone_lr):
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Training loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Validation loss")
    plt.axvline(best_epoch, linestyle="--", label="Best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    lr_description = f"FC LR={fc_lr:.1e}"
    if backbone_lr is not None:
        lr_description += f" | Backbone LR={backbone_lr:.1e}"
    plt.title(f"{title}\n{lr_description} | Batch={BATCH_SIZE}", fontsize=11, pad=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrices(matrices):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, strategy in zip(axes.flat, STRATEGY_NAMES):
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrices[strategy], display_labels=CLASS_NAMES
        )
        display.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
        ax.set_title(STRATEGY_NAMES[strategy])
    fig.suptitle("Phase 2 test confusion matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Framework: PyTorch / TorchVision (Phase 1 Keras file is unchanged)")
    print("Training device:", device)
    print("Image size:", IMAGE_SIZE, "x", IMAGE_SIZE)
    print("Pretrained weights: ResNet18_Weights.IMAGENET1K_V1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_dir = OUTPUT_DIR / "learning_curves"
    curve_dir.mkdir(exist_ok=True)

    if get_token() is None:
        print("Sign in to Hugging Face after accepting the dataset access conditions.")
        login()

    print("\nLoading the same real and SD 1.5 dataset splits as Phase 1...")
    train_data = load_sd15_task("train")
    val_data = load_sd15_task("validation")
    test_data = load_sd15_task("test")
    for split, dataset in (("Train", train_data), ("Validation", val_data), ("Test", test_data)):
        counts = Counter(dataset["label"])
        if set(counts) != {0, 1}:
            raise ValueError(f"{split}: expected labels 0/1, got {dict(counts)}")
        print(f"{split}: {len(dataset)} images (Real={counts[0]}, AI={counts[1]})")

    val_loader = make_loader(val_data, shuffle=False, seed=SEED)
    test_loader = make_loader(test_data, shuffle=False, seed=SEED)
    criterion = nn.CrossEntropyLoss()

    trials = []
    comparison = []
    matrices = {}
    for strategy, candidates in LR_OPTIONS.items():
        print(f"\n{'=' * 65}\nSTRATEGY: {STRATEGY_NAMES[strategy]}\n{'=' * 65}")
        best = None
        strategy_seconds = 0.0

        for trial_number, (fc_lr, backbone_lr) in enumerate(candidates, start=1):
            print(f"\nLR trial {trial_number}/{len(candidates)}: FC={fc_lr:.1e}, backbone={backbone_lr}")
            result = fit_trial(
                strategy, fc_lr, backbone_lr, train_data, val_loader, device
            )
            strategy_seconds += result["training_seconds"]
            save_learning_curve(
                result["history"], result["best_epoch"],
                curve_dir / f"{strategy}_trial_{trial_number:02d}.png",
                f"{STRATEGY_NAMES[strategy]}: trial {trial_number}", fc_lr, backbone_lr,
            )
            trial_row = {
                "strategy": strategy,
                "lr_trial": trial_number,
                "fc_lr": fc_lr,
                "backbone_lr": "" if backbone_lr is None else backbone_lr,
                "best_epoch": result["best_epoch"],
                "epochs_run": len(result["history"]),
                "best_val_f1": f"{result['best_val_f1']:.4f}",
                "best_val_loss": f"{result['best_val_loss']:.4f}",
                "training_seconds": f"{result['training_seconds']:.1f}",
                "trainable_params": result["trainable_params"],
            }
            trials.append(trial_row)
            save_csv(OUTPUT_DIR / "learning_rate_trials.csv", trials, list(trial_row))
            print(f"Best validation F1 for this LR trial: {result['best_val_f1']:.4f}")

            if is_better(result, best):
                best = result
                best["lr_trial"] = trial_number


        print(f"\nSelected learning-rate trial: {best['lr_trial']}")
        print(f"Best validation F1: {best['best_val_f1']:.4f}")
        set_seed(SEED)
        model = create_resnet(strategy)
        model.load_state_dict(best["weights"])
        model = model.to(device)
        _, test_metrics, true_labels, predictions = evaluate(
            model, test_loader, criterion, device
        )
        matrices[strategy] = confusion_matrix(true_labels, predictions, labels=[0, 1])

        result_row = {
            "strategy": strategy,
            "strategy_name": STRATEGY_NAMES[strategy],
            "fc_lr": best["fc_lr"],
            "backbone_lr": "" if best["backbone_lr"] is None else best["backbone_lr"],
            "selected_lr_trial": best["lr_trial"],
            "best_epoch": best["best_epoch"],
            "best_val_f1": f"{best['best_val_f1']:.4f}",
            "accuracy": f"{test_metrics['accuracy']:.4f}",
            "precision": f"{test_metrics['precision']:.4f}",
            "recall": f"{test_metrics['recall']:.4f}",
            "f1": f"{test_metrics['f1']:.4f}",
            "trainable_params": best["trainable_params"],
            "total_params": best["total_params"],
            "training_seconds_all_lr_trials": f"{strategy_seconds:.1f}",
            "training_seconds_selected_trial": f"{best['training_seconds']:.1f}",
        }
        comparison.append(result_row)
        save_csv(OUTPUT_DIR / "strategy_comparison.csv", comparison, list(result_row))
        print("TEST RESULTS (AI = positive class)")
        for name in ("accuracy", "precision", "recall", "f1"):
            print(f"  {name.title()}: {test_metrics[name]:.4f}")

        del model, best
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_confusion_matrices(matrices)

    print("\nFINAL PHASE 2 COMPARISON (test set, AI = positive class)")
    print(f"{'Strategy':<21} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for row in comparison:
        print(
            f"{row['strategy_name']:<21} {row['accuracy']:>9} "
            f"{row['precision']:>10} {row['recall']:>8} {row['f1']:>8}"
        )
    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")
    print("No trained models were saved.")


if __name__ == "__main__":
    main()
