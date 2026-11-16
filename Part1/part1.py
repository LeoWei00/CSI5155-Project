import os

os.environ["KERAS_BACKEND"] = "torch"  # Must be set before importing Keras.

import csv
import random
from collections import Counter
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import get_token, login
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

SEED = 65 # Better distribution for random search
TRIALS = 8
EPOCHS = 30
PATIENCE = 5
NUM_WORKERS = 4 
OUTPUT_DIR = Path(__file__).resolve().parent / Path("part1_results")

DATASET_NAME = "Shanmuk4622/ai-image-detection-dataset"
CLASS_NAMES = ["Real", "AI"]


# 1. Load only real photographs and Stable Diffusion 1.5 images.
def load_sd15_task(split):
    from datasets import concatenate_datasets, load_dataset

    real = load_dataset(DATASET_NAME, "real", split=split)
    sd15 = load_dataset(DATASET_NAME, "sd15", split=split)
    return concatenate_datasets([real, sd15]).shuffle(seed=42)


# 2. Preprocessing.
def downsample(img):
    return img.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)


def get_batch(batch):
    images = np.stack([
        np.asarray(downsample(example["image"]), dtype=np.float32) / 255.0
        for example in batch
    ])
    labels = np.asarray([[example["label"]] for example in batch], dtype=np.float32)
    return torch.from_numpy(images), torch.from_numpy(labels)


# 3. Configurable CNN.
def create_cnn(num_blocks, start_filters, dropout):
    layers = [keras.layers.Input(shape=(128, 128, 3))]

    for i in range(num_blocks):
        filters = start_filters * (2 ** i)
        layers.extend([
            keras.layers.Conv2D(filters, kernel_size=3, padding="same"),
            keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D(2),
        ])

    layers.extend([
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    return keras.Sequential(layers)

# 4. Random sample space for hyperparameter tuning.
def random_search_space(seed, num_trials):
    rng = random.Random(seed)
    configs = []

    for _ in range(num_trials):
        configs.append({
            "learning_rate": 10 ** rng.uniform(-4, -2.5),
            "batch_size": rng.choice([16, 32, 64]),
            "dropout": rng.uniform(0.0, 0.5),
            "num_blocks": rng.choice([2, 3, 4]),
            "start_filters": rng.choice([16, 24, 32]),
        })

    return configs

# 5. Training and hyperparameter tuning.
def fit_trial(params, train_data, val_data):
    keras.backend.clear_session()
    model = create_cnn(
        num_blocks=params["num_blocks"],
        start_filters=params["start_filters"],
        dropout=params["dropout"],
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=get_batch,
    )
    val_loader = DataLoader(
        val_data, batch_size=params["batch_size"], shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=get_batch,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=params["learning_rate"], weight_decay=1e-4
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(average="micro", threshold=0.5, name="f1"),
        ],
    )

    # Keras trains, validates, stops early, and restores the best weights.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_f1", mode="max", patience=PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    ).history

    best_epoch = int(np.argmax(history["val_f1"])) + 1
    return {
        "params": params,
        "best_val_f1": float(history["val_f1"][best_epoch - 1]),
        "best_epoch": best_epoch,
        "history": history,
        "weights": model.get_weights(),
    }

# 6. Draw learning curves.
def save_learning_curve(history, best_epoch, path, title, params):
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, history["loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.axvline(best_epoch, linestyle="--", label="Best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title(
        f"{title}\n"
        f"LR={params['learning_rate']:.2e} | Batch={params['batch_size']} | "
        f"Dropout={params['dropout']:.3f}\n"
        f"Blocks={params['num_blocks']} | Start filters={params['start_filters']}",
        fontsize=11,
        pad=12,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

# 6. Experiment and test evaluation.
def main():
    keras.utils.set_random_seed(SEED)
    print("Keras backend:", keras.backend.backend())
    print("Training device:", "cuda" if torch.cuda.is_available() else "cpu")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_plot_dir = OUTPUT_DIR / "trial_learning_curves"
    trial_plot_dir.mkdir(exist_ok=True)

    if get_token() is None:
        print("Sign in to Hugging Face after accepting the dataset's access conditions.")
        login()

    print("Loading real and Stable Diffusion 1.5 images...")
    train = load_sd15_task("train")
    val = load_sd15_task("validation")
    test = load_sd15_task("test")

    print("\nDATASET SANITY CHECKS")
    for name, ds in [("Train", train), ("Validation", val), ("Test", test)]:
        counts = Counter(ds["label"])
        if set(counts) != {0, 1}:
            raise ValueError(f"{name}: expected labels 0 and 1; found {dict(counts)}")
        print(f"\n{name}: {len(ds)} images")
        print(f"  Real: {counts[0]}, AI: {counts[1]}")
        print(f"  AI proportion: {counts[1] / len(ds):.1%}")

    print("\nDataset columns:", train.column_names)
    sample = train[0]["image"]
    print("Example image size:", sample.size)
    print("Example image mode:", sample.mode)

    configs = random_search_space(SEED, TRIALS)
    best_trial = None

    for i, params in enumerate(configs, start=1):
        print(f"\nTrial {i}/{TRIALS}: {params}")
        keras.utils.set_random_seed(SEED + i)
        result = fit_trial(params, train, val)
        print(f"  Best validation F1: {result['best_val_f1']:.4f}")

        save_learning_curve(
            result["history"], result["best_epoch"],
            trial_plot_dir / f"trial_{i:02d}.png",
            f"Trial {i}: training and validation loss", params,
        )

        if best_trial is None or result["best_val_f1"] > best_trial["best_val_f1"]:
            best_trial = result
            best_trial["trial"] = i

    print("\nSELECTED CNN")
    print("Trial:", best_trial["trial"])
    print("Hyperparameters:", best_trial["params"])
    print("Best epoch:", best_trial["best_epoch"])
    print(f"Validation F1: {best_trial['best_val_f1']:.4f}")

    # Evaluate only the selected model on the held-out test set.
    params = best_trial["params"]
    best_model = create_cnn(params["num_blocks"], params["start_filters"], params["dropout"])
    best_model.to("cuda" if torch.cuda.is_available() else "cpu")
    best_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=params["learning_rate"], weight_decay=1e-4
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(average="micro", threshold=0.5, name="f1"),
        ],
    )
    best_model.set_weights(best_trial["weights"])

    test_loader = DataLoader(
        test, batch_size=params["batch_size"], shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=get_batch,
    )
    test_metrics = best_model.evaluate(test_loader, verbose=0, return_dict=True)
    probabilities = best_model.predict(test_loader, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    true_labels = np.asarray(test["label"], dtype=int)
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])

    with open(OUTPUT_DIR / "test_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name in ("accuracy", "precision", "recall", "f1"):
            writer.writerow([name.title(), f"{test_metrics[name]:.4f}"])

    print("\nFINAL TEST METRICS (AI = positive class)")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"  {name.title()}: {test_metrics[name]:.4f}")

    save_learning_curve(
        best_trial["history"], best_trial["best_epoch"],
        OUTPUT_DIR / "learning_curves.png",
        "Best CNN trial: training and validation loss", params,
    )
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    display.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title("Final test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
