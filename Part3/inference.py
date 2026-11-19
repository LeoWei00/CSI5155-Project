import os
from pathlib import Path

DATASET = "Shanmuk4622/ai-image-detection-dataset"   # <- swap this
CONFIG = "real"          # dataset configuration name, or None
TEST_SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"      # 0 = real, 1 = AI-generated

# Set EVALUATE=True to skip training and tests (used when the exported script is evaluated).
EVALUATE = os.environ.get("EVALUATE", "false").strip().lower() in ("true", "1", "yes")

# The dataset is gated, so Hugging Face needs an access token (https://huggingface.co/settings/tokens).
# An HF_TOKEN environment variable (or a `huggingface-cli login`) is used if set; otherwise paste your token here.
# This notebook is included in your submission, so clear the token before submitting.
HF_TOKEN = ""
if not os.environ.get("HF_TOKEN") and HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from huggingface_hub import get_token

if not EVALUATE and get_token() is None:
    print("Warning: no Hugging Face token found; downloading a gated dataset will fail.")

# weights.pt sits next to this file; __file__ is only defined in the exported script.
WEIGHTS = (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "weights.pt"


from datasets import load_dataset
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def make_model():
    """
    Create your model architecture. This is a placeholder model that flattens the input and applies
    a single linear layer. You should replace this with your own architecture.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

preprocess = transforms.Compose([
    transforms.Resize(
        (128, 128),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def transform(image):
    """
    Preprocess the image for your model. This is a placeholder transform that resizes the image to 128x128 and converts it to a tensor.
    You should replace this with your own preprocessing steps.
    """

    return preprocess(image.convert("RGB"))


def load_model(device):
    """
    Load your model and weights. This function loads the model architecture and the weights from 'weights.pt'.
    You should ensure that the model architecture matches the one used during training.
    You should not need to change this function.
    """

    model = make_model().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    return model.eval()


@torch.inference_mode()
def predict(model, source, split, device):
    """
    Predict a label for every image in load_dataset(source, split=split), in dataset order
    (0 for real, 1 for AI-generated). The evaluator passes a local folder of images with split "train".
    You can also run it on the Hub, e.g. predict(model, "Shanmuk4622/ai-image-detection-dataset", "test", device).

    Do not change the function signature or the assertions at the end of this function.
    You may change the implementation to ensure that the assertions pass,
     but the function must return a list of integer labels (0 or 1), one per dataset image.
    """
    dataset = load_dataset(
        source,
        split=split
    ).with_transform(
        lambda rows: {
            "pixels": [transform(image) for image in rows[IMAGE_COLUMN]]
        }
    )

    labels = []

    for batch in DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    ):
        images = batch["pixels"].to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        labels.extend(predictions.cpu().tolist())

    # Do not change the following assertions; they are used to validate your submission.
    # You may change the code above to ensure that these assertions pass.
    assert type(labels) is list, "predict must return a list of labels"
    assert len(labels) == len(dataset), "predict must return one label per image"
    assert all(type(label) is int and label in (0, 1) for label in labels), "Labels must be Python integers: 0=real, 1=AI"


    return labels


from itertools import islice

def load_samples(config, split, limit):
    dataset = load_dataset(
        DATASET,
        name=config,
        split=split,
        streaming=True
    )

    dataset = dataset.shuffle(seed=42, buffer_size=1000)

    samples = []
    expected_label = 0 if config == "real" else 1

    for row in islice(dataset, limit):
        label = int(row[LABEL_COLUMN])

        if label != expected_label:
            raise ValueError(f"Unexpected label in {config}")

        image = row[IMAGE_COLUMN].convert("RGB")
        image = image.resize((128, 128), Image.Resampling.BICUBIC)

        samples.append((image, label))

    if len(samples) != limit:
        raise ValueError(f"Not enough samples in {config}/{split}")

    print(f"Loaded {config}/{split}: {len(samples)}")
    return samples

from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.samples[index]
        return transform(image), label

from sklearn.metrics import f1_score

def train():
    """
    Train a model on a small subset of the dataset. This function is a placeholder and does not perform actual training.
    You should replace this with your own training code that trains your model on the dataset and saves
    the trained weights to 'weights.pt'.
    """

    project_root = Path.cwd()

    if project_root.name == "Part3":
        project_root = project_root.parent

    checkpoint_path = (
        project_root
        / "Part2"
        / "part2_results"
        / "best_resnet18_weights.pt"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = make_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)

    print("Loaded Phase 2 checkpoint:", checkpoint_path)
    print("Training device:", device)

    generators = ["sd15", "flux_schnell", "kandinsky22"]

    train_samples = load_samples("real", "train", 3000)
    val_samples = load_samples("real", "validation", 450)

    for generator in generators:
        train_samples.extend(
            load_samples(generator, "train", 1000)
        )
        val_samples.extend(
            load_samples(generator, "validation", 150)
        )

    print("Total training samples:", len(train_samples))
    print("Total validation samples:", len(val_samples))

    train_loader = DataLoader(
        ImageDataset(train_samples),
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        ImageDataset(val_samples),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune layer3, layer4, and fc
    for layer in [model.layer3, model.layer4, model.fc]:
        for param in layer.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer3.parameters(),
                "lr": 3e-5
            },
            {
                "params": model.layer4.parameters(),
                "lr": 3e-5
            },
            {
                "params": model.fc.parameters(),
                "lr": 1e-4
            }
        ],
        weight_decay=1e-4
    )

    epochs = 5
    best_f1 = -1.0

    for epoch in range(epochs):
        model.train()

        # Keep BatchNorm statistics frozen in frozen layers
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not any(
                    p.requires_grad
                    for p in module.parameters(recurse=False)
                ):
                    module.eval()

        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_predictions = []
        val_labels = []

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                val_predictions.extend(
                    predictions.cpu().tolist()
                )
                val_labels.extend(labels.tolist())

        val_f1 = f1_score(val_labels, val_predictions)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), WEIGHTS)
            print(f"Best model saved: {WEIGHTS}")

    print(f"Training completed. Best Val F1: {best_f1:.4f}")



# DO NOT CHANGE THIS LINE. The evaluator will set EVALUATE=True when it runs your exported script.
# THIS SHOULD BE THE ONLY ENTRY POINT FOR TRAINING. Handle all validation, tuning, etc, in train() or in other functions called by train().
if not EVALUATE:
    train()


if not EVALUATE:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    test_samples = load_samples("real", "test", 150)
    test_samples += load_samples("wuerstchen", "test", 150)

    test_loader = DataLoader(
        ImageDataset(test_samples),
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))


if not EVALUATE:
    baseline_model = make_model().to(device)

    baseline_path = (
        WEIGHTS.parent.parent
        / "Part2"
        / "part2_results"
        / "best_resnet18_weights.pt"
    )

    baseline_model.load_state_dict(
        torch.load(
            baseline_path,
            map_location=device,
            weights_only=True
        )
    )

    baseline_model.eval()
    baseline_predictions = []

    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = baseline_model(images.to(device))
            predictions = outputs.argmax(dim=1)
            baseline_predictions.extend(
                predictions.cpu().tolist()
            )

    print("Phase 2 Baseline:")
    print("Accuracy:", accuracy_score(y_true, baseline_predictions))
    print("Precision:", precision_score(y_true, baseline_predictions))
    print("Recall:", recall_score(y_true, baseline_predictions))
    print("F1:", f1_score(y_true, baseline_predictions))



if not EVALUATE:
    sd15_samples = load_samples("real", "test", 150)
    sd15_samples += load_samples("sd15", "test", 150)

    sd15_loader = DataLoader(
        ImageDataset(sd15_samples),
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    sd15_labels = [label for _, label in sd15_samples]

    for name, tested_model in [
        ("Phase 2", baseline_model),
        ("Phase 3", model)
    ]:
        tested_model.eval()
        predictions = []

        with torch.inference_mode():
            for images, _ in sd15_loader:
                outputs = tested_model(images.to(device))
                predictions.extend(
                    outputs.argmax(dim=1).cpu().tolist()
                )

        print(f"\n{name} - SD1.5 Test")
        print("Accuracy:", accuracy_score(sd15_labels, predictions))
        print("Precision:", precision_score(sd15_labels, predictions))
        print("Recall:", recall_score(sd15_labels, predictions))
        print("F1:", f1_score(sd15_labels, predictions))




if not EVALUATE:
    from torch.utils.data import IterableDataset

    class OriginalTestDataset(IterableDataset):
        def __iter__(self):
            for config in ["real", "sd15"]:
                dataset = load_dataset(
                    DATASET,
                    name=config,
                    split="test",
                    streaming=True
                )

                for row in dataset:
                    image = transform(row[IMAGE_COLUMN])
                    label = int(row[LABEL_COLUMN])
                    yield image, label

    for name, tested_model in [
        ("Phase 2", baseline_model),
        ("Phase 3", model)
    ]:
        test_loader = DataLoader(
            OriginalTestDataset(),
            batch_size=16,
            num_workers=0
        )

        tested_model.eval()
        y_true = []
        y_pred = []

        with torch.inference_mode():
            for images, labels in test_loader:
                outputs = tested_model(images.to(device))
                predictions = outputs.argmax(dim=1)

                y_true.extend(labels.tolist())
                y_pred.extend(predictions.cpu().tolist())

        print(f"\n{name} - Full Original Test Set")
        print("Samples:", len(y_true))
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("F1:", f1_score(y_true, y_pred))



if not EVALUATE:
    class WuerstchenTestDataset(IterableDataset):
        def __iter__(self):
            for config in ["real", "wuerstchen"]:
                dataset = load_dataset(
                    DATASET,
                    name=config,
                    split="test",
                    streaming=True
                )

                for row in dataset:
                    yield (
                        transform(row[IMAGE_COLUMN]),
                        int(row[LABEL_COLUMN])
                    )

    test_loader = DataLoader(
        WuerstchenTestDataset(),
        batch_size=16,
        num_workers=0
    )

    for name, tested_model in [
        ("Phase 2", baseline_model),
        ("Phase 3", model)
    ]:
        tested_model.eval()
        y_true = []
        y_pred = []

        with torch.inference_mode():
            for images, labels in test_loader:
                outputs = tested_model(images.to(device))
                predictions = outputs.argmax(dim=1)

                y_true.extend(labels.tolist())
                y_pred.extend(predictions.cpu().tolist())

        print(f"\n{name} - Full Wuerstchen Test")
        print("Samples:", len(y_true))
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("F1:", f1_score(y_true, y_pred))


import json
import subprocess
import sys
import time
from itertools import islice
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

NUM_IMAGES = 20             # images used for the submission-style test
SPEED_IMAGES = 5000         # the speed test requires SPEED_IMAGES images in under SPEED_LIMIT seconds on a T4
SPEED_LIMIT = 10 * 60
SPEED_SAMPLE = 250          # the speed test stops after SPEED_SAMPLE images or SPEED_DURATION seconds, whichever is first
SPEED_DURATION = 60
SPEED_CHUNK = 25            # images per predict call; the time limit is checked between calls
EXPORT = WEIGHTS.parent / "inference.py"   # the exported script, next to weights.pt

STUDENT_NUMBER = ""         # set this (digits only) to save submission_STUDENT_NUMBER.zip when the tests pass
# Holds Part1, Part2 and requirements.txt: the project folder when this notebook is in Part3, else /content in Colab.
ROOT = WEIGHTS.parent.parent if WEIGHTS.parent.name == "Part3" else WEIGHTS.parent
IGNORED = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}

# Runs in a separate isolated interpreter (-I) with EVALUATE=True and loads the exported script
# the way the evaluator does. Checks predict()'s output here too, in case its own assertions were edited away.
# With a folder of image chunks and a duration, it then calls predict on each chunk until they are
# done or the duration has passed, and prints the image count and elapsed seconds.
WORKER = """
import importlib.util
import os
import sys
import time
spec = importlib.util.spec_from_file_location("student_inference", "inference.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
device = sys.argv[1]
model = module.load_model(device)
labels = module.predict(model, sys.argv[2], 'train', device)
assert type(labels) is list, "predict must return a list of labels"
assert len(labels) == len(os.listdir(sys.argv[2])), "predict must return one label per image"
assert all(type(label) is int and label in (0, 1) for label in labels), "Labels must be Python integers: 0=real, 1=AI"
if len(sys.argv) > 3:
    count, start = 0, time.perf_counter()
    for chunk in sorted(os.listdir(sys.argv[3])):
        count += len(module.predict(model, os.path.join(sys.argv[3], chunk), 'train', device))
        if time.perf_counter() - start >= float(sys.argv[4]):
            break
    print(count, time.perf_counter() - start)
"""


def get_device():
    """The first CUDA GPU, else the Apple GPU (MPS), else the CPU."""
    return "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def is_t4():
    return torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)


def notebook_json():
    """This notebook: from the Colab editor, else the saved .ipynb that Jupyter or VS Code is running."""
    try:
        from google.colab import _message
    except ImportError:
        path = globals().get("__vsc_ipynb_file__") or os.environ.get("JPY_SESSION_NAME") or "inference.ipynb"
        return json.loads(Path(path).read_text())
    return _message.blocking_request("get_ipynb")["ipynb"]


def export_notebook():
    """Write this notebook's code cells to EXPORT, as a script."""
    sources = ["".join(cell["source"]) for cell in notebook_json()["cells"] if cell["cell_type"] == "code"]
    magics = [line for source in sources for line in source.splitlines() if line.lstrip().startswith(("%", "!"))]
    assert not magics, f"Remove these IPython magics; they do not run in the exported script: {magics}"
    EXPORT.write_text("\n\n\n".join(sources) + "\n")
    print(f"Exported this notebook to {EXPORT}")


def write_submission(archive):
    """Zip requirements.txt, Part1, Part2 and Part3 as test_submission.py does."""
    assert WEIGHTS.is_file(), f"Missing {WEIGHTS}; run train() first."
    assert (ROOT / "requirements.txt").is_file(), f"Missing {ROOT / 'requirements.txt'}; upload the template's and add your packages."
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        bundle.write(ROOT / "requirements.txt", "requirements.txt")
        parts = ["Part1", "Part2"] + (["Part3"] if WEIGHTS.parent.name == "Part3" else [])
        for part in parts:
            if not (ROOT / part).is_dir():
                print(f"Warning: no {part} folder in {ROOT}; it is missing from the zip.")
            for path in sorted((ROOT / part).rglob("*")):
                relative = path.relative_to(ROOT)
                if path.is_file() and not IGNORED & set(relative.parts) and path.suffix != ".zip":
                    bundle.write(path, relative.as_posix())
        if "Part3" not in parts:  # Colab: this notebook, its export and the weights
            bundle.writestr("Part3/inference.ipynb", json.dumps(notebook_json(), indent=1))
            bundle.write(EXPORT, "Part3/inference.py")
            bundle.write(WEIGHTS, "Part3/weights.pt")


def run_exported(device, part3, *args):
    """Run WORKER on the exported script in part3 and return its output."""
    process = subprocess.run([sys.executable, "-I", "-B", "-c", WORKER, device, *map(str, args)], cwd=part3,
                             capture_output=True, text=True, env={**os.environ, "EVALUATE": "True"})
    assert process.returncode == 0, f"The exported inference.py failed:\n{process.stderr}"
    return process.stdout


def save_test_images(folder, count):
    """Write the first count test images to a flat, unlabelled folder, as the evaluator passes to predict."""
    rows = load_dataset(DATASET, name=CONFIG, split=TEST_SPLIT, streaming=True)
    for i, row in enumerate(islice(rows, count)):
        row[IMAGE_COLUMN].convert("RGB").save(Path(folder) / f"{i:06d}.png", compress_level=1)
    return len(os.listdir(folder))


def test_submission_format(device, part3=EXPORT.parent):
    with TemporaryDirectory() as folder:
        saved = save_test_images(folder, NUM_IMAGES)
        assert saved == NUM_IMAGES, f"expected {NUM_IMAGES} test images, got {saved}"
        run_exported(device, part3, folder)
    print(f"PASS: exported model loads and predicts valid labels on {NUM_IMAGES} images from {DATASET} ({TEST_SPLIT}).")


def test_speed(device, part3=EXPORT.parent, duration=SPEED_DURATION):
    """
    Time the exported predict on SPEED_SAMPLE images, stopping after duration seconds if that comes first,
    and check the rate meets SPEED_IMAGES in SPEED_LIMIT seconds.
    """
    with TemporaryDirectory() as images, TemporaryDirectory() as folder:
        save_test_images(images, 1)
        save_test_images(folder, SPEED_SAMPLE)
        files = sorted(Path(folder).iterdir())
        for i in range(0, len(files), SPEED_CHUNK):
            chunk = Path(folder) / f"chunk{i // SPEED_CHUNK:03d}"
            chunk.mkdir()
            for file in files[i:i + SPEED_CHUNK]:
                file.rename(chunk / file.name)
        count, elapsed = run_exported(device, part3, images, folder, duration).split()[-2:]
    count, elapsed = int(count), float(elapsed)
    projected = SPEED_IMAGES * elapsed / count
    print(f"{count} images in {elapsed:.1f}s on {device}: {count / elapsed:.1f} images/s, "
          f"projected {projected:.0f}s for {SPEED_IMAGES} images (limit {SPEED_LIMIT}s)")
    assert projected < SPEED_LIMIT, f"too slow: {SPEED_IMAGES} images would take {projected:.0f}s, limit is {SPEED_LIMIT}s"
    print("PASS: speed")


if not EVALUATE:
    assert not STUDENT_NUMBER or STUDENT_NUMBER.isdigit(), "STUDENT_NUMBER must contain only digits"
    device = get_device()
    export_notebook()
    with TemporaryDirectory() as folder:
        # Test the zip contents, not the working folder.
        archive = Path(folder) / f"submission_{STUDENT_NUMBER or 'test'}.zip"
        write_submission(archive)
        with ZipFile(archive) as bundle:
            bundle.extractall(Path(folder) / "extracted")
        part3 = Path(folder) / "extracted" / "Part3"

        # Runs automatically on a T4 only; call test_speed(device) manually on other hardware.
        tests = [test_submission_format] + ([test_speed] if is_t4() else [])
        for test in tests:
            print(f"Running {test.__name__} on {device}...")
            start = time.perf_counter()
            test(device, part3)
            print(f"{test.__name__} took {time.perf_counter() - start:.0f}s\n")

        if not STUDENT_NUMBER:
            print("Set STUDENT_NUMBER in the tests cell above to save your submission zip.")
        else:
            # Only replace an existing submission once the new one passed.
            output = ROOT / archive.name
            output.write_bytes(archive.read_bytes())
            print(f"Submit {output}")
            try:
                from google.colab import files
            except ImportError:
                pass
            else:
                files.download(str(output))
