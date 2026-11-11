# test_final.py — final test of B0 and C1 on last year's val (my test set), then ONNX export of C1 seed 0
import json, os
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# These 4 test images are byte-identical copies of old train images (found by MD5 hash), so they are left out.
duplicates = ["maple/maple_004.jpg", "maple/maple_027.jpg", "maple/maple_028.jpg", "maple/maple_029.jpg"]


class LeafDataset(Dataset):
    # Copied unchanged from 3a.
    def __init__(self, folder, transform):
        self.transform = transform
        # The class names: the sorted subfolders of the folder.
        folder_path = Path(folder)
        self.classes = sorted([d.name for d in folder_path.iterdir() if d.is_dir()])

        # A list of (file path, class index) pairs: every class, its files in sorted order.
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = folder_path / class_name
            # Find files in sorted order
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file():
                    self.samples.append((str(img_path), class_idx))

    def __len__(self):
        # The number of samples.
        return len(self.samples)

    def __getitem__(self, i):
        # Open the i-th image as RGB, apply the transform, return it with its class index.
        img_path, class_idx = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, class_idx


test_ds = LeafDataset("data/old_leaves/val", transform)
test_ds.samples = [s for s in test_ds.samples if "/".join(Path(s[0]).parts[-2:]) not in duplicates]
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
print("test images:", len(test_ds), "| classes:", test_ds.classes)


def load_model(path):
    # Same architecture as in task4.py (MobileNetV3 small, last layer 4 outputs), weights from the file.
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def accuracy(model, loader):
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


# V9 added after the final test (post-hoc, exploratory).
for name in ["B0", "C1", "V9"]:
    accs = [accuracy(load_model(f"models/{name}_seed{seed}.pt"), test_loader) for seed in [0, 1, 2]]
    print(f"{name}: " + "  ".join(f"seed {s} {a:.3f}" for s, a in enumerate(accs)) +
          f"  | mean {np.mean(accs):.3f} ± {np.std(accs, ddof=1):.3f}")

# Confusion matrices for seed 0: rows = true class, columns = predicted class. (V9 post-hoc.)
for name in ["V9", "C1"]:  # C1 last: the export below uses this model
    model = load_model(f"models/{name}_seed0.pt")
    matrix = torch.zeros(4, 4, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in test_loader:
            for t, p in zip(labels, model(images).argmax(dim=1)):
                matrix[t, p] += 1
    print(f"\n{name} seed 0 confusion matrix (rows true, columns predicted)")
    print(" " * 10 + "".join(f"{c:>10}" for c in test_ds.classes))
    for i, c in enumerate(test_ds.classes):
        print(f"{c:>10}" + "".join(f"{v:>10d}" for v in matrix[i].tolist()))
print("one test image = 1/116 ≈ 0.9 %")

# Export C1 seed 0 (chosen before the test) exactly as in cell 2e, into a separate file.
torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), "leaf_model_best.onnx",
                  input_names=["image"], output_names=["logits"], external_data=False)
model_info = {"classes": test_ds.classes, "input_size": 96, "mean": mean, "std": std}
onnx_model = onnx.load("leaf_model_best.onnx")
onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
onnx.save(onnx_model, "leaf_model_best.onnx")
print(f"\nleaf_model_best.onnx: {os.path.getsize('leaf_model_best.onnx') / 1e6:.1f} MB")

# Check: onnxruntime gives the same probabilities as PyTorch on check_image.jpg.
x = transform(Image.open("check_image.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    p_torch = model(x).softmax(dim=1)[0].numpy()
logits = ort.InferenceSession("leaf_model_best.onnx").run(None, {"image": x.numpy()})[0]
p_onnx = torch.tensor(logits).softmax(dim=1)[0].numpy()
for c, a, b in zip(test_ds.classes, p_torch, p_onnx):
    print(f"{c:>10}  pytorch {a:.3f}  onnx {b:.3f}")
print("same to 3 decimals:", bool((p_torch.round(3) == p_onnx.round(3)).all()))
