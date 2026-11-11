# test_final_v2.py — Phase C: export of C1 seed 0, then the ONE evaluation of B0 and C1 on the test split.
# Order: (1) the saved models are checked on val, (2) export + onnxruntime check, (3) the test, last.
import csv, json, os
import numpy as np, onnx, onnxruntime as ort
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

classes = ["apple", "cherry", "chestnut", "maple"]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


class LeafDataset(Dataset):
    # Copied from 3a, but the (file path, class index) pairs come from splits.csv instead of a folder.
    def __init__(self, samples, transform):
        self.transform = transform
        self.samples = samples

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


def loader(split, source=None):
    samples = [(r["path"], classes.index(r["label"])) for r in csv.DictReader(open("splits.csv"))
               if r["split"] == split and source in (None, r["source"])]
    return DataLoader(LeafDataset(samples, transform), batch_size=32, shuffle=False)


def load_model(name, seed):
    # Same networks as in task4_v2.py (B0: mobilenet_v3_small, C1: efficientnet_b0), weights from the file.
    if name == "B0":
        model = models.mobilenet_v3_small()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    else:
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(torch.load(f"models/v2_final_{name}_seed{seed}.pt", map_location="cpu"))
    return model.eval()


def accuracy(model, loader):
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


# (1) The saved models give the val_last of results_v2.csv (on CPU here, so up to one image may differ).
results = {(r["variant"], int(r["seed"])): float(r["val_last"]) for r in csv.DictReader(open("results_v2.csv"))}
val_loader = loader("val")
for name in ["B0", "C1"]:
    for seed in [0, 1, 2]:
        acc = accuracy(load_model(name, seed), val_loader)
        print(f"val check {name} seed {seed}: {acc:.3f} (results_v2.csv {results[(name, seed)]:.3f})")

# (2) Export C1 seed 0 (chosen before the test) exactly as in cell 2e.
model = load_model("C1", 0)
torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), "leaf_model_final.onnx",
                  input_names=["image"], output_names=["logits"], external_data=False)
model_info = {"classes": classes, "input_size": 96, "mean": mean, "std": std}
onnx_model = onnx.load("leaf_model_final.onnx")
onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
onnx.save(onnx_model, "leaf_model_final.onnx")
print(f"\nleaf_model_final.onnx: {os.path.getsize('leaf_model_final.onnx') / 1e6:.1f} MB")
x = transform(Image.open("check_image.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    p_torch = model(x).softmax(dim=1)[0].numpy()
logits = ort.InferenceSession("leaf_model_final.onnx").run(None, {"image": x.numpy()})[0]
p_onnx = torch.tensor(logits).softmax(dim=1)[0].numpy()
for c, a, b in zip(classes, p_torch, p_onnx):
    print(f"{c:>10}  pytorch {a:.3f}  onnx {b:.3f}")
print("same to 3 decimals:", bool((p_torch.round(3) == p_onnx.round(3)).all()))

# (3) The test split, evaluated once: B0 and C1, seeds 0, 1, 2.
test_loaders = {"test": loader("test"), "test_mine": loader("test", "leaves"), "test_old": loader("test", "old_leaves")}
print("\ntest images:", {k: len(v.dataset) for k, v in test_loaders.items()})
rows = []
for name in ["B0", "C1"]:
    for seed in [0, 1, 2]:
        model = load_model(name, seed)
        rows.append({"variant": name, "seed": seed, **{k: accuracy(model, v) for k, v in test_loaders.items()}})
    for stat, f in [("mean", np.mean), ("std", lambda v: np.std(v, ddof=1))]:
        rows.append({"variant": name, "seed": stat,
                     **{k: f([r[k] for r in rows if r["variant"] == name and r["seed"] in [0, 1, 2]])
                        for k in test_loaders}})
with open("test_results_v2.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "seed", "test", "test_mine", "test_old"])
    writer.writeheader()
    writer.writerows(rows)
print("variant  seed      test  test_mine  test_old")
for r in rows:
    print(f"{r['variant']:7s}  {str(r['seed']):5s}  {r['test']:.3f}  {r['test_mine']:9.3f}  {r['test_old']:8.3f}")
print("one test image = 1/87 ≈ 1.1 %; my part 1/29 ≈ 3.4 %; old part 1/58 ≈ 1.7 %")

# Confusion matrices for seed 0: rows = true class, columns = predicted class.
for name in ["C1", "B0"]:
    model = load_model(name, 0)
    matrix = torch.zeros(4, 4, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in test_loaders["test"]:
            for t, p in zip(labels, model(images).argmax(dim=1)):
                matrix[t, p] += 1
    print(f"\n{name} seed 0 confusion matrix (rows true, columns predicted)")
    print(" " * 10 + "".join(f"{c:>10}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>10}" + "".join(f"{v:>10d}" for v in matrix[i].tolist()))
