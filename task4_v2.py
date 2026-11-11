# task4_v2.py — Task 4, round 2: one change at a time against B0, on the splits of splits.csv.
# The test split is NOT used here.
import copy, csv, os
import numpy as np, onnxruntime as ort
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

classes = ["apple", "cherry", "chestnut", "maple"]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# A1: my aug_transform from 3d + color jitter after the rotation.
color_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


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


rows = list(csv.DictReader(open("splits.csv")))
train_samples = [(r["path"], classes.index(r["label"])) for r in rows if r["split"] == "train"]
val_rows = [r for r in rows if r["split"] == "val"]
val_samples = [(r["path"], classes.index(r["label"])) for r in val_rows]
val_loader = DataLoader(LeafDataset(val_samples, transform), batch_size=32, shuffle=False)
# The two parts of val: my group and last years.
mine_loader = DataLoader(LeafDataset([(r["path"], classes.index(r["label"])) for r in val_rows
                                      if r["source"] == "leaves"], transform), batch_size=32)
old_loader = DataLoader(LeafDataset([(r["path"], classes.index(r["label"])) for r in val_rows
                                     if r["source"] == "old_leaves"], transform), batch_size=32)
print("train", len(train_samples), "| val", len(val_samples), "| my val", len(mine_loader.dataset),
      "| old val", len(old_loader.dataset))

# Camera-like val set: every val image transformed 3 times, ONCE, with a fixed seed, reused for every variant.
camera_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
camera_ds = LeafDataset(val_samples, camera_transform)
torch.manual_seed(1234)
pairs = [camera_ds[i] for _ in range(3) for i in range(len(camera_ds))]
camera_loader = DataLoader(TensorDataset(torch.stack([p[0] for p in pairs]),
                                         torch.tensor([p[1] for p in pairs])), batch_size=32)


def make_model(arch, upsample):
    # Pretrained backbone, its last layer replaced by 4 outputs.
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights="DEFAULT")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    elif arch == "resnet18":
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, 4)
    if upsample:  # A3: the network sees 224x224, the input stays 96x96
        model = nn.Sequential(nn.Upsample(size=224, mode="bilinear"), model)
    return model.to(device)


def accuracy(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def train_one_variant(seed, arch="mobilenet_v3_small", upsample=False, train_tf=transform, lr=1e-3,
                      epochs=15, cosine=False):
    torch.manual_seed(seed)
    model = make_model(arch, upsample)
    train_ds = LeafDataset(train_samples, train_tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None

    best_acc, best_epoch = 0.0, 0
    for epoch in range(epochs):
        # The training part of 3c, unchanged.
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        if scheduler:
            scheduler.step()

        val_accuracy = accuracy(model, val_loader)
        if val_accuracy > best_acc:
            best_acc, best_epoch = val_accuracy, epoch + 1
        mean_loss = loss_sum / len(train_ds)
        print(f"epoch {epoch + 1:2d}  training loss {mean_loss:.4f}  validation accuracy {val_accuracy:.3f}")

    return model, {"val_last": accuracy(model, val_loader), "val_best": best_acc, "best_epoch": best_epoch,
                   "camera_last": accuracy(model, camera_loader),
                   "val_mine": accuracy(model, mine_loader), "val_old": accuracy(model, old_loader)}


variants = {
    "B0": {},
    "A1": {"train_tf": color_transform},
    "A2": {"cosine": True, "epochs": 30},
    "A3": {"upsample": True},
    "A4": {"arch": "mobilenet_v3_large"},
    "A5": {"arch": "efficientnet_b0"},
    "A6": {"arch": "resnet18"},
    "A7": {"lr": 1e-4},
    "C1": {"arch": "efficientnet_b0", "train_tf": color_transform, "cosine": True, "epochs": 30},
    "C2": {"arch": "efficientnet_b0", "cosine": True, "epochs": 30},
}
backbones = ["B0", "A3", "A4", "A5", "A6"]  # one per network: parameters and ONNX export (seed 0)

# Variants already in results_v2.csv are not trained again: their rows are read back, new rows are appended.
rows_out, backbone_info = [], []
if os.path.exists("results_v2.csv"):
    with open("results_v2.csv") as f:
        rows_out = [{k: (v if k == "variant" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
done = {r["variant"] for r in rows_out}
new_rows = []

for name, changes in variants.items():
    if name in done:
        continue
    for seed in [0, 1, 2]:
        print(f"--- {name} seed {seed}")
        model, result = train_one_variant(seed, **changes)
        print(result)
        new_rows.append({"variant": name, "seed": seed, **result})

        if seed == 0 and name in backbones:
            # ONNX export as in cell 2e (input "image", output "logits"), checked with onnxruntime on val images.
            model.eval().cpu()
            path = f"models/v2_{name}.onnx"
            os.makedirs("models", exist_ok=True)
            torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), path,
                              input_names=["image"], output_names=["logits"], external_data=False)
            images = torch.stack([LeafDataset(val_samples, transform)[i][0] for i in range(8)])
            session = ort.InferenceSession(path)
            diffs = []
            for x in images:
                with torch.no_grad():
                    p_torch = model(x.unsqueeze(0)).softmax(dim=1).numpy()
                logits = session.run(None, {"image": x.unsqueeze(0).numpy()})[0]
                p_onnx = torch.tensor(logits).softmax(dim=1).numpy()
                diffs.append(np.abs(p_torch - p_onnx).max())
            backbone_info.append({"variant": name, "network": changes.get("arch", "mobilenet_v3_small")
                                  + (" + upsample 224" if changes.get("upsample") else ""),
                                  "parameters": sum(p.numel() for p in model.parameters()),
                                  "onnx_MB": os.path.getsize(path) / 1e6, "onnx_max_prob_diff": max(diffs)})

rows_out += new_rows
with open("results_v2.csv", "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "seed", "val_last", "val_best", "best_epoch",
                                           "camera_last", "val_mine", "val_old"])
    if not done:
        writer.writeheader()
    writer.writerows(new_rows)

# Summary: mean ± std (sample std) over the 3 seeds.
columns = ["val_last", "val_best", "best_epoch", "camera_last", "val_mine", "val_old"]
print("\nvariant  " + "  ".join(f"{c:>15}" for c in columns))
for name in variants:
    cells = []
    for c in columns:
        v = torch.tensor([r[c] for r in rows_out if r["variant"] == name], dtype=torch.float)
        cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
    print(f"{name:7s}  " + "  ".join(f"{c:>15}" for c in cells))
print("Note: one val image = 1/90 ≈ 1.1 %; my val 1/32 ≈ 3.1 %; old val 1/58 ≈ 1.7 %; "
      "camera-like 1/270 ≈ 0.4 % (3 versions of the same 90 leaves).")

# Selection rule (fixed before the C1/C2 results): the higher mean of (val_last + camera_last) / 2 wins.
print("\nscore = mean over seeds of (val_last + camera_last) / 2")
for name in ["B0", "A5", "C1", "C2"]:
    v = torch.tensor([(r["val_last"] + r["camera_last"]) / 2 for r in rows_out if r["variant"] == name])
    print(f"{name:7s}  {v.mean():.3f} ± {v.std():.3f}")

if backbone_info:
    print("\nvariant  network                            parameters  ONNX MB  ONNX max prob diff")
for b in backbone_info:
    print(f"{b['variant']:7s}  {b['network']:33s} {b['parameters']:11,d}  {b['onnx_MB']:7.1f}  {b['onnx_max_prob_diff']:.2e}")

# Phase C: B0 and the chosen C1 retrained with the same code and saved for test_final_v2.py.
for name in ["B0", "C1"]:
    for seed in [0, 1, 2]:
        path = f"models/v2_final_{name}_seed{seed}.pt"
        if os.path.exists(path):
            continue
        print(f"--- final {name} seed {seed}")
        model, result = train_one_variant(seed, **variants[name])
        print(result)
        old = [r for r in rows_out if r["variant"] == name and r["seed"] == seed][0]
        print("val_last and camera_last match results_v2.csv:",
              result["val_last"] == old["val_last"] and result["camera_last"] == old["camera_last"])
        torch.save(model.state_dict(), path)
