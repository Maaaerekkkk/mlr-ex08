# task4.py — Task 4: one change at a time against the Task 2 baseline (B0)
import copy, csv
from pathlib import Path
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchvision import models, transforms

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Copied unchanged from 3d.
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# V4: aug_transform + color jitter after the rotation.
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


class LeafNetBN(nn.Module):
    # Copied unchanged from 3d: LeafNet with a batch normalization after every convolution, before the ReLU.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 96 -> 48 -> 24 -> 12 -> 6
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


val_ds = LeafDataset("data/leaves/val", transform)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
# V9 uses last year's train images; they must have the same classes. (data/old_leaves/val = test set, not used here.)
assert LeafDataset("data/old_leaves/train", transform).classes == val_ds.classes

# Camera-like val set: every val image transformed 3 times, ONCE, with a fixed seed, reused for every variant.
camera_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
camera_ds = LeafDataset("data/leaves/val", camera_transform)
torch.manual_seed(1234)
pairs = [camera_ds[i] for _ in range(3) for i in range(len(camera_ds))]
camera_loader = DataLoader(TensorDataset(torch.stack([p[0] for p in pairs]),
                                         torch.tensor([p[1] for p in pairs])), batch_size=32)


def make_model(arch, freeze):
    if arch == "leafnetbn":
        return LeafNetBN().to(device)
    model = models.mobilenet_v3_small(weights="DEFAULT")
    if freeze:  # V2: nothing in the pretrained network is trained
        for p in model.parameters():
            p.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)  # new last layer, trainable
    return model.to(device)


def accuracy(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def train_one_variant(seed, arch="mobilenet", train_tf=transform, lr=1e-3, freeze=False, epochs=15,
                      cosine=False, smoothing=0.0, adamw=False, keep_best=False, old_data=False):
    torch.manual_seed(seed)
    model = make_model(arch, freeze)
    train_ds = LeafDataset("data/leaves/train", train_tf)
    if old_data:  # V9: my train + last year's train
        train_ds = ConcatDataset([train_ds, LeafDataset("data/old_leaves/train", train_tf)])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    if adamw:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None

    best_acc, best_epoch, best_state = 0.0, 0, None
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
            best_acc, best_epoch, best_state = val_accuracy, epoch + 1, copy.deepcopy(model.state_dict())
        mean_loss = loss_sum / len(train_ds)
        print(f"epoch {epoch + 1:2d}  training loss {mean_loss:.4f}  validation accuracy {val_accuracy:.3f}")

    if keep_best:  # V8: go back to the weights of the best epoch
        model.load_state_dict(best_state)
    return model, {"val_last": accuracy(model, val_loader), "val_best": best_acc,
                   "best_epoch": best_epoch, "camera_last": accuracy(model, camera_loader)}


variants = {
    "B0": {},
    "V1": {"lr": 1e-4},
    "V2": {"freeze": True},
    "V3": {"train_tf": aug_transform},
    "V4": {"train_tf": color_transform},
    "V5": {"cosine": True, "epochs": 30},
    "V6": {"smoothing": 0.1},
    "V7": {"adamw": True},
    "V8": {"keep_best": True},
    "V9": {"old_data": True},
    "R": {"arch": "leafnetbn", "train_tf": aug_transform},
    "V5b": {"cosine": True},
    "C1": {"train_tf": color_transform, "cosine": True, "epochs": 30},
    "C2": {"train_tf": aug_transform, "cosine": True, "epochs": 30},
}

# Variants already in results.csv are not trained again: their rows are read back, new rows are appended.
rows = []
if Path("results.csv").exists():
    with open("results.csv") as f:
        rows = [{k: (v if k == "variant" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
done = {r["variant"] for r in rows}

new_rows = []
for name, changes in variants.items():
    if name in done:
        continue
    for seed in [0, 1, 2]:
        print(f"--- {name} seed {seed}")
        model, result = train_one_variant(seed, **changes)
        print(result)
        new_rows.append({"variant": name, "seed": seed, **result})
rows += new_rows

with open("results.csv", "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "seed", "val_last", "val_best", "best_epoch", "camera_last"])
    if not done:
        writer.writeheader()
    writer.writerows(new_rows)

# Summary: mean ± std (sample std) over the 3 seeds.
columns = ["val_last", "val_best", "best_epoch", "camera_last"]
print("\nvariant  " + "  ".join(f"{c:>15}" for c in columns))
for name in variants:
    values = [[r[c] for r in rows if r["variant"] == name] for c in columns]
    cells = [f"{torch.tensor(v, dtype=torch.float).mean():.3f} ± {torch.tensor(v, dtype=torch.float).std():.3f}"
             for v in values]
    print(f"{name:7s}  " + "  ".join(f"{c:>15}" for c in cells))
print("Note: one val image = 1/62 ≈ 1.6 %; one camera-like image = 1/186 ≈ 0.5 % (3 versions of the same 62 leaves).")

# Final models: B0 and the chosen C1 retrained with the same code and saved for test_final.py.
# V9 added after the final test (post-hoc, exploratory).
Path("models").mkdir(exist_ok=True)
for name in ["B0", "C1", "V9"]:
    for seed in [0, 1, 2]:
        if Path(f"models/{name}_seed{seed}.pt").exists():
            continue
        print(f"--- final {name} seed {seed}")
        model, result = train_one_variant(seed, **variants[name])
        print(result)
        torch.save(model.state_dict(), f"models/{name}_seed{seed}.pt")
