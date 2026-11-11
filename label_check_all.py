# label_check_all.py — cross-fitted label check over train + val of splits.csv (the test split is never used).
# The pool is split randomly into two halves (seed 0); B0 is trained on one half and predicts the other,
# and vice versa. Images misclassified with p > 0.9 go to labels_check_all.png. No labels are changed.
import csv, random
from pathlib import Path
import torch, torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

device = "mps" if torch.backends.mps.is_available() else "cpu"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
classes = ["apple", "cherry", "chestnut", "maple"]


class LeafDataset(Dataset):
    # Copied from 3a, but the (file path, class index) pairs are given instead of read from a folder.
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


# The pool: train + val of splits.csv (exclusions already out). Test images are not used at all.
pool = [(r["path"], classes.index(r["label"])) for r in csv.DictReader(open("splits.csv"))
        if r["split"] in ["train", "val"]]
test_paths = {r["path"] for r in csv.DictReader(open("splits.csv")) if r["split"] == "test"}
assert not test_paths & {path for path, _ in pool}
random.Random(0).shuffle(pool)
halves = [pool[: len(pool) // 2], pool[len(pool) // 2:]]
print("pool:", len(pool), "| halves:", len(halves[0]), len(halves[1]))


def train_b0(samples):
    # B0: pretrained mobilenet_v3_small, whole network, Adam 1e-3, 15 epochs, no augmentation, batch 32.
    torch.manual_seed(0)
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    model = model.to(device)
    train_loader = DataLoader(LeafDataset(samples, transform), batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(0))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(15):
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
        print(f"epoch {epoch + 1:2d}  training loss {loss_sum / len(samples):.4f}")
    return model.eval()


hits = []
for k in [0, 1]:
    model = train_b0(halves[k])
    other = halves[1 - k]
    correct = 0
    with torch.no_grad():
        for path, label in other:
            p = model(transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)).softmax(dim=1)[0].cpu()
            pred = p.argmax().item()
            correct += pred == label
            if pred != label and p[pred].item() > 0.9:
                hits.append((path, classes[label], classes[pred], p[pred].item()))
    print(f"trained on half {k}, accuracy on half {1 - k}: {correct / len(other):.3f}")

hits.sort()
print(f"\n{len(hits)} images misclassified with p > 0.9")
for path, label, pred, prob in hits:
    print(f"  {path:45s} label {label:9s} pred {pred:9s} p {prob:.3f}")

# Contact sheet: image, path, label, prediction, probability.
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
cols, tw, th = 6, 220, 270
sheet = Image.new("RGB", (cols * tw, max(1, (len(hits) + cols - 1) // cols) * th), "white")
draw = ImageDraw.Draw(sheet)
for i, (path, label, pred, prob) in enumerate(hits):
    x, y = (i % cols) * tw, (i // cols) * th
    sheet.paste(Image.open(path).convert("RGB").resize((192, 192), Image.NEAREST), (x + 14, y + 8))
    draw.text((x + 6, y + 204), path.replace("data/", ""), fill="black", font=font)
    draw.text((x + 6, y + 222), f"label: {label}", fill="black", font=font)
    draw.text((x + 6, y + 240), f"pred: {pred}  p={prob:.3f}", fill="red", font=font)
sheet.save("labels_check_all.png")
