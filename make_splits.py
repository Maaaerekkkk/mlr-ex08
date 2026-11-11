# make_splits.py — writes splits.csv (path, label, source, split)
# train = my train + old train; the teacher's two val folders are split randomly into val and test
# (half / half, stratified by source and class, seed 0). Exact duplicates (MD5 of decoded RGB pixels) removed.
import csv, glob, hashlib, random, collections
from PIL import Image

classes = ["apple", "cherry", "chestnut", "maple"]
excluded = ["data/leaves/val/apple/apple_007.jpg"]  # a branch with several leaves, not one leaf
label_fixes = {}  # path -> correct label, filled in after the label check

# All images, train folders first, so that the train copy of a duplicate is the one that is kept.
order = {"train": 0, "val": 1}
files = sorted(glob.glob("data/leaves/*/*/*.jpg") + glob.glob("data/old_leaves/*/*/*.jpg"),
               key=lambda f: (order[f.split("/")[2]], f))

# Exact duplicates: keep the first copy; if the copies have different labels, drop all of them.
groups = collections.defaultdict(list)
for f in files:
    groups[hashlib.md5(Image.open(f).convert("RGB").tobytes()).hexdigest()].append(f)
for g in groups.values():
    if len(g) > 1:
        excluded += g if len({f.split("/")[3] for f in g}) > 1 else g[1:]
print("excluded:", len(excluded))

rows = []
for f in files:
    if f not in excluded:
        _, source, folder, label, _ = f.split("/")
        rows.append({"path": f, "label": label_fixes.get(f, label), "source": source, "split": folder})

# The teacher's val folders -> val / test, half each, per (source, class); an odd extra image goes to val.
rng = random.Random(0)
for source in ["leaves", "old_leaves"]:
    for label in classes:
        part = [r for r in rows if r["split"] == "val" and r["source"] == source and r["path"].split("/")[3] == label]
        rng.shuffle(part)
        for r in part[: len(part) // 2]:
            r["split"] = "test"

with open("splits.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
    writer.writeheader()
    writer.writerows(rows)

print("split  source       apple cherry chestnut maple  total")
for split in ["train", "val", "test"]:
    for source in ["leaves", "old_leaves", "ALL"]:
        r = [x for x in rows if x["split"] == split and source in (x["source"], "ALL")]
        c = collections.Counter(x["label"] for x in r)
        print(f"{split:6s} {source:11s} {c['apple']:6d} {c['cherry']:6d} {c['chestnut']:8d} {c['maple']:5d} {len(r):6d}")
