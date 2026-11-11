# prepare_test.py — raw phone photos data/new_test/<class>/ -> 96x96 JPEGs in data/new_test_96/<class>/
# Same format as the teacher's ZIP: centre square crop, 96x96, JPEG quality 85.
import sys
from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # lets PIL open .heic / .heif files from iPhones

src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/new_test")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "data/new_test_96")

for class_dir in sorted(d for d in src.iterdir() if d.is_dir()):
    (dst / class_dir.name).mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(p for p in class_dir.iterdir() if p.is_file() and not p.name.startswith(".")):
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"  skipped {path}: {e}")
            continue
        img = ImageOps.exif_transpose(img).convert("RGB")  # phone photos store their rotation in EXIF
        # The largest centred square, then 96x96.
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((96, 96), Image.LANCZOS)
        img.save(dst / class_dir.name / f"{class_dir.name}_{count:03d}.jpg", quality=85)
        count += 1
    print(f"{class_dir.name}: {count} images")
