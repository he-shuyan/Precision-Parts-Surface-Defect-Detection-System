# -*- coding: utf-8 -*-
"""生成合成零件表面缺陷数据集（用于训练 YOLO 检测模型）。

由于项目初期没有真实缺陷样本，这里用程序合成"精密零件表面"图像：
  划痕(scratch) / 锈迹(rust) / 缺漆(paint_loss)
并自动输出 YOLO 格式标注，目录结构：
    dataset/images/train, dataset/images/val
    dataset/labels/train, dataset/labels/val
    dataset/data.yaml
"""

import math
import os
import random

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "dataset")
SIZE = 640
CLASSES = ["scratch", "rust", "paint_loss"]
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}


def _composite(base, overlay, blur=0.0):
    if blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(blur))
    canvas = base.convert("RGBA")
    canvas.alpha_composite(overlay)
    return canvas.convert("RGB")


def make_background(rng):
    base = rng.randint(105, 195)
    img = Image.new("RGB", (SIZE, SIZE), (base, base, base))

    # 拉丝纹理
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(80, 170)):
        y = rng.randrange(SIZE)
        x0 = rng.randrange(-120, SIZE)
        x1 = x0 + rng.randint(90, 460)
        v = max(6, min(249, base + rng.randint(-16, 16)))
        draw.line([(x0, y), (x1, y)], fill=(v, v, v), width=1)

    # 光照渐变
    small = Image.new("L", (2, 2))
    for yy in range(2):
        for xx in range(2):
            small.putpixel((xx, yy), rng.randint(70, 200))
    grad = small.resize((SIZE, SIZE), Image.BILINEAR)
    img = Image.blend(img, Image.merge("RGB", (grad, grad, grad)), rng.uniform(0.15, 0.35))

    # 颗粒噪声
    noise = Image.effect_noise((SIZE, SIZE), rng.uniform(6, 15)).convert("L")
    img = Image.blend(img, Image.merge("RGB", (noise, noise, noise)), rng.uniform(0.06, 0.14))
    return img


def _sample_gray(img, x, y):
    x = max(0, min(SIZE - 1, int(x)))
    y = max(0, min(SIZE - 1, int(y)))
    r, g, b = img.getpixel((x, y))
    return (r + g + b) / 3.0


def draw_scratch(img, rng):
    x = rng.uniform(70, SIZE - 70)
    y = rng.uniform(70, SIZE - 70)
    angle = rng.uniform(0, 2 * math.pi)
    segs = rng.randint(1, 4)
    total = rng.randint(70, 330)
    width = rng.randint(1, 4)
    delta = rng.choice([-1, 1]) * rng.randint(40, 95)

    pts = [(x, y)]
    step = total / segs
    for _ in range(segs):
        angle += rng.uniform(-0.28, 0.28)
        x += step * math.cos(angle)
        y += step * math.sin(angle)
        pts.append((x, y))

    base_gray = _sample_gray(img, pts[len(pts) // 2][0], pts[len(pts) // 2][1])
    value = int(max(6, min(249, base_gray + delta)))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).line(pts, fill=(value, value, value, 255), width=width, joint="curve")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = width + 3
    box = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    return _composite(img, overlay, blur=0.5), box


def draw_rust(img, rng):
    cx = rng.uniform(80, SIZE - 80)
    cy = rng.uniform(80, SIZE - 80)
    radius = rng.randint(24, 72)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(rng.randint(5, 13)):
        ox = rng.randint(-radius // 2, radius // 2)
        oy = rng.randint(-radius // 2, radius // 2)
        rr = rng.randint(radius // 3, radius)
        draw.ellipse(
            [cx + ox - rr, cy + oy - rr, cx + ox + rr, cy + oy + rr],
            fill=(
                rng.randint(140, 196),
                rng.randint(58, 110),
                rng.randint(18, 62),
                255,
            ),
        )
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    return _composite(img, overlay, blur=rng.uniform(0.4, 1.1)), box


def draw_paint_loss(img, rng):
    cx = rng.uniform(80, SIZE - 80)
    cy = rng.uniform(80, SIZE - 80)
    radius = rng.randint(26, 78)
    base_gray = _sample_gray(img, cx, cy)
    delta = rng.choice([-1, 1]) * rng.randint(45, 88)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(rng.randint(6, 14)):
        ox = rng.randint(-radius // 2, radius // 2)
        oy = rng.randint(-radius // 2, radius // 2)
        rr = rng.randint(radius // 2, int(radius * 1.1))
        v = int(max(8, min(247, base_gray + delta + rng.randint(-18, 18))))
        draw.ellipse(
            [cx + ox - rr, cy + oy - rr, cx + ox + rr, cy + oy + rr],
            fill=(v, v, v, 255),
        )
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    return _composite(img, overlay, blur=rng.uniform(0.5, 1.3)), box


def clip_box(box):
    x0, y0, x1, y1 = box
    x0 = max(0, min(SIZE - 1, x0))
    y0 = max(0, min(SIZE - 1, y0))
    x1 = max(0, min(SIZE, x1))
    y1 = max(0, min(SIZE, y1))
    if x1 - x0 < 22 or y1 - y0 < 22:
        return None
    return x0, y0, x1, y1


def make_sample(rng):
    img = make_background(rng)
    n = rng.choices([0, 1, 2, 3, 4], weights=[8, 30, 30, 20, 12])[0]
    labels = []
    for _ in range(n):
        kind = rng.choice(CLASSES)
        if kind == "scratch":
            img, box = draw_scratch(img, rng)
        elif kind == "rust":
            img, box = draw_rust(img, rng)
        else:
            img, box = draw_paint_loss(img, rng)
        box = clip_box(box)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2 / SIZE
        cy = (y0 + y1) / 2 / SIZE
        w = (x1 - x0) / SIZE
        h = (y1 - y0) / SIZE
        labels.append(f"{CLASS_IDS[kind]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # 全局轻度增强，模拟真实成像差异
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.18))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.82, 1.25))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    return img, labels


def main(total=1100, val_ratio=0.15, seed=2026):
    rng = random.Random(seed)
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATA_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "labels", split), exist_ok=True)

    n_val = int(total * val_ratio)
    counts = {"train": 0, "val": 0}
    class_counts = {c: 0 for c in CLASSES}
    for i in range(total):
        img, labels = make_sample(rng)
        split = "val" if i < n_val else "train"
        name = f"{split}_{i:05d}"
        img.save(os.path.join(DATA_DIR, "images", split, name + ".jpg"), quality=rng.randint(78, 96))
        with open(os.path.join(DATA_DIR, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(labels))
        counts[split] += 1
        for line in labels:
            class_counts[CLASSES[int(line.split()[0])]] += 1

    with open(os.path.join(DATA_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {DATA_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 3\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")

    print("数据集生成完成：", counts)
    print("各类缺陷框数量：", class_counts)


if __name__ == "__main__":
    main()
