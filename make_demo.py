# -*- coding: utf-8 -*-
"""生成演示用的零件表面图片（划痕 / 锈迹 / 缺漆 / 无缺陷）。"""

import os
import random

from PIL import Image, ImageDraw


ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT, "demo")
SIZE = (800, 600)


def metal_base(seed=42):
    rnd = random.Random(seed)
    base = Image.new("RGB", SIZE, (158, 162, 168))
    noise = Image.effect_noise(SIZE, 14).convert("L")
    gray = Image.merge("RGB", (noise, noise, noise))
    img = Image.blend(base, gray, 0.06)
    # 少量细小颗粒，增加金属质感
    draw = ImageDraw.Draw(img)
    for _ in range(900):
        x, y = rnd.randrange(SIZE[0]), rnd.randrange(SIZE[1])
        n = rnd.randint(120, 200)
        draw.point((x, y), fill=(n, n + 2, n + 4))
    return img


def add_scratch(img):
    draw = ImageDraw.Draw(img)
    draw.line([(120, 140), (455, 152)], fill=(42, 45, 50), width=4)
    draw.line([(150, 330), (430, 480)], fill=(49, 52, 57), width=4)
    draw.line([(540, 160), (690, 440)], fill=(45, 48, 53), width=4)
    return img


def add_rust(img, seed=7):
    rnd = random.Random(seed)
    draw = ImageDraw.Draw(img)
    spots = [(210, 170, 48), (610, 430, 60), (430, 290, 28)]
    for cx, cy, r in spots:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(170, 90, 45))
        for _ in range(8):
            ox, oy = rnd.randint(-r // 2, r // 2), rnd.randint(-r // 2, r // 2)
            rr = rnd.randint(r // 3, r // 2)
            draw.ellipse(
                [cx + ox - rr, cy + oy - rr, cx + ox + rr, cy + oy + rr],
                fill=(176, 96, 50),
            )
    return img


def add_paint_loss(img, seed=9):
    rnd = random.Random(seed)
    draw = ImageDraw.Draw(img)
    # 亮色缺漆斑（不规则）
    for cx, cy, r in [(260, 180, 70), (620, 440, 54)]:
        for _ in range(7):
            ox, oy = rnd.randint(-r // 2, r // 2), rnd.randint(-r // 2, r // 2)
            rr = rnd.randint(r // 2, r)
            draw.ellipse(
                [cx + ox - rr, cy + oy - rr, cx + ox + rr, cy + oy + rr],
                fill=(226, 229, 233),
            )
    # 暗色缺漆斑
    for _ in range(6):
        ox, oy = rnd.randint(-40, 40), rnd.randint(-40, 40)
        rr = rnd.randint(26, 48)
        draw.ellipse(
            [430 + ox - rr, 330 + oy - rr, 430 + ox + rr, 330 + oy + rr],
            fill=(55, 58, 62),
        )
    return img


def main():
    os.makedirs(DEMO_DIR, exist_ok=True)
    cases = [
        ("demo_01_金属板_划痕.png", add_scratch),
        ("demo_02_金属板_锈迹.png", add_rust),
        ("demo_03_金属板_缺漆.png", add_paint_loss),
    ]
    for name, fn in cases:
        img = metal_base()
        fn(img)
        img.save(os.path.join(DEMO_DIR, name))
        print("已生成", name)

    clean = metal_base()
    clean.save(os.path.join(DEMO_DIR, "demo_04_金属板_无缺陷.png"))
    print("已生成 demo_04_金属板_无缺陷.png")

    # 额外生成一张 JPEG，验证常见格式可读
    img = metal_base()
    add_rust(img)
    img.save(os.path.join(DEMO_DIR, "demo_02_金属板_锈迹.jpg"), quality=92)
    print("已生成 demo_02_金属板_锈迹.jpg")


if __name__ == "__main__":
    main()
