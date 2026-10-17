# -*- coding: utf-8 -*-
"""精密零件表面缺陷检测 —— 检测引擎。

v1 版本使用经典图像处理算法识别三类常见缺陷：
  划痕、锈迹、缺漆/斑点。

检测引擎采用可插拔接口：默认使用 ClassicalEngine（无需模型、开箱即用），
后续拿到训练好的 YOLO/ONNX 模型后，可在 ModelEngine 中接入，接口保持不变。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import List

from PIL import Image, ImageChops, ImageDraw, ImageOps


ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PT = os.path.join(ROOT, "models", "defect_best.pt")
MODEL_ONNX = os.path.join(ROOT, "models", "defect_best.onnx")


# 缺陷类型与显示颜色（供前端标注使用）
DEFECT_TYPES = {
    "scratch": {"label": "划痕", "color": "#e53935"},
    "rust": {"label": "锈迹", "color": "#fb8c00"},
    "paint_loss": {"label": "缺漆/斑点", "color": "#1e88e5"},
    "stain": {"label": "污渍", "color": "#fdd835"},
}


@dataclass
class Defect:
    kind: str
    label: str
    x: int
    y: int
    w: int
    h: int
    confidence: float
    area: int

    def to_dict(self) -> dict:
        return asdict(self)


class BaseEngine:
    """检测引擎基类。"""

    name = "base"

    def detect(self, image: Image.Image) -> List[Defect]:
        raise NotImplementedError


class ModelEngine(BaseEngine):
    """预留的深度学习模型接口。

    在已安装 torch + ultralytics 且提供 .pt/.onnx 模型文件时，
    用真实模型替换默认的经典算法引擎。
    """

    name = "model"

    def __init__(self, model_path: str):
        self.model_path = model_path

    def detect(self, image: Image.Image) -> List[Defect]:
        raise NotImplementedError(
            "当前环境未接入深度学习模型。系统默认使用经典图像处理引擎，"
            "识别划痕、锈迹、缺漆/斑点三类缺陷。"
        )


class YoloEngine(BaseEngine):
    """基于 YOLOv8 的深度学习检测引擎（本项目训练所得）。"""

    name = "yolov8"

    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        from ultralytics import YOLO

        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self._model = YOLO(weights)

    def detect(self, image: Image.Image) -> List[Defect]:
        results = self._model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        defects: List[Defect] = []
        for res in results:
            names = res.names
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                kind = names.get(cls_id, str(cls_id))
                if kind not in DEFECT_TYPES:
                    kind = "stain"
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                w = max(1, int(round(x2 - x1)))
                h = max(1, int(round(y2 - y1)))
                defects.append(
                    Defect(
                        kind=kind,
                        label=DEFECT_TYPES[kind]["label"],
                        x=int(round(x1)),
                        y=int(round(y1)),
                        w=w,
                        h=h,
                        confidence=round(float(box.conf[0]), 2),
                        area=w * h,
                    )
                )
        defects.sort(key=lambda d: (d.y, d.x))
        return defects


def _letterbox(ndarray, image, imgsz):
    """等比缩放 + 灰边(114)填充到 imgsz×imgsz，与 ultralytics 的预处理一致。

    返回 (NCHW float32 张量, 缩放比, 左边距, 上边距)。
    """
    width, height = image.size
    scale = min(imgsz / float(width), imgsz / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    array = ndarray.asarray(canvas, dtype=ndarray.float32) / 255.0
    array = ndarray.transpose(array, (2, 0, 1))[None, ...]
    return ndarray.ascontiguousarray(array), scale, pad_x, pad_y


def _iou_matrix(ndarray, box, boxes):
    """一个框 (x1,y1,x2,y2) 与一组框的交并比。"""
    x1 = ndarray.maximum(box[0], boxes[:, 0])
    y1 = ndarray.maximum(box[1], boxes[:, 1])
    x2 = ndarray.minimum(box[2], boxes[:, 2])
    y2 = ndarray.minimum(box[3], boxes[:, 3])
    inter = ndarray.maximum(0.0, x2 - x1) * ndarray.maximum(0.0, y2 - y1)
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter)
    return inter / ndarray.maximum(union, 1e-9)


class OnnxEngine(BaseEngine):
    """基于 ONNX Runtime 的 YOLOv8 推理引擎（不依赖 PyTorch）。

    与 YoloEngine 输出完全一致（同样的 Defect 列表、同样的类别与坐标口径），
    区别只在运行方式：
      * 只需要 onnxruntime，不需要 torch —— 打包体积小、启动更快；
      * 不会把 PyQt5 的 Qt5\\bin 与 PyTorch 的 MSVC 运行库搅在一起，
        桌面端启动更稳。
    """

    name = "yolov8-onnx"

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45,
                 imgsz: int = 640, providers=None):
        import numpy
        import onnxruntime as ort

        self._np = numpy
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        available = ort.get_available_providers()
        if providers is None:
            providers = [name for name in ("CUDAExecutionProvider",
                                           "CPUExecutionProvider")
                         if name in available] or ["CPUExecutionProvider"]
        self.providers = providers
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.names = self._read_names()

    def _read_names(self) -> dict:
        """类别名优先取 ONNX 元数据里写好的 names，取不到就按三类的固定顺序。"""
        default = {index: kind for index, kind in enumerate(
            [k for k in ("scratch", "rust", "paint_loss")])}
        try:
            import ast
            raw = self._session.get_modelmeta().custom_metadata_map.get("names")
            parsed = ast.literal_eval(raw) if raw else None
            if isinstance(parsed, dict) and parsed:
                return {int(k): str(v) for k, v in parsed.items()}
        except Exception:  # noqa: BLE001  元数据异常不影响推理
            pass
        return default

    def detect(self, image: Image.Image) -> List[Defect]:
        rgb = image.convert("RGB")
        width, height = rgb.size
        tensor, scale, pad_x, pad_y = _letterbox(self._np, rgb, self.imgsz)
        output = self._session.run(None, {self._input_name: tensor})[0]
        return self._to_defects(output[0], scale, pad_x, pad_y, width, height)

    def _to_defects(self, prediction, scale, pad_x, pad_y, width, height):
        """把 (4+类别数, 候选框数) 的原始输出解码成 Defect 列表。"""
        np = self._np
        pred = prediction.T                      # (候选框数, 4+类别数)
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confs = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = confs >= self.conf
        if not keep.any():
            return []
        boxes = boxes_xywh[keep]
        confs = confs[keep]
        class_ids = class_ids[keep]

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        # 先在 letterbox 坐标系里做逐类 NMS（与 ultralytics 的顺序一致：
        # 它也是先 NMS、再把框映射回原图并裁剪），最后才换算坐标。
        picked = []
        for class_id in np.unique(class_ids):
            index = np.where(class_ids == class_id)[0]
            order = index[np.argsort(-confs[index])]
            while order.size:
                best = order[0]
                picked.append(best)
                if order.size == 1:
                    break
                rest = order[1:]
                order = rest[_iou_matrix(np, xyxy[best], xyxy[rest]) <= self.iou]

        defects: List[Defect] = []
        for index in picked:
            x1, y1, x2, y2 = [float(v) for v in xyxy[index]]
            # 反 letterbox：去边距、除缩放比，再裁到原图范围
            x1 = min(max((x1 - pad_x) / scale, 0.0), float(width))
            y1 = min(max((y1 - pad_y) / scale, 0.0), float(height))
            x2 = min(max((x2 - pad_x) / scale, 0.0), float(width))
            y2 = min(max((y2 - pad_y) / scale, 0.0), float(height))
            box_w = max(1, int(round(x2 - x1)))
            box_h = max(1, int(round(y2 - y1)))
            kind = self.names.get(int(class_ids[index]), "stain")
            if kind not in DEFECT_TYPES:
                kind = "stain"
            defects.append(Defect(
                kind=kind,
                label=DEFECT_TYPES[kind]["label"],
                x=int(round(x1)), y=int(round(y1)),
                w=box_w, h=box_h,
                confidence=round(float(confs[index]), 2),
                area=box_w * box_h))
        defects.sort(key=lambda d: (d.y, d.x))
        return defects


def _components(data: List[int], w: int, h: int, min_area: int):
    """在二值掩码（0/255）上提取连通域，返回 (bbox, area, pixel_indices)。"""
    visited = bytearray(w * h)
    comps = []
    for y in range(h):
        row_base = y * w
        for x in range(w):
            idx = row_base + x
            if visited[idx] or data[idx] == 0:
                continue
            # 广度优先搜索
            stack = [idx]
            visited[idx] = 1
            area = 0
            minx = maxx = x
            miny = maxy = y
            pixels = []
            while stack:
                cur = stack.pop()
                area += 1
                cx = cur % w
                cy = cur // w
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                if area <= 6000:
                    pixels.append(cur)

                # 上下左右四个邻域
                if cx + 1 < w and not visited[cur + 1] and data[cur + 1]:
                    visited[cur + 1] = 1
                    stack.append(cur + 1)
                if cx - 1 >= 0 and not visited[cur - 1] and data[cur - 1]:
                    visited[cur - 1] = 1
                    stack.append(cur - 1)
                if cy + 1 < h and not visited[cur + w] and data[cur + w]:
                    visited[cur + w] = 1
                    stack.append(cur + w)
                if cy - 1 >= 0 and not visited[cur - w] and data[cur - w]:
                    visited[cur - w] = 1
                    stack.append(cur - w)

            if area >= min_area:
                bw = maxx - minx + 1
                bh = maxy - miny + 1
                comps.append((minx, miny, bw, bh, area, pixels))
    return comps


def _line_ratio(pixels: List[int], w: int) -> float:
    """根据像素坐标的协方差特征值比，衡量连通域是否为细长线条。

    返回值越大，说明该区域越接近一条线（与线条方向无关）。
    """
    n = len(pixels)
    if n < 8:
        return 1.0
    sx = 0
    sy = 0
    for i in pixels:
        sx += i % w
        sy += i // w
    mx = sx / n
    my = sy / n
    sxx = syy = sxy = 0.0
    for i in pixels:
        dx = (i % w) - mx
        dy = (i // w) - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(0.0, tr * tr - 4.0 * det) ** 0.5
    l1 = (tr + disc) / 2.0
    l2 = max(1e-6, (tr - disc) / 2.0)
    return l1 / l2


def _iou(a, b):
    """两个框的交并比。a/b 均为 (x, y, w, h)。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


class ClassicalEngine(BaseEngine):
    """经典图像处理引擎（无模型依赖）。"""

    name = "classical"

    def __init__(self, max_side: int = 1024):
        self.max_side = max_side

    def detect(self, image: Image.Image) -> List[Defect]:
        rgb = image.convert("RGB")
        orig_w, orig_h = rgb.size

        # 大图缩小后再分析，结果坐标再映射回原图，保证速度
        work = rgb.copy()
        if max(work.size) > self.max_side:
            work.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        scale = orig_w / work.width
        w, h = work.size

        gray = ImageOps.grayscale(work)
        gray_data = list(gray.getdata())

        # 锈迹：橙色/棕色区域（色相 + 饱和度 + 亮度共同约束）
        hsv = work.convert("HSV")
        h_ch, s_ch, v_ch = hsv.split()
        rust_mask = ImageChops.multiply(
            ImageChops.multiply(
                h_ch.point(lambda p: 255 if 8 <= p <= 35 else 0),
                s_ch.point(lambda p: 255 if p >= 45 else 0),
            ),
            v_ch.point(lambda p: 255 if 45 <= p <= 235 else 0),
        )

        # 亮度异常区域：过暗 / 过亮（用于划痕与缺漆/斑点）
        dark = gray.point(lambda p: 255 if p < 90 else 0)
        bright = gray.point(lambda p: 255 if p > 215 else 0)
        anomaly_mask = ImageChops.lighter(dark, bright)

        rust_data = list(rust_mask.getdata())
        anomaly_data = list(anomaly_mask.getdata())
        h_data = list(h_ch.getdata())
        s_data = list(s_ch.getdata())
        v_data = list(v_ch.getdata())

        min_area = max(16, int(w * h * 0.00002))
        defects: List[Defect] = []

        # 锈迹
        for (x, y, bw, bh, area, pixels) in _components(
            rust_data, w, h, min_area
        ):
            if bw < 6 or bh < 6:
                continue
            if bw > 0.6 * w and bh > 0.6 * h:
                continue
            sample = pixels if len(pixels) <= 1000 else pixels[:: max(1, len(pixels) // 1000)]
            if not sample:
                continue
            mean_s = sum(s_data[i] for i in sample) / len(sample)
            conf = min(0.93, 0.62 + 0.30 * min(mean_s, 140) / 140)
            if area > 800:
                conf = min(0.96, conf + 0.03)
            defects.append(
                Defect("rust", DEFECT_TYPES["rust"]["label"],
                       int(x * scale), int(y * scale),
                       int(bw * scale), int(bh * scale),
                       round(conf, 2), int(area * scale * scale))
            )

        # 划痕 / 缺漆（亮度异常连通域按形状区分）
        for (x, y, bw, bh, area, pixels) in _components(
            anomaly_data, w, h, min_area
        ):
            short = min(bw, bh)
            long = max(bw, bh)
            if long < 18:
                continue
            if bw > 0.6 * w and bh > 0.6 * h:
                continue
            sample = pixels if len(pixels) <= 1000 else pixels[:: max(1, len(pixels) // 1000)]
            if not sample:
                continue
            mean_s = sum(s_data[i] for i in sample) / len(sample)
            mean_v = sum(v_data[i] for i in sample) / len(sample)
            line_ratio = _line_ratio(pixels, w)

            if line_ratio >= 25.0 and long >= 24 and area <= 30000:
                conf = min(0.95, 0.58 + 0.30 * min(line_ratio, 400) / 400)
                defects.append(
                    Defect("scratch", DEFECT_TYPES["scratch"]["label"],
                           int(x * scale), int(y * scale),
                           int(bw * scale), int(bh * scale),
                           round(conf, 2), int(area * scale * scale))
                )
            elif mean_s < 70 and (mean_v > 190 or mean_v < 110):
                conf = min(0.92, 0.58 + 0.25 * min(area, 2000) / 2000)
                defects.append(
                    Defect("paint_loss", DEFECT_TYPES["paint_loss"]["label"],
                           int(x * scale), int(y * scale),
                           int(bw * scale), int(bh * scale),
                           round(conf, 2), int(area * scale * scale))
                )
            else:
                defects.append(
                    Defect("stain", DEFECT_TYPES["stain"]["label"],
                           int(x * scale), int(y * scale),
                           int(bw * scale), int(bh * scale),
                           round(min(0.65, 0.45 + 0.01 * min(area, 2000) / 100), 2),
                           int(area * scale * scale))
                )

        # 非极大值抑制：去掉高度重叠的重复框
        defects.sort(key=lambda d: d.area, reverse=True)
        keep: List[Defect] = []
        for d in defects:
            if not any(
                _iou((d.x, d.y, d.w, d.h), (k.x, k.y, k.w, k.h)) > 0.45
                for k in keep
            ):
                keep.append(d)
        keep.sort(key=lambda d: (d.y, d.x))
        return keep


def annotate(image: Image.Image, defects: List[Defect]) -> Image.Image:
    """在原图上画出缺陷框，返回标注后的 RGB 图像。"""
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    for d in defects:
        color = DEFECT_TYPES.get(d.kind, {}).get("color", "#ffffff")
        for t in range(3):
            draw.rectangle(
                [d.x - t, d.y - t, d.x + d.w + t, d.y + d.h + t],
                outline=color,
            )
    return im


def get_engine(prefer_model: bool = True):
    """返回检测引擎。选择顺序：

      1) ONNX 模型（onnxruntime，不需要 PyTorch，加载快、体积小）；
      2) .pt 模型（ultralytics + PyTorch）；
      3) 经典图像处理算法（兜底）。

    如果同时存在 .onnx 与 .pt，用"谁更新用谁"的规则：ONNX 是导出的副本，
    一旦有人只重新训练了 .pt、忘记重新导出 ONNX，也不会误用旧模型。

    想强制指定引擎时可用环境变量 DEFECT_ENGINE=onnx|pt|classical。
    """
    if not prefer_model:
        return ClassicalEngine()

    forced = os.environ.get("DEFECT_ENGINE", "").strip().lower()
    if forced == "classical":
        return ClassicalEngine()

    has_onnx = os.path.isfile(MODEL_ONNX)
    has_pt = os.path.isfile(MODEL_PT)
    if has_onnx and has_pt:
        # 谁更新用谁；同时刻则优先 ONNX
        prefer_onnx = (os.path.getmtime(MODEL_ONNX) >= os.path.getmtime(MODEL_PT))
    else:
        prefer_onnx = has_onnx

    if forced == "pt":
        prefer_onnx = False
    elif forced == "onnx":
        prefer_onnx = True

    if prefer_onnx and has_onnx:
        try:
            return OnnxEngine(MODEL_ONNX)
        except Exception:  # noqa: BLE001  onnxruntime 缺失/模型损坏时继续往下退
            pass
    if has_pt:
        try:
            return YoloEngine(MODEL_PT)
        except Exception:  # noqa: BLE001
            pass
    if has_onnx:
        try:
            return OnnxEngine(MODEL_ONNX)
        except Exception:  # noqa: BLE001
            pass
    return ClassicalEngine()
