# -*- coding: utf-8 -*-
"""训练 YOLOv8 零件表面缺陷检测模型，并导出 ONNX。"""

import os
import shutil

from ultralytics import YOLO


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "dataset", "data.yaml")
WEIGHTS = os.path.join(ROOT, "models", "yolov8n.pt")
PROJECT = os.path.join(ROOT, "runs")
NAME = "defect_v1"


def main():
    os.makedirs(PROJECT, exist_ok=True)
    model = YOLO(WEIGHTS)
    model.train(
        data=DATA,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=PROJECT,
        name=NAME,
        patience=30,
        seed=0,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    best = os.path.join(PROJECT, NAME, "weights", "best.pt")
    print("最佳权重：", best)

    # 验证集评估
    metrics = YOLO(best).val(data=DATA, imgsz=640, device=0)
    print(f"mAP@50    = {metrics.box.map50:.4f}")
    print(f"mAP@50-95 = {metrics.box.map:.4f}")

    # 导出 ONNX（便于部署）
    onnx_path = YOLO(best).export(format="onnx", imgsz=640, opset=12)
    print("ONNX 导出：", onnx_path)

    # 复制到模型目录，方便程序调用
    dst_dir = os.path.join(ROOT, "models")
    shutil.copy2(best, os.path.join(dst_dir, "defect_best.pt"))
    onnx_src = best.replace(".pt", ".onnx")
    if os.path.isfile(onnx_src):
        shutil.copy2(onnx_src, os.path.join(dst_dir, "defect_best.onnx"))
    print("模型已复制到：", dst_dir)


if __name__ == "__main__":
    main()
