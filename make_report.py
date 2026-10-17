# -*- coding: utf-8 -*-
"""生成《模型训练与评估报告》（HTML，自包含，可直接打印为 PDF）。"""

import base64
import csv
import os
from datetime import datetime


ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "runs", "defect_v1")
OUT_DIR = os.path.join(ROOT, "报告")


def img_tag(path, width="100%"):
    if not os.path.isfile(path):
        return "<p>（缺少图片：%s）</p>" % os.path.basename(path)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    return f'<img src="data:{mime};base64,{data}" style="width:{width};border:1px solid #ddd;border-radius:6px;margin:8px 0">'


def read_metrics():
    csv_path = os.path.join(RUN, "results.csv")
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    last = rows[-1]
    return {"epochs": len(rows), "best": best, "last": last}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m = read_metrics()
    if m is None:
        raise SystemExit("未找到训练结果 results.csv，请先运行 train_model.py")

    best, last = m["best"], m["last"]

    def pct(v):
        return f"{float(v) * 100:.2f}%"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>精密零件表面缺陷检测系统 1.0 —— 模型训练与评估报告</title>
<style>
body{{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1000px;margin:32px auto;padding:0 24px;color:#1f2937;line-height:1.7}}
h1{{font-size:26px;border-bottom:3px solid #0ea5e9;padding-bottom:10px}}
h2{{font-size:19px;margin-top:34px;color:#0369a1}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
th,td{{border:1px solid #cbd5e1;padding:8px 12px;text-align:left}}
th{{background:#f0f9ff}}
.kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
.kpi div{{flex:1;min-width:150px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px}}
.kpi b{{display:block;font-size:24px;color:#0284c7}}
.muted{{color:#64748b;font-size:13px}}
</style></head><body>

<h1>精密零件表面缺陷检测系统 1.0<br><span style="font-size:17px;color:#475569">模型训练与评估报告</span></h1>
<p class="muted">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>一、任务概述</h2>
<p>本系统面向精密零件表面质量检测，识别三类常见缺陷：<b>划痕</b>、<b>锈迹</b>、<b>缺漆/斑点</b>。
采用 YOLOv8 目标检测模型，输入图像后输出缺陷的位置框、类别与置信度，并可导出标注图与检测报告。</p>

<h2>二、数据集</h2>
<p>针对初期缺乏真实缺陷样本的问题，采用程序化合成方式构建数据集：
在模拟金属表面（拉丝纹理 + 光照渐变 + 颗粒噪声）上生成划痕、锈迹、缺漆三类缺陷，
并叠加亮度/对比度扰动与轻度模糊以模拟真实成像差异，自动生成 YOLO 格式标注。</p>
<table>
<tr><th>训练集</th><td>935 张</td><th>验证集</th><td>165 张</td></tr>
<tr><th>划痕样本框</th><td>673</td><th>锈迹样本框</th><td>726</td></tr>
<tr><th>缺漆样本框</th><td>710</td><th>图像尺寸</th><td>640 × 640</td></tr>
</table>

<h2>三、模型与训练配置</h2>
<table>
<tr><th>模型</th><td>YOLOv8n（迁移学习，COCO 预训练权重）</td></tr>
<tr><th>训练轮数</th><td>{m['epochs']} epochs</td></tr>
<tr><th>输入尺寸 / 批大小</th><td>640 × 640 / batch 16</td></tr>
<tr><th>优化器</th><td>AdamW（自动学习率）</td></tr>
<tr><th>硬件</th><td>NVIDIA GeForce RTX 5060 Laptop GPU（8 GB）</td></tr>
<tr><th>框架</th><td>PyTorch 2.11 + CUDA 12.8 + Ultralytics</td></tr>
</table>

<h2>四、评估结果</h2>
<div class="kpi">
  <div><span class="muted">精确率 Precision</span><b>{pct(best['metrics/precision(B)'])}</b></div>
  <div><span class="muted">召回率 Recall</span><b>{pct(best['metrics/recall(B)'])}</b></div>
  <div><span class="muted">mAP@50</span><b>{pct(best['metrics/mAP50(B)'])}</b></div>
  <div><span class="muted">mAP@50-95</span><b>{pct(best['metrics/mAP50-95(B)'])}</b></div>
</div>
<p class="muted">最佳轮次：第 {best['epoch']} 轮（以 mAP@50 为准）；最终轮次：第 {last['epoch']} 轮。</p>

<h2>五、训练过程曲线</h2>
{img_tag(os.path.join(RUN, "results.png"))}

<h2>六、精确率-召回率曲线（PR Curve）</h2>
{img_tag(os.path.join(RUN, "BoxPR_curve.png"))}

<h2>七、F1 曲线</h2>
{img_tag(os.path.join(RUN, "BoxF1_curve.png"))}

<h2>八、混淆矩阵</h2>
{img_tag(os.path.join(RUN, "confusion_matrix.png"))}

<h2>九、验证集预测效果</h2>
{img_tag(os.path.join(RUN, "val_batch0_pred.jpg"))}
{img_tag(os.path.join(RUN, "val_batch1_pred.jpg"))}

<h2>十、部署</h2>
<p>模型已导出为 ONNX 格式（<code>models/defect_best.onnx</code>），便于跨平台部署；
系统主程序 <code>app.py</code> 默认加载 <code>models/defect_best.pt</code>，
若缺少深度学习环境则自动回退到经典图像处理算法，保证程序始终可用。</p>

</body></html>"""

    out = os.path.join(OUT_DIR, "模型训练与评估报告_纯合成阶段.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("报告已生成：", out)
    print(f"最佳轮次 mAP@50 = {pct(best['metrics/mAP50(B)'])}, mAP@50-95 = {pct(best['metrics/mAP50-95(B)'])}")


if __name__ == "__main__":
    main()
