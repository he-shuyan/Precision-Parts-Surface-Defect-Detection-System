# -*- coding: utf-8 -*-
"""精密零件表面缺陷检测系统 1.0 —— 本地服务入口。

启动后使用浏览器打开 http://127.0.0.1:8000 即可使用：
  - 连续导入图片（多选文件或整个文件夹）
  - 自动识别缺陷并分类（划痕 / 锈迹 / 缺漆斑点）
  - 图上标注缺陷、导出标注图和检测报告
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(ROOT, "vendor")
RUNS_DIR = os.path.join(ROOT, "runs")

# 嵌入式 Python 不会自动把脚本目录加入搜索路径，这里显式补上
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 把第三方库的配置与缓存目录限制在项目内，避免写入用户目录
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(RUNS_DIR, "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNS_DIR, "mplconfig"))
for _d in (os.environ["YOLO_CONFIG_DIR"], os.environ["MPLCONFIGDIR"]):
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError:
        pass

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    if os.path.isdir(VENDOR_DIR):
        sys.path.insert(0, VENDOR_DIR)
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise SystemExit(
                "缺少图像处理库 Pillow。请运行：pip install pillow\n"
                "（或检查 vendor 目录中的 Pillow 是否与当前 Python 版本匹配）"
            ) from exc
    else:
        raise SystemExit("缺少图像处理库 Pillow。请运行：pip install pillow")

from detector import DEFECT_TYPES, annotate, get_engine

WEB_DIR = os.path.join(ROOT, "web")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = get_engine()


def _decode_image(data_url_or_b64: str) -> Image.Image:
    raw = data_url_or_b64
    if "," in raw and raw.strip().lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("图片数据不是有效的 base64 编码") from exc
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError as exc:
        raise ValueError("无法识别的图片格式，请使用 JPG/PNG/BMP 等常见格式") from exc
    return img.convert("RGB")


class Handler(BaseHTTPRequestHandler):
    server_version = "DefectDetect/1.0"

    # ---------- 工具方法 ----------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type=None):
        if not os.path.isfile(path):
            self._send_json({"error": "资源不存在"}, 404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            content_type or mimetypes.guess_type(path)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ---------- 路由 ----------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
        elif path.startswith("/web/"):
            rel = path[len("/web/"):]
            safe = os.path.normpath(rel).lstrip("\\/")
            if ".." in safe:
                self._send_json({"error": "非法路径"}, 400)
                return
            self._send_file(os.path.join(WEB_DIR, safe))
        elif path == "/api/health":
            self._send_json({
                "ok": True,
                "engine": engine.name,
                "types": DEFECT_TYPES,
            })
        else:
            self._send_json({"error": "接口不存在"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/detect", "/api/annotate"):
            self._send_json({"error": "接口不存在"}, 404)
            return

        try:
            payload = json.loads(self._read_body().decode("utf-8") or "{}")
            image = _decode_image(payload.get("image", ""))
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 400)
            return

        defects = engine.detect(image)
        result = {
            "engine": engine.name,
            "width": image.width,
            "height": image.height,
            "count": len(defects),
            "defects": [d.to_dict() for d in defects],
        }

        if path == "/api/detect":
            self._send_json(result)
            return

        # /api/annotate：返回标注后的图片
        annotated = annotate(image, defects)
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        result["image"] = base64.b64encode(buf.getvalue()).decode("ascii")
        self._send_json(result)

    def log_message(self, fmt, *args):  # 静默访问日志，保持控制台整洁
        pass


def _open_browser(url):
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass


def main():
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            port = 8000

    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print("=" * 52)
    print(" 精密零件表面缺陷检测系统 1.0 已启动")
    print(f" 请在浏览器打开：{url}")
    print(f" 检测引擎：{engine.name}（经典图像处理）")
    print(" 按 Ctrl+C 可停止服务")
    print("=" * 52)
    if os.environ.get("DEFECT_NO_BROWSER") != "1":
        threading.Timer(0.8, _open_browser, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
        server.server_close()


if __name__ == "__main__":
    main()
