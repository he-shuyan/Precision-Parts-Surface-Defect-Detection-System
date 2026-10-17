# -*- coding: utf-8 -*-
"""精密零件表面缺陷检测系统 1.0 —— PyQt5 桌面版入口。

用法：
    双击「启动桌面版.bat」，或在项目目录执行 python main.py
也可以把图片/文件夹拖到窗口里，或作为命令行参数传入。
"""

from __future__ import annotations

import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _fatal(message: str) -> None:
    """启动前的致命错误：尽量用弹窗提示，其次打印到控制台。"""
    try:
        print(message)          # pythonw 下 stdout 为 None，忽略即可
    except Exception:  # noqa: BLE001
        pass
    _write_crash_log(message)
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "启动失败", message)
        del app
    except Exception:  # noqa: BLE001  PyQt5 本身就缺失时只能靠控制台
        pass
    sys.exit(1)


def _write_crash_log(text: str) -> None:
    """把错误信息落盘，方便看不到控制台时排查。"""
    try:
        log_dir = os.path.join(ROOT, "output")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "桌面版运行日志.txt"), "a",
                  encoding="utf-8") as fh:
            fh.write("\n===== %s =====\n%s\n"
                     % (time.strftime("%Y-%m-%d %H:%M:%S"), text))
    except Exception:  # noqa: BLE001
        pass


def _check_dependencies() -> None:
    try:
        import PIL  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        _fatal("缺少图像处理库 Pillow。\n\n"
               "请在项目目录执行：\n"
               r"python\python.exe -m pip install pillow")

    try:
        import PyQt5.QtWidgets  # noqa: F401
    except ImportError:
        _fatal("缺少界面库 PyQt5。\n\n"
               "请在项目目录执行：\n"
               r"python\python.exe -m pip install PyQt5")


def _install_excepthook() -> None:
    """未捕获异常时给出弹窗，避免 pythonw 静默退出。"""
    def hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            if sys.stderr is not None:
                sys.stderr.write(text)
        except Exception:  # noqa: BLE001
            pass
        _write_crash_log(text)
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is not None:
                box = QMessageBox()
                box.setWindowTitle("程序出现异常")
                box.setIcon(QMessageBox.Critical)
                box.setText("程序运行时出现异常，已捕获以避免闪退：")
                box.setDetailedText(text)
                box.exec_()
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


def main() -> int:

    # 必须赶在 import PyQt5 之前：先绑定系统版 MSVC 运行库，
    # 否则 PyQt5 的 Qt5\bin 会顶掉它，导致 PyTorch 加载失败、用不上 YOLO 模型。
    from desktop.qt_boot import preload_system_runtime
    preload_system_runtime()

    _check_dependencies()
    _install_excepthook()

    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication

    # 项目路径含中文时，Qt 自己找不到平台插件，必须先手动指定插件目录
    from desktop.qt_boot import patch_plugin_path
    plugin_path = patch_plugin_path()
    if plugin_path is None:
        print("[警告] 没有找到 Qt 插件目录，界面可能无法启动。")

    # Windows 高分屏适配（必须在创建 QApplication 之前设置）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("精密零件表面缺陷检测系统")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("HUST-DC")

    # 统一字体与外观（Fusion 风格 + 全局 style.qss）
    from desktop.theme import apply_app_theme
    apply_app_theme(app)

    from desktop.engine_loader import load_engine
    from desktop.main_window import MainWindow

    # 命令行里传入的图片/文件夹（支持"打开方式"关联）
    initial = [arg for arg in sys.argv[1:] if os.path.exists(arg)]

    window = MainWindow(initial_paths=initial)
    window.show()

    # 先把窗口显示出来，再加载检测引擎（YOLOv8 首次加载要几秒），
    # 避免双击之后几秒钟没有任何反应。
    def _load_engine():
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            engine, warning = load_engine()
        except Exception as exc:  # noqa: BLE001
            from detector import ClassicalEngine
            engine = ClassicalEngine()
            warning = ("加载检测引擎时出错，已回退到经典图像处理算法。\n\n%s: %s"
                       % (type(exc).__name__, exc))
        finally:
            QApplication.restoreOverrideCursor()
        window.set_engine(engine, warning)

    QTimer.singleShot(0, _load_engine)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
