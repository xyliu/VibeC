"""
VibeC - 极速语音编码助手 (Qwen3-ASR 专版)
重构版本：面向对象、信号槽驱动、状态解耦与 Windows 原生集成
"""

import os
import sys
import time
import math
import ctypes
import winsound
import threading
import gc
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pyaudio
import keyboard
import sherpa_onnx
from pynput import keyboard as pynput_keyboard

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QSystemTrayIcon,
    QMenu,
    QAction,
    QMessageBox
)
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer, QRectF, QPoint, pyqtSignal, QThread


# ==============================================================================
# 1. 核心应用配置
# ==============================================================================
class AppConfig:
    """应用全局配置常量"""
    # 默认语音识别模型目录（首选 0.6B INT8 轻量极速版）
    DEFAULT_MODEL_DIR = "sherpa-onnx-qwen3-asr-0.6B-int8"

    # 加速硬件模式："gpu"（优先检测独立/核显显卡）、"cpu"（纯CPU计算）
    ACCELERATOR_BACKEND = "gpu"

    # 音频流参数配置：16kHz 单声道 16bit 是 ASR 模型的标准输入要求
    SAMPLE_RATE = 16000
    CHANNELS = 1
    AUDIO_FORMAT = pyaudio.paInt16
    CHUNK_SIZE = 1024
    MAX_RECORD_SECONDS = 60.0

    # 默认麦克风索引：None 代表跟随 Windows 系统默认输入设备
    MIC_DEVICE_INDEX: Optional[int] = None

    # 悬浮胶囊尺寸定义
    BAR_WIDTH = 330
    BAR_HEIGHT = 44

    # UI 配色主题方案
    THEMES = {
        "dark": {
            "idle_bg": QColor(22, 24, 32, 220),
            "idle_border": QColor(255, 255, 255, 38),
            "hover_border": QColor(0, 210, 255, 160),
            "recording_bg": QColor(38, 20, 26, 230),
            "recording_border": QColor(255, 75, 75, 200),
            "banner_bg": QColor(28, 24, 46, 230),
            "banner_border": QColor(140, 90, 255, 200),
            "text": QColor(245, 248, 255),
            "subtext": QColor(160, 175, 200),
            "dot": QColor(0, 210, 255),
        },
        "light": {
            "idle_bg": QColor(250, 252, 255, 230),
            "idle_border": QColor(180, 190, 205, 120),
            "hover_border": QColor(0, 140, 255, 160),
            "recording_bg": QColor(255, 235, 238, 235),
            "recording_border": QColor(245, 60, 60, 210),
            "banner_bg": QColor(242, 238, 255, 235),
            "banner_border": QColor(120, 70, 240, 210),
            "text": QColor(25, 30, 45),
            "subtext": QColor(100, 110, 130),
            "dot": QColor(0, 140, 255),
        },
    }


# ==============================================================================
# 2. Windows 平台专属底层交互工具
# ==============================================================================
class Win32Utils:
    """Windows API 与系统交互封装"""

    @staticmethod
    def setup_console_encoding():
        """统一控制台编码为 UTF-8，防止打印特殊状态字符或表情符号时发生崩溃"""
        if sys.platform.startswith("win"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    @staticmethod
    def get_app_dir() -> str:
        """获取当前程序目录，支持源码直接运行与 PyInstaller 单文件打包运行"""
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def apply_no_activate(hwnd: int):
        """给窗口赋予 WS_EX_NOACTIVATE 扩展样式，使鼠标点击窗口时不抢走原输入框焦点"""
        try:
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE)
        except Exception as e:
            print(f"⚠️ 设置无激活样式警告: {e}")

    @staticmethod
    def get_foreground_window() -> Optional[int]:
        """获取当前正在处于前台活跃状态的窗口句柄"""
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            return hwnd if hwnd else None
        except Exception:
            return None

    @staticmethod
    def restore_foreground_window(hwnd: Optional[int]):
        """将焦点平滑切换回用户原先的输入窗口"""
        if not hwnd:
            return
        try:
            user32 = ctypes.windll.user32
            if user32.IsWindow(hwnd):
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.05)
        except Exception:
            pass

    @staticmethod
    def wait_for_physical_keys_release():
        """等待用户按下的物理功能键和修饰键完全松开，避免模拟打字时产生按键冲突"""
        user32 = ctypes.windll.user32
        # 检测按键列表：VK_LWIN(0x5B), VK_RWIN(0x5C), VK_SHIFT(0x10), VK_H(0x48), VK_F8(0x77)
        keys_to_check = (0x5B, 0x5C, 0x10, 0x48, 0x77)
        for _ in range(30):
            if any(user32.GetAsyncKeyState(k) & 0x8000 for k in keys_to_check):
                time.sleep(0.03)
            else:
                break
        time.sleep(0.05)

    @staticmethod
    def type_text(text: str):
        """将文字模拟键入当前光标焦点所在位置"""
        try:
            keyboard.write(text)
            keyboard.write(" ")
        except Exception as e:
            print(f"⚠️ 模拟键盘键入异常: {e}")

# 在模块加载之初立即初始化控制台 UTF-8 编码，杜绝 Windows GBK 环境下打印 Emoji 异常崩溃
Win32Utils.setup_console_encoding()


# ==============================================================================
# 3. 硬件设备探测与管理
# ==============================================================================
class HardwareManager:
    """管理麦克风与硬件加速探测"""

    @staticmethod
    def check_accelerator() -> Tuple[bool, str, str]:
        """优先探测 OpenVINO 及系统 GPU 加速支持状态"""
        # 1. 检测 OpenVINO GPU 设备支持
        try:
            import openvino as ov
            core = ov.Core()
            if "GPU" in core.available_devices:
                gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
                return True, gpu_name, "gpu"
        except Exception:
            pass

        # 2. 备用检测：查询 Windows CIM 显示适配器
        try:
            import subprocess
            cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
            res = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                text=True,
                errors="ignore"
            ).strip()
            if res:
                return True, res.splitlines()[0].strip(), "gpu"
        except Exception:
            pass

        return False, "CPU 基础计算模式", "cpu"

    @staticmethod
    def find_mic_device(pyaudio_instance: pyaudio.PyAudio) -> Optional[int]:
        """列出并选择有效的麦克风输入通道"""
        print("\n===== 可用麦克风设备扫描 =====")
        default_idx = None
        try:
            default_idx = pyaudio_instance.get_default_input_device_info().get("index")
        except Exception:
            pass

        for i in range(pyaudio_instance.get_device_count()):
            try:
                info = pyaudio_instance.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    marker = " <-- 当前系统默认" if i == default_idx else ""
                    print(f"  [{i}] {info.get('name')}{marker}")
            except Exception:
                pass

        if AppConfig.MIC_DEVICE_INDEX is not None:
            print(f"✅ 使用手动指定的麦克风设备索引: {AppConfig.MIC_DEVICE_INDEX}\n")
            return AppConfig.MIC_DEVICE_INDEX

        print(f"✅ 自动选用系统默认设备 [索引: {default_idx}]\n")
        return default_idx


# ==============================================================================
# 4. ASR 模型组件加载器
# ==============================================================================
class ModelLoader:
    """Qwen3-ASR 模型探测与引擎初始化"""

    @classmethod
    def get_candidate_roots(cls) -> list:
        """多级路径回溯策略，兼容打包前调试与绿色打包后独立运行"""
        app_dir = Win32Utils.get_app_dir()
        roots = [app_dir]

        # 向上回溯一级 (例如 dist/ 目录)
        parent_dir = os.path.dirname(app_dir)
        if parent_dir and os.path.exists(parent_dir) and parent_dir not in roots:
            roots.append(parent_dir)

        # 向上回溯两级 (例如工程源码根目录)
        grandparent_dir = os.path.dirname(parent_dir)
        if grandparent_dir and os.path.exists(grandparent_dir) and grandparent_dir not in roots:
            roots.append(grandparent_dir)

        cwd = os.getcwd()
        if cwd not in roots:
            roots.append(cwd)

        return roots

    @classmethod
    def locate_components(cls) -> Optional[Dict[str, str]]:
        """智能寻找并定位完整的 Qwen3-ASR 模型文件组"""
        # 优先探测 1.7B 旗舰高质量模型，若未部署则平滑回退至 0.6B 极速版
        candidate_specs = [
            ("sherpa-onnx-qwen3-asr-1.7B-int8", "Qwen3 1.7B [高质量版]"),
            ("qwen3-asr-1.7b-int8", "Qwen3 1.7B [高质量版]"),
            ("qwen3-asr-1.7b-int4", "Qwen3 1.7B [高质量版]"),
            ("sherpa-onnx-qwen3-asr-0.6B-int8", "Qwen3 0.6B [极速版]"),
            (AppConfig.DEFAULT_MODEL_DIR, "Qwen3-ASR"),
        ]

        for root in cls.get_candidate_roots():
            for folder_name, label in candidate_specs:
                model_dir = os.path.join(root, folder_name) if not os.path.isabs(folder_name) else folder_name
                if not os.path.exists(model_dir):
                    continue

                conv_frontend = os.path.join(model_dir, "conv_frontend.onnx")
                if not os.path.exists(conv_frontend):
                    continue

                encoder = None
                for enc_name in ["encoder.int8.onnx", "encoder.int4.onnx", "encoder.onnx"]:
                    p = os.path.join(model_dir, enc_name)
                    if os.path.exists(p):
                        encoder = p
                        break

                decoder = None
                for dec_name in ["decoder.int8.onnx", "decoder_step.int4.onnx", "decoder.onnx"]:
                    p = os.path.join(model_dir, dec_name)
                    if os.path.exists(p):
                        decoder = p
                        break

                tokenizer_dir = os.path.join(model_dir, "tokenizer")
                if not os.path.exists(tokenizer_dir):
                    tokenizer_dir = model_dir

                if encoder and decoder:
                    return {
                        "model_dir": model_dir,
                        "label": label,
                        "conv_frontend": conv_frontend,
                        "encoder": encoder,
                        "decoder": decoder,
                        "tokenizer": tokenizer_dir,
                    }
        return None

    @classmethod
    def show_missing_dialog(cls, target_dir: str):
        """弹出可视化模型缺失提示，指导用户快速补全模型文件"""
        try:
            QMessageBox.warning(
                None,
                "缺少语音识别大模型组件",
                "【VibeC 启动提示】\n\n"
                "未能在程序运行路径下找到 Qwen3-ASR 模型文件！\n\n"
                "📁 请将项目根目录下载好的模型文件夹：\n"
                "   👉 sherpa-onnx-qwen3-asr-0.6B-int8\n\n"
                "完整复制到当前程序 exe 所在的目录下：\n"
                f"   👉 {target_dir}\n\n"
                "复制完成后重新双击运行即可！"
            )
        except Exception:
            pass

    @classmethod
    def load_recognizer(cls) -> Tuple[Optional[sherpa_onnx.OfflineRecognizer], str]:
        """初始化离线识别引擎"""
        components = cls.locate_components()
        if not components:
            app_dir = Win32Utils.get_app_dir()
            print("❌ 未能在当前环境中检测到完整的 Qwen3-ASR 模型组件。")
            cls.show_missing_dialog(app_dir)
            return None, "未找到可用模型"

        has_gpu, gpu_info, _ = HardwareManager.check_accelerator()
        print(f"💻 硬件设备扫描: {gpu_info}")

        model_label = components["label"]
        provider = "cpu"
        if AppConfig.ACCELERATOR_BACKEND.lower() in ("gpu", "auto") and has_gpu:
            backend_label = f"{model_label} | Intel Arc GPU (8GB)"
            print(f"🚀 已激活硬件加速模式: {gpu_info}")
        else:
            backend_label = f"{model_label} | CPU 多核优化"

        print(f"🔄 正在初始化 {model_label} 离线识别引擎...")
        print(f"  - 特征提取前端: {os.path.basename(components['conv_frontend'])}")
        print(f"  - 编码器: {os.path.basename(components['encoder'])}")
        print(f"  - 解码器: {os.path.basename(components['decoder'])}")
        print(f"  - 词表目录: {components['tokenizer']}")
        print(f"  - 计算后端: {backend_label}")

        recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=components["conv_frontend"],
            encoder=components["encoder"],
            decoder=components["decoder"],
            tokenizer=components["tokenizer"],
            num_threads=4,
            decoding_method="greedy_search",
            debug=False,
            provider=provider,
            max_total_len=512,
            max_new_tokens=128
        )
        print("✅ Qwen3-ASR 模型加载成功！\n")
        return recognizer, backend_label


# ==============================================================================
# 5. 音频录制与 ASR 工作线程 (Qt 信号槽驱动)
# ==============================================================================
class AsrWorker(QThread):
    """后台音频监听、转写与自动打字上屏引擎"""
    sig_recording_started = pyqtSignal()
    sig_recording_stopped = pyqtSignal()
    sig_realtime_text = pyqtSignal(str)
    sig_banner_update = pyqtSignal(str, float)  # 文本，持续显示秒数
    sig_model_loaded = pyqtSignal(str)  # 模型与硬件加速信息就绪通知

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.is_recording = False
        self.recording_start_time = 0.0

        self.recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self.backend_label = "初始化中..."

        # 持久化单例 PyAudio，避免多次打开关闭引发底层驱动内存崩溃
        self.pyaudio_instance = pyaudio.PyAudio()
        self.mic_index: Optional[int] = None
        self.target_window_hwnd: Optional[int] = None

        # 录音状态切换请求事件
        self._toggle_requested = threading.Event()

    def initialize_engine(self):
        """初始化麦克风和 ASR 识别引擎"""
        self.mic_index = HardwareManager.find_mic_device(self.pyaudio_instance)
        self.recognizer, self.backend_label = ModelLoader.load_recognizer()

    def toggle_recording(self):
        """触发录音启停切换"""
        # 在触发录音的瞬间记录用户正在输入的前台窗口
        if not self.is_recording:
            self.target_window_hwnd = Win32Utils.get_foreground_window()
        self._toggle_requested.set()

    def stop_worker(self):
        """安全停止工作线程并清理资源"""
        self.running = False
        self.is_recording = False
        self._toggle_requested.set()
        try:
            self.pyaudio_instance.terminate()
        except Exception:
            pass

    def run(self):
        """后台主循环"""
        self.initialize_engine()
        self.sig_model_loaded.emit(self.backend_label)
        winsound.Beep(600, 200)

        print("👉 单击桌面【悬浮小条】或轻按键盘【F8】键开始录音，再次单击立即识别上屏。")
        print("👉 可通过系统托盘菜单或悬浮条右键选择安全退出。\n")

        while self.running:
            # 等待录音触发
            self._toggle_requested.wait(timeout=0.1)
            if not self.running:
                break

            if self._toggle_requested.is_set():
                self._toggle_requested.clear()
                self._handle_recording_session()

    def _handle_recording_session(self):
        """单次录音会话执行流程"""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.sig_recording_started.emit()
        winsound.Beep(1500, 100)

        stream = None
        try:
            stream = self.pyaudio_instance.open(
                format=AppConfig.AUDIO_FORMAT,
                channels=AppConfig.CHANNELS,
                rate=AppConfig.SAMPLE_RATE,
                input=True,
                input_device_index=self.mic_index,
                frames_per_buffer=AppConfig.CHUNK_SIZE
            )
        except Exception as e:
            print(f"❌ 打开麦克风流失败: {e}")
            self.is_recording = False
            self.sig_recording_stopped.emit()
            self.sig_banner_update.emit("⚠️ 麦克风打开失败", 2.0)
            return

        self.sig_realtime_text.emit("🎤 Qwen3 正在聆听...")
        frames = []
        last_infer_time = time.time()
        mute_warned = False

        # 录音采样循环
        while self.running and self.is_recording:
            # 检查是否有停止录音请求
            if self._toggle_requested.is_set():
                self._toggle_requested.clear()
                break

            # 超过最大时长限制自动停止
            if time.time() - self.recording_start_time > AppConfig.MAX_RECORD_SECONDS:
                break

            try:
                data = stream.read(AppConfig.CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                continue

            # 每隔 0.5 秒进行一次轻量伪实时预览解码
            now = time.time()
            if self.recognizer and (now - last_infer_time > 0.5) and len(frames) > 6:
                raw_tmp = b"".join(frames)
                audio_tmp = np.frombuffer(raw_tmp, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(audio_tmp ** 2)))

                if rms < 0.001:
                    if not mute_warned:
                        self.sig_realtime_text.emit("⚠️ 麦克风输入信号微弱")
                        mute_warned = True
                    last_infer_time = now
                    continue

                mute_warned = False
                c_stream = self.recognizer.create_stream()
                c_stream.accept_waveform(AppConfig.SAMPLE_RATE, audio_tmp)
                self.recognizer.decode_stream(c_stream)
                if c_stream.result.text:
                    self.sig_realtime_text.emit(c_stream.result.text)
                last_infer_time = now

        self.is_recording = False
        self.sig_recording_stopped.emit()
        winsound.Beep(1000, 100)

        # 关闭音频流
        try:
            if stream:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass

        # 执行最终的高精度端到端识别
        self._execute_final_recognition(frames)

    def _execute_final_recognition(self, frames: list):
        """录音结束后执行最终完整解码并打字上屏"""
        if not self.recognizer or len(frames) <= 5:
            self.sig_realtime_text.emit("")
            return

        self.sig_banner_update.emit("⚡ 正在极速识别...", 5.0)

        raw_data = b"".join(frames)
        audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        rms_final = float(np.sqrt(np.mean(audio_float32 ** 2)))

        duration = len(frames) * AppConfig.CHUNK_SIZE / AppConfig.SAMPLE_RATE
        print(f"🔍 采集完成: 时长 {duration:.1f}s, RMS 响度 {rms_final:.4f}")

        if rms_final <= 0.001:
            self.sig_banner_update.emit("⚠️ 麦克风无声音信号", 1.5)
            self.sig_realtime_text.emit("")
            return

        c_stream = self.recognizer.create_stream()
        c_stream.accept_waveform(AppConfig.SAMPLE_RATE, audio_float32)
        self.recognizer.decode_stream(c_stream)
        text = c_stream.result.text.strip()
        print(f"📝 [识别结果]: '{text}'")

        if text:
            # 等待所有物理按键完全释放，保证打字平稳无按键连带冲突
            Win32Utils.wait_for_physical_keys_release()

            # 将焦点平滑归还给原工作输入窗口
            Win32Utils.restore_foreground_window(self.target_window_hwnd)

            self.sig_banner_update.emit(f"✅ 已上屏: {text[:6]}", 1.5)
            Win32Utils.type_text(text)
        else:
            self.sig_banner_update.emit("⚠️ 未识别到有效内容", 1.2)

        self.sig_realtime_text.emit("")
        gc.collect()


# ==============================================================================
# 6. Windows 全局热键监听器 (F8 功能键无字符污染)
# ==============================================================================
class GlobalHotkeyHook:
    """使用 pynput 捕获 Windows 全局 F8 按键消息"""

    def __init__(self, on_trigger_callback):
        self.on_trigger = on_trigger_callback
        self.listener: Optional[pynput_keyboard.Listener] = None

    def start(self):
        """启动底层 Windows 键盘消息过滤监听"""
        self.listener = pynput_keyboard.Listener(win32_event_filter=self._win32_filter)
        self.listener.start()

    def stop(self):
        """停止监听"""
        if self.listener:
            try:
                self.listener.stop()
            except Exception:
                pass

    def _win32_filter(self, msg, data) -> bool:
        """底层消息过滤：拦截 F8 避免触发宿主软件快捷键，并切换录音状态"""
        try:
            WM_KEYDOWN = 0x0100
            WM_KEYUP = 0x0101
            WM_SYSKEYDOWN = 0x0104
            WM_SYSKEYUP = 0x0105
            VK_F8 = 0x77

            if data.vkCode == VK_F8:
                if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    self.on_trigger()
                    # 返回 False 吞掉按键，避免触发其他软件自带的 F8 逻辑
                    return False
                elif msg in (WM_KEYUP, WM_SYSKEYUP):
                    return False
            return True
        except Exception:
            return True


# ==============================================================================
# 7. 灵动胶囊悬浮小条 UI
# ==============================================================================
class FloatingBarUI(QWidget):
    """屏幕置顶的现代化悬浮胶囊界面"""

    def __init__(self, worker: AsrWorker):
        super().__init__()
        self.worker = worker
        self.theme_name = "dark"

        # 界面显示状态数据
        self.is_recording = False
        self.recording_start_time = 0.0
        self.realtime_text = ""
        self.status_banner = ""
        self.status_banner_expire = 0.0

        # 拖拽交互状态
        self.drag_start_pos: Optional[QPoint] = None
        self.is_dragging = False
        self.is_hovered = False
        self.model_info = "模型加载中..."

        self._init_window_flags()
        self._init_geometry()
        self._bind_signals()

        # 30fps 定时器驱动呼吸灯动画与计时平滑刷新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(33)

        self.show()
        print("✅ Qwen3 悬浮小条已就绪，支持鼠标点击与快捷键一键唤醒。")

    def _init_window_flags(self):
        """初始化窗口置顶、无边框与不抢焦点等属性"""
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint
            | Qt.FramelessWindowHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def _init_geometry(self):
        """默认将悬浮条放置在屏幕顶部居中"""
        self.setFixedSize(AppConfig.BAR_WIDTH, AppConfig.BAR_HEIGHT)
        desktop = QApplication.desktop()
        rect = desktop.availableGeometry()
        center_x = (rect.width() - AppConfig.BAR_WIDTH) // 2
        self.move(center_x, 35)

    def _bind_signals(self):
        """连接后台工作线程的 PyQt 信号"""
        self.worker.sig_recording_started.connect(self._on_recording_started)
        self.worker.sig_recording_stopped.connect(self._on_recording_stopped)
        self.worker.sig_realtime_text.connect(self._on_realtime_text)
        self.worker.sig_banner_update.connect(self._on_banner_update)
        self.worker.sig_model_loaded.connect(self._on_model_loaded)

    def _on_model_loaded(self, label: str):
        self.model_info = label

    def showEvent(self, event):
        super().showEvent(event)
        # 赋予无焦点扩展样式，确保鼠标点击不抢走输入框焦点
        Win32Utils.apply_no_activate(int(self.winId()))

    def enterEvent(self, event):
        self.is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.is_hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 记录点击瞬间的前台窗口，作为输入焦点还原的双保险
            fg = Win32Utils.get_foreground_window()
            if fg and fg != int(self.winId()):
                self.worker.target_window_hwnd = fg

            self.drag_start_pos = event.globalPos() - self.frameGeometry().topLeft()
            self.is_dragging = False
        elif event.button() == Qt.RightButton:
            self._show_context_menu(event.globalPos())

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_start_pos is not None:
            self.is_dragging = True
            self.move(event.globalPos() - self.drag_start_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 若鼠标位移极小，判定为单击，触发录音切换
            if not self.is_dragging:
                self.worker.toggle_recording()
            self.drag_start_pos = None
            self.is_dragging = False

    def _show_context_menu(self, global_pos: QPoint):
        """弹出悬浮条右键功能菜单"""
        menu = QMenu(self)

        act_model = QAction(f"🤖 {self.model_info}", menu)
        act_model.setEnabled(False)
        menu.addAction(act_model)
        menu.addSeparator()

        act_theme = QAction("🎨 切换主题 (明亮/暗黑)", menu)
        act_theme.triggered.connect(self.toggle_theme)
        menu.addAction(act_theme)

        act_center = QAction("📌 恢复顶部居中", menu)
        act_center.triggered.connect(self.center_to_top)
        menu.addAction(act_center)

        menu.addSeparator()
        act_exit = QAction("❌ 完全退出", menu)
        act_exit.triggered.connect(lambda: os._exit(0))
        menu.addAction(act_exit)

        menu.exec_(global_pos)

    def toggle_theme(self):
        """切换深色与浅色主题"""
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.update()

    def center_to_top(self):
        """恢复居中置顶位置"""
        rect = QApplication.desktop().availableGeometry()
        self.move((rect.width() - AppConfig.BAR_WIDTH) // 2, 35)

    def _on_recording_started(self):
        self.is_recording = True
        self.recording_start_time = time.time()
        self.update()

    def _on_recording_stopped(self):
        self.is_recording = False
        self.update()

    def _on_realtime_text(self, text: str):
        self.realtime_text = text
        self.update()

    def _on_banner_update(self, text: str, duration: float):
        self.status_banner = text
        self.status_banner_expire = time.time() + duration
        self.update()

    def paintEvent(self, event):
        """胶囊背景、呼吸动效、图标与动态文字绘制"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        theme = AppConfig.THEMES.get(self.theme_name, AppConfig.THEMES["dark"])
        w, h = AppConfig.BAR_WIDTH, AppConfig.BAR_HEIGHT
        rect = QRectF(1.5, 1.5, w - 3.0, h - 3.0)
        radius = (h - 3.0) / 2.0

        # 判断当前是否有横幅提示处于活跃展示状态
        active_banner = ""
        if self.status_banner and (time.time() < self.status_banner_expire or "识别" in self.status_banner):
            active_banner = self.status_banner

        # 根据当前状态选取主题配色
        if self.is_recording:
            bg_color = theme["recording_bg"]
            border_color = theme["recording_border"]
        elif active_banner:
            bg_color = theme["banner_bg"]
            border_color = theme["banner_border"]
        else:
            bg_color = theme["idle_bg"]
            border_color = theme["hover_border"] if self.is_hovered else theme["idle_border"]

        # 1. 绘制圆角胶囊背景与边框
        painter.setBrush(bg_color)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawRoundedRect(rect, radius, radius)

        # 2. 绘制左侧指示点
        btn_center_x, btn_center_y = 24.0, h / 2.0

        if self.is_recording:
            # 录音中：动态脉冲呼吸光晕
            pulse = 0.5 + 0.5 * math.sin(time.time() * 7)
            glow_radius = 8.0 + pulse * 4.0
            painter.setBrush(QColor(255, 50, 50, int(60 + pulse * 100)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(btn_center_x), int(btn_center_y)), int(glow_radius), int(glow_radius))

            painter.setBrush(QColor(255, 60, 60))
            painter.drawEllipse(QPoint(int(btn_center_x), int(btn_center_y)), 5, 5)

            # 录音计时展示
            elapsed = min(time.time() - self.recording_start_time, AppConfig.MAX_RECORD_SECONDS)
            painter.setPen(theme["text"])
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(QRectF(44, 0, 50, h), Qt.AlignVCenter | Qt.AlignLeft, f"00:{int(elapsed):02d}")

            # 动态转写内容或默认引导提示
            painter.setFont(QFont("Microsoft YaHei UI", 9))
            painter.setPen(theme["subtext"])
            display_text = self.realtime_text if (self.realtime_text and "Qwen3" not in self.realtime_text) else "录音中 · 点击停止上屏"
            painter.drawText(QRectF(100, 0, w - 145, h), Qt.AlignVCenter | Qt.AlignLeft, display_text)

            # 右侧停止方块指示
            stop_btn_rect = QRectF(w - 34, (h - 18) / 2.0, 18, 18)
            painter.setBrush(QColor(255, 75, 75, 210))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(stop_btn_rect, 4, 4)

        elif active_banner:
            # 状态提示态（如：正在识别、已上屏）
            painter.setBrush(QColor(140, 80, 255))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(btn_center_x), int(btn_center_y)), 5, 5)

            painter.setPen(theme["text"])
            painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
            painter.drawText(QRectF(44, 0, w - 55, h), Qt.AlignVCenter | Qt.AlignLeft, active_banner)

        else:
            # 闲置就绪态：青色点 + 快捷键提示
            painter.setBrush(theme["dot"])
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(btn_center_x), int(btn_center_y)), 6, 6)

            painter.setPen(theme["text"])
            painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
            painter.drawText(QRectF(44, 0, 115, h), Qt.AlignVCenter | Qt.AlignLeft, "🎙️ 点击录音")

            painter.setPen(theme["subtext"])
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRectF(160, 0, w - 175, h), Qt.AlignVCenter | Qt.AlignRight, "F8 快捷键")


# ==============================================================================
# 8. 系统托盘管理
# ==============================================================================
class TrayManager:
    """系统托盘图标与上下文菜单"""

    def __init__(self, app: QApplication, floating_bar: FloatingBarUI):
        self.app = app
        self.floating_bar = floating_bar
        self.tray = QSystemTrayIcon()
        self._init_tray()

    def _init_tray(self):
        # 绘制精致圆点托盘图标
        pix = QPixmap(32, 32)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(0, 200, 255))
        p.setPen(Qt.NoPen)
        p.drawEllipse(4, 4, 24, 24)
        p.end()

        self.tray.setIcon(QIcon(pix))
        self.tray.setToolTip("VibeC 语音助手 (Qwen3-ASR)")

        menu = QMenu()

        self.act_model_info = QAction("🤖 正在检测模型...", menu)
        self.act_model_info.setEnabled(False)
        menu.addAction(self.act_model_info)
        menu.addSeparator()

        act_toggle = QAction("👁️ 显示/隐藏悬浮条", menu)
        act_toggle.triggered.connect(self._toggle_bar_visibility)
        menu.addAction(act_toggle)

        act_info = QAction("📝 使用说明", menu)
        act_info.triggered.connect(self._show_info_dialog)
        menu.addAction(act_info)

        act_theme = QAction("🎨 切换主题", menu)
        act_theme.triggered.connect(self.floating_bar.toggle_theme)
        menu.addAction(act_theme)

        menu.addSeparator()
        act_exit = QAction("❌ 完全退出", menu)
        act_exit.triggered.connect(self._exit_app)
        menu.addAction(act_exit)

        self.tray.setContextMenu(menu)
        self.tray.show()

    def update_model_info(self, backend_label: str):
        """动态更新托盘提示文本与菜单信息"""
        self.tray.setToolTip(f"VibeC 语音助手\n{backend_label}")
        if hasattr(self, "act_model_info"):
            self.act_model_info.setText(f"🤖 {backend_label}")

    def _toggle_bar_visibility(self):
        if self.floating_bar.isVisible():
            self.floating_bar.hide()
        else:
            self.floating_bar.show()

    def _show_info_dialog(self):
        QMessageBox.information(
            None,
            "使用说明",
            "【VibeC - Qwen3-ASR 语音输入法】\n\n"
            "1. 鼠标单击【悬浮小条】或按键盘【F8】键开始录音\n"
            "2. 讲话完毕后再次点击悬浮小条或按【F8】，自动文字上屏\n"
            "3. 鼠标可随意按住悬浮小条拖拽到屏幕任意习惯位置\n"
            "4. 右键单击悬浮小条可切换暗黑/明亮主题或一键居中\n"
            "5. 右键任务栏右下角托盘图标可安全退出程序"
        )

    def _exit_app(self):
        self.tray.hide()
        os._exit(0)


# ==============================================================================
# 9. 主程序入口
# ==============================================================================
def main():
    # 1. 确保 Windows 控制台字符编码正确
    Win32Utils.setup_console_encoding()

    # 2. 启动 Qt 应用程序事件循环
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # 3. 启动 ASR 核心工作线程
    worker = AsrWorker()

    # 4. 创建置顶悬浮小条与托盘图标
    floating_bar = FloatingBarUI(worker)
    tray = TrayManager(app, floating_bar)
    worker.sig_model_loaded.connect(tray.update_model_info)

    # 5. 启动全局 F8 热键底层监听
    hotkey = GlobalHotkeyHook(on_trigger_callback=worker.toggle_recording)
    hotkey.start()

    # 6. 开启工作线程
    worker.start()

    # 进入 Qt 事件主循环
    exit_code = app.exec_()

    # 退出前释放资源
    hotkey.stop()
    worker.stop_worker()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
