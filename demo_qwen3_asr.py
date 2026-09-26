import time
import math
import pyaudio
import keyboard
import numpy as np
import sherpa_onnx
import os
import sys
import ctypes
import winsound
import threading
import gc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pynput import keyboard as pynput_keyboard

from PyQt5.QtWidgets import QApplication, QWidget, QSystemTrayIcon, QMenu, QAction, QMessageBox
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer, QRectF

# 适配 Windows 控制台默认 GBK 编码环境，避免打印状态 Emoji 时发生 Unicode 编码崩溃
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ================= 配置参数 =================
# Qwen3-ASR 模型目录，支持官方导出的 0.6B INT8 轻量版
MODEL_DIR_NAME = "sherpa-onnx-qwen3-asr-0.6B-int8"

# 加速后端模式：可选 "gpu"（优先Intel Arc GPU加速，推荐）、"cpu"（纯CPU运算）、"npu"（实验性NPU）
ACCELERATOR_BACKEND = "gpu"

HOTKEY = "windows+shift+h"
EXIT_HOTKEY = "ctrl+shift+q"

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_RECORD_SECONDS = 60.0
# 麦克风设备索引：None 为自动选择系统默认设备
MIC_DEVICE_INDEX = None
# ============================================

# 全局运行状态与悬浮小条交互缓冲
is_recording = False
recording_start = 0
realtime_text = ""
current_backend_label = "Intel Arc GPU"
target_input_hwnd = None
status_banner = ""
status_banner_expire = 0

def get_application_path():
    # 获取运行目录，确保以源码运行或打包为独立文件时均能正确定位同级模型
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def check_hardware_accelerator():
    # 动态检测当前电脑的加速硬件（优先识别 Intel Arc 8GB GPU 独立/核显显卡）
    gpu_name = "未检测到独立/集成 GPU"
    has_gpu = False
    try:
        import openvino as ov
        core = ov.Core()
        if "GPU" in core.available_devices:
            has_gpu = True
            gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            return has_gpu, gpu_name, "gpu"
    except Exception:
        pass

    # 备选检查系统显卡信息
    try:
        import subprocess
        cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        res = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, errors="ignore").strip()
        if res:
            first_gpu = res.splitlines()[0].strip()
            gpu_name = first_gpu
            has_gpu = True
            return has_gpu, gpu_name, "gpu"
    except Exception:
        pass

    return False, "CPU 基础模式", "cpu"

def get_mic_volume_interface():
    # 访问系统麦克风端点对象，用于自动恢复被误静音的录音状态
    try:
        device = AudioUtilities.GetMicrophone()
        if device is None:
            return None
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as e:
        print(f"⚠️ 无法访问麦克风接口: {e}")
        return None

def get_mic_mute() -> bool:
    vol = get_mic_volume_interface()
    return bool(vol.GetMute()) if vol else False

def set_mic_mute(mute: bool):
    vol = get_mic_volume_interface()
    if vol:
        vol.SetMute(1 if mute else 0, None)
        print(f"🎙️ 麦克风静音状态已切换为: {'静音' if mute else '开启'}")

def find_mic_device():
    # 遍历音频输入设备，防止多声卡环境下录入无声通道
    p = pyaudio.PyAudio()
    print("\n===== 可用麦克风设备列表 =====")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            marker = " <-- 当前默认" if i == p.get_default_input_device_info()['index'] else ""
            print(f"  [{i}] {info['name']}{marker}")
    p.terminate()

    if MIC_DEVICE_INDEX is not None:
        print(f"\n✅ 采用手动指定的麦克风设备索引: {MIC_DEVICE_INDEX}")
        return MIC_DEVICE_INDEX

    default_idx = None
    p2 = pyaudio.PyAudio()
    try:
        default_idx = p2.get_default_input_device_info()['index']
    except Exception:
        pass
    p2.terminate()
    print(f"\n✅ 使用系统默认输入设备 [索引 {default_idx}]")
    print("============================\n")
    return default_idx

def toggle_recording():
    global is_recording
    is_recording = not is_recording
    print(f"DEBUG: 录音开关状态 -> {is_recording}")

def ensure_win_h_disabled():
    # 自动在当前用户注册表中写入 DisabledHotkeys='H'，从根源切断 Windows Explorer 响应 Win+H
    try:
        import winreg
        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_ALL_ACCESS) as key:
            try:
                val, _ = winreg.QueryValueEx(key, "DisabledHotkeys")
            except FileNotFoundError:
                val = ""
            if "H" not in val:
                new_val = val + "H"
                winreg.SetValueEx(key, "DisabledHotkeys", 0, winreg.REG_SZ, new_val)
                # 广播设置变更消息，通知系统 Shell 刷新热键规则
                HWND_BROADCAST = 0xFFFF
                WM_SETTINGCHANGE = 0x001A
                ctypes.windll.user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, "TraySettings", 2, 2000, ctypes.byref(ctypes.c_ulong()))
                print("🔒 已在系统注册表中成功禁用 Win+H 默认绑定！")
    except Exception as e:
        print(f"⚠️ 配置系统热键屏蔽项提示: {e}")

pynput_listener = None
h_suppressed = False
win_pressed = False
shift_pressed = False
suppress_next_win_up = False

def log_debug(message):
    try:
        app_path = get_application_path()
        log_file = os.path.join(app_path, "keyboard_debug.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass

def win32_event_filter(msg, data):
    # 底层键盘钩子，精准监听 Win + Shift + H 组合键，吞掉 H 键防止在文本框中打出字符
    global h_suppressed, win_pressed, shift_pressed, suppress_next_win_up
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105

    VK_LWIN = 0x5B
    VK_RWIN = 0x5C
    VK_SHIFT = 0x10
    VK_LSHIFT = 0xA0
    VK_RSHIFT = 0xA1
    VK_H = 0x48
    VK_DUMMY = 0xFF

    vk_code = data.vkCode

    # 监听 Win 键状态
    if vk_code in (VK_LWIN, VK_RWIN):
        if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
            win_pressed = True
        elif msg in (WM_KEYUP, WM_SYSKEYUP):
            win_pressed = False
            h_suppressed = False
            if suppress_next_win_up:
                suppress_next_win_up = False
                return False

    # 监听 Shift 键状态
    if vk_code in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
        if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
            shift_pressed = True
        elif msg in (WM_KEYUP, WM_SYSKEYUP):
            shift_pressed = False

    # 捕获 H 键：同时使用状态变量与系统底层实时状态进行双重校验
    if vk_code == VK_H:
        # 实时检测物理 Win 与 Shift 是否按压中
        is_win_down = win_pressed or bool(ctypes.windll.user32.GetAsyncKeyState(VK_LWIN) & 0x8000) or bool(ctypes.windll.user32.GetAsyncKeyState(VK_RWIN) & 0x8000)
        is_shift_down = shift_pressed or bool(ctypes.windll.user32.GetAsyncKeyState(VK_SHIFT) & 0x8000)

        if is_win_down and is_shift_down:
            if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
                if not h_suppressed:
                    h_suppressed = True
                    suppress_next_win_up = True
                    # 注入虚拟按键中和 Win 状态
                    ctypes.windll.user32.keybd_event(VK_DUMMY, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(VK_DUMMY, 0, 2, 0)
                    log_debug("Win+Shift+H 命中，切换录音状态")
                    toggle_recording()
                return False
            elif msg in (WM_KEYUP, WM_SYSKEYUP):
                return False

    return True

def init_qwen3_recognizer(base_dir: str):
    # 自动探测优先使用可用且完整的模型版本（0.6B 极速版或 1.7B 版）
    global current_backend_label

    candidate_dirs = [
        ("sherpa-onnx-qwen3-asr-0.6B-int8", "Qwen3 0.6B [极速版]"),
        (base_dir, "Qwen3-ASR"),
        ("qwen3-asr-1.7b-int4", "Qwen3 1.7B [高质量版]")
    ]

    app_path = get_application_path()
    selected_components = None

    for d_name, v_label in candidate_dirs:
        model_dir = os.path.join(app_path, d_name)
        if not os.path.exists(model_dir):
            continue

        conv_frontend = os.path.join(model_dir, "conv_frontend.onnx")
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

        if conv_frontend and os.path.exists(conv_frontend) and encoder and decoder:
            selected_components = {
                "model_dir": model_dir,
                "label": v_label,
                "conv_frontend": conv_frontend,
                "encoder": encoder,
                "decoder": decoder,
                "tokenizer_dir": tokenizer_dir
            }
            break

    if selected_components is None:
        print("❌ 未能在当前环境中检测到完整的 Qwen3-ASR 模型组件。")
        print(f"请检查模型存放路径: {os.path.join(app_path, MODEL_DIR_NAME)}")
        return None

    model_dir = selected_components["model_dir"]
    model_version_label = selected_components["label"]
    conv_frontend = selected_components["conv_frontend"]
    encoder = selected_components["encoder"]
    decoder = selected_components["decoder"]
    tokenizer_dir = selected_components["tokenizer_dir"]

    # 检测并配置运行硬件提供方（Execution Provider）
    has_gpu, gpu_info, dev_type = check_hardware_accelerator()
    print(f"💻 硬件设备扫描: {gpu_info}")

    selected_provider = "cpu"
    if ACCELERATOR_BACKEND.lower() in ("gpu", "auto") and has_gpu:
        current_backend_label = f"{model_version_label} | Intel Arc GPU (8GB)"
        print(f"🚀 已激活硬件加速模式: {gpu_info}")
    else:
        current_backend_label = f"{model_version_label} | CPU 多核优化"

    print(f"🔄 正在初始化 {model_version_label} 离线识别引擎...")
    print(f"  - 特征提取前端: {os.path.basename(conv_frontend)}")
    print(f"  - 编码器: {os.path.basename(encoder)}")
    print(f"  - 解码器 (带KV Cache): {os.path.basename(decoder)}")
    print(f"  - 词表目录: {tokenizer_dir}")
    print(f"  - 运算后端: {current_backend_label}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=conv_frontend,
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer_dir,
        num_threads=4,
        decoding_method="greedy_search",
        debug=False,
        provider=selected_provider,
        max_total_len=512,
        max_new_tokens=128
    )
    print("✅ Qwen3-ASR 模型加载成功！\n")
    return recognizer

def background_task():
    global is_recording, recording_start

    try:
        app_path = get_application_path()
        log_file = os.path.join(app_path, "keyboard_debug.log")
        if os.path.exists(log_file):
            os.remove(log_file)
    except Exception:
        pass
    log_debug("=== Qwen3-ASR Keyboard Listener Started ===")

    # 启动时确保系统注册表已屏蔽 Win+H 默认唤起行为
    ensure_win_h_disabled()

    global pynput_listener
    pynput_listener = pynput_keyboard.Listener(win32_event_filter=win32_event_filter)
    pynput_listener.start()

    keyboard.add_hotkey(EXIT_HOTKEY, lambda: os._exit(0))
    mic_device_idx = find_mic_device()

    app_path = get_application_path()
    model_dir = os.path.join(app_path, MODEL_DIR_NAME)

    recognizer = init_qwen3_recognizer(model_dir)
    if recognizer is None:
        print("⚠️ 未找到可用的 Qwen3-ASR 模型文件，程序进入待命状态。")
        print(f"💡 请将下载好的模型解压到: {model_dir}\n")

    print("👉 单击桌面【悬浮小条】或按下【Win+Shift+H】开启录音，再次单击立即识别上屏。")
    print(f"👉 按下【{EXIT_HOTKEY}】安全退出后台。")
    winsound.Beep(600, 200)

    while True:
        try:
            if is_recording:
                # 记录录音开始前的前台目标窗口句柄，确保无论鼠标如何点击均能精准打字上屏
                global target_input_hwnd, status_banner, status_banner_expire
                fg = ctypes.windll.user32.GetForegroundWindow()
                if fg:
                    target_input_hwnd = fg

                winsound.Beep(1500, 100)
                recording_start = time.time()
                frames = []

                p = pyaudio.PyAudio()
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=mic_device_idx,
                    frames_per_buffer=CHUNK
                )

                original_mute_state = get_mic_mute()
                if original_mute_state:
                    print("🎙️ 检测到麦克风处于静音，自动取消静音以确保正常录音...")
                    set_mic_mute(False)

                global realtime_text
                realtime_text = "🎤 Qwen3 正在聆听..."
                last_infer_time = time.time()
                mute_warned = False

                while is_recording and (time.time() - recording_start) <= MAX_RECORD_SECONDS:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)

                    # 每隔 0.5 秒进行一次快速伪实时语音预览
                    if recognizer and (time.time() - last_infer_time > 0.5) and len(frames) > 6:
                        raw_tmp = b''.join(frames)
                        audio_tmp = np.frombuffer(raw_tmp, dtype=np.int16).astype(np.float32) / 32768.0

                        rms = float(np.sqrt(np.mean(audio_tmp**2)))
                        if rms < 0.001:
                            if not mute_warned:
                                realtime_text = "⚠️ 麦克风输入信号微弱\n请检查设备是否静音"
                                mute_warned = True
                            last_infer_time = time.time()
                            continue

                        mute_warned = False
                        c_stream = recognizer.create_stream()
                        c_stream.accept_waveform(RATE, audio_tmp)
                        recognizer.decode_stream(c_stream)
                        if c_stream.result.text:
                            realtime_text = c_stream.result.text
                        last_infer_time = time.time()

                is_recording = False
                winsound.Beep(1000, 100)
                stream.stop_stream()
                stream.close()
                p.terminate()

                if original_mute_state:
                    set_mic_mute(True)
                    print("🎙️ 录音结束，恢复静音状态")

                # 录音完成后执行最终端到端精准识别
                if recognizer and len(frames) > 5:
                    status_banner = "⚡ 正在极速识别..."
                    raw_data = b''.join(frames)
                    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    rms_final = float(np.sqrt(np.mean(audio_float32**2)))
                    print(f"🔍 [诊断信息] 采集时长:{len(frames)*CHUNK/RATE:.1f}s, 响度RMS:{rms_final:.4f}")

                    if rms_final > 0.001:
                        c_stream = recognizer.create_stream()
                        c_stream.accept_waveform(RATE, audio_float32)
                        recognizer.decode_stream(c_stream)
                        text = c_stream.result.text.strip()
                        print(f"📝 [Qwen3-ASR 识别结果]: '{text}'")
                        if text:
                            # 关键防御：严格等待物理 Win 键、Shift 键与 H 键真正释放，防止模拟按键与物理按压冲突
                            while (ctypes.windll.user32.GetAsyncKeyState(0x5B) & 0x8000) or \
                                  (ctypes.windll.user32.GetAsyncKeyState(0x5C) & 0x8000) or \
                                  (ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000) or \
                                  (ctypes.windll.user32.GetAsyncKeyState(0x48) & 0x8000):
                                time.sleep(0.03)
                            time.sleep(0.05)

                            # 关键焦点还原：将焦点平滑切换回用户最初正在输入的窗口
                            if target_input_hwnd and ctypes.windll.user32.IsWindow(target_input_hwnd):
                                ctypes.windll.user32.SetForegroundWindow(target_input_hwnd)
                                time.sleep(0.05)

                            realtime_text = text
                            status_banner = f"✅ 已上屏: {text[:6]}"
                            status_banner_expire = time.time() + 1.5
                            keyboard.write(text)
                            keyboard.write(" ")
                            time.sleep(0.5)
                        else:
                            status_banner = "⚠️ 未识别到有效内容"
                            status_banner_expire = time.time() + 1.2
                    else:
                        realtime_text = "⚠️ 麦克风无声音信号"
                        status_banner = "⚠️ 麦克风无声音信号"
                        status_banner_expire = time.time() + 1.5
                        time.sleep(1.2)
                else:
                    status_banner = ""

                realtime_text = ""
                time.sleep(0.2)
                gc.collect()

            if keyboard.is_pressed(EXIT_HOTKEY):
                winsound.Beep(400, 300)
                os._exit(0)

            time.sleep(0.05)

        except KeyboardInterrupt:
            os._exit(0)
        except Exception as e:
            print(f"运行异常: {e}")
            time.sleep(1)

class FloatingBarUI(QWidget):
    # 极简现代化悬浮小条 (Floating Capsule Bar)，支持鼠标点击录音/结束与随意拖动
    def __init__(self):
        super().__init__()
        self.theme = "dark"
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

        self.bar_w = 330
        self.bar_h = 44
        self.setFixedSize(self.bar_w, self.bar_h)

        # 默认放置在屏幕顶部居中偏下位置
        desktop = QApplication.desktop()
        rect = desktop.availableGeometry()
        self.center_x = (rect.width() - self.bar_w) // 2
        self.center_y = 35
        self.move(self.center_x, self.center_y)

        # 鼠标拖动与交互状态管理
        self.drag_start_pos = None
        self.is_dragging = False
        self.is_hovered = False

        # 应用 Windows 底层无焦点属性，彻底保证鼠标点击不抢走输入框焦点
        self.apply_no_activate()

        # 30fps 定时器驱动呼吸灯动画与状态同步
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_state)
        self.timer.start(30)

        self.show()
        print("✅ Qwen3 悬浮小条 (Floating Bar) 已加载就绪，可随时鼠标点击或快捷键呼出。")

    def apply_no_activate(self):
        # 赋予窗口 WS_EX_NOACTIVATE 属性，使鼠标点击窗口时不激活窗口，从而保持光标焦点在原文本框
        try:
            hwnd = int(self.winId())
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE)
        except Exception as e:
            print(f"⚠️ 设置 WS_EX_NOACTIVATE 属性提示: {e}")

    def showEvent(self, event):
        super().showEvent(event)
        self.apply_no_activate()

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
            # 记录点击瞬间的前台窗口，作为输入焦点双保险
            global target_input_hwnd
            fg = ctypes.windll.user32.GetForegroundWindow()
            if fg and fg != int(self.winId()):
                target_input_hwnd = fg

            self.drag_start_pos = event.globalPos() - self.frameGeometry().topLeft()
            self.is_dragging = False
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_start_pos is not None:
            self.is_dragging = True
            new_pos = event.globalPos() - self.drag_start_pos
            self.move(new_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 如果位移极小，判定为正常单击，触发切换录音
            if not self.is_dragging:
                toggle_recording()
            self.drag_start_pos = None
            self.is_dragging = False

    def show_context_menu(self, global_pos):
        menu = QMenu(self)

        act_theme = QAction("🎨 切换主题 (明亮/暗黑)", menu)
        def on_toggle_theme():
            self.theme = "light" if self.theme == "dark" else "dark"
            self.update()
        act_theme.triggered.connect(on_toggle_theme)
        menu.addAction(act_theme)

        act_center = QAction("📌 恢复顶部居中", menu)
        def on_center():
            desktop = QApplication.desktop()
            rect = desktop.availableGeometry()
            self.move((rect.width() - self.bar_w) // 2, 35)
        act_center.triggered.connect(on_center)
        menu.addAction(act_center)

        menu.addSeparator()
        act_exit = QAction("❌ 完全退出", menu)
        act_exit.triggered.connect(lambda: os._exit(0))
        menu.addAction(act_exit)

        menu.exec_(global_pos)

    def update_state(self):
        # 持续触发界面局部重绘以呈现呼吸灯和状态平滑过渡
        self.update()

    def paintEvent(self, event):
        global is_recording, recording_start, MAX_RECORD_SECONDS, realtime_text
        global status_banner, status_banner_expire

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(1.5, 1.5, self.bar_w - 3.0, self.bar_h - 3.0)
        radius = (self.bar_h - 3.0) / 2.0

        # 当前是否处于横幅提示阶段 (例如 "⚡ 正在极速识别..." 或 "✅ 已上屏")
        active_banner = ""
        if status_banner and (time.time() < status_banner_expire or "识别" in status_banner):
            active_banner = status_banner

        # 主题色彩适配
        if self.theme == "dark":
            if is_recording:
                # 录音中：深红暗夜磨砂背景
                bg_color = QColor(38, 20, 26, 230)
                border_color = QColor(255, 75, 75, 200)
            elif active_banner:
                # 识别与反馈态：紫蓝色微光背景
                bg_color = QColor(28, 24, 46, 230)
                border_color = QColor(140, 90, 255, 200)
            else:
                # 待命态：半透明极客黑
                bg_color = QColor(22, 24, 32, 220)
                border_color = QColor(0, 210, 255, 160) if self.is_hovered else QColor(255, 255, 255, 38)
            text_color = QColor(245, 248, 255)
            subtext_color = QColor(160, 175, 200)
        else:
            if is_recording:
                bg_color = QColor(255, 235, 238, 235)
                border_color = QColor(245, 60, 60, 210)
            elif active_banner:
                bg_color = QColor(242, 238, 255, 235)
                border_color = QColor(120, 70, 240, 210)
            else:
                bg_color = QColor(250, 252, 255, 230)
                border_color = QColor(0, 140, 255, 160) if self.is_hovered else QColor(180, 190, 205, 120)
            text_color = QColor(25, 30, 45)
            subtext_color = QColor(100, 110, 130)

        # 绘制胶囊背景与边框
        painter.setBrush(bg_color)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawRoundedRect(rect, radius, radius)

        # 绘制左侧状态指示器 (待命为青色麦克风小圆点，录音中为动态呼吸红点)
        btn_center_x = 24.0
        btn_center_y = self.bar_h / 2.0

        if is_recording:
            # 录音中：脉冲呼吸红点
            pulse = 0.5 + 0.5 * math.sin(time.time() * 7)
            glow_radius = 8.0 + pulse * 4.0
            painter.setBrush(QColor(255, 50, 50, int(60 + pulse * 100)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(btn_center_x), int(btn_center_y)), int(glow_radius), int(glow_radius))

            painter.setBrush(QColor(255, 60, 60))
            painter.drawEllipse(QPoint(int(btn_center_x), int(btn_center_y)), 5, 5)

            # 计算录音计时
            elapsed = time.time() - recording_start
            if elapsed > MAX_RECORD_SECONDS:
                elapsed = MAX_RECORD_SECONDS

            # 显示录音计时
            painter.setPen(text_color)
            font_time = QFont("Segoe UI", 10, QFont.Bold)
            painter.setFont(font_time)
            time_str = f"00:{int(elapsed):02d}"
            painter.drawText(QRectF(44, 0, 50, self.bar_h), Qt.AlignVCenter | Qt.AlignLeft, time_str)

            # 显示动态提示或实时转写
            font_hint = QFont("Microsoft YaHei UI", 9)
            painter.setFont(font_hint)
            painter.setPen(subtext_color)
            display_text = realtime_text if (realtime_text and "Qwen3" not in realtime_text) else "录音中 · 点击停止上屏"
            painter.drawText(QRectF(100, 0, self.bar_w - 145, self.bar_h), Qt.AlignVCenter | Qt.AlignLeft, display_text)

            # 右侧停止方块图标
            stop_btn_rect = QRectF(self.bar_w - 34, (self.bar_h - 18) / 2.0, 18, 18)
            painter.setBrush(QColor(255, 75, 75, 210))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(stop_btn_rect, 4, 4)

        elif active_banner:
            # 正在识别或上屏反馈
            painter.setBrush(QColor(140, 80, 255))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(btn_center_x), int(btn_center_y)), 5, 5)

            painter.setPen(text_color)
            font_banner = QFont("Microsoft YaHei UI", 10, QFont.Bold)
            painter.setFont(font_banner)
            painter.drawText(QRectF(44, 0, self.bar_w - 55, self.bar_h), Qt.AlignVCenter | Qt.AlignLeft, active_banner)

        else:
            # 待命闲置状态：麦克风小圆点 + 点击录音提示
            painter.setBrush(QColor(0, 210, 255) if self.theme == "dark" else QColor(0, 140, 255))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(btn_center_x), int(btn_center_y)), 6, 6)

            # 主提示文字
            painter.setPen(text_color)
            font_title = QFont("Microsoft YaHei UI", 10, QFont.Bold)
            painter.setFont(font_title)
            painter.drawText(QRectF(44, 0, 115, self.bar_h), Qt.AlignVCenter | Qt.AlignLeft, "🎙️ 点击录音")

            # 快捷键副标题
            painter.setPen(subtext_color)
            font_shortcut = QFont("Segoe UI", 9)
            painter.setFont(font_shortcut)
            painter.drawText(QRectF(160, 0, self.bar_w - 175, self.bar_h), Qt.AlignVCenter | Qt.AlignRight, "Win+Shift+H")

def create_tray_icon(app, floating_bar):
    # 构建托盘图标与右键菜单，方便在无控制台模式下管理与退出程序
    tray = QSystemTrayIcon()
    pix = QPixmap(32, 32)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(0, 200, 255))
    p.setPen(Qt.NoPen)
    p.drawEllipse(4, 4, 24, 24)
    p.end()
    tray.setIcon(QIcon(pix))
    tray.setToolTip("VibeC 语音助手 (Qwen3-ASR 加速版)")

    menu = QMenu()

    def toggle_bar_visibility():
        if floating_bar.isVisible():
            floating_bar.hide()
        else:
            floating_bar.show()
    act_toggle = QAction("👁️ 显示/隐藏悬浮条", menu)
    act_toggle.triggered.connect(toggle_bar_visibility)
    menu.addAction(act_toggle)

    act_info = QAction("📝 使用说明", menu)
    def show_info():
        QMessageBox.information(
            None,
            "使用说明",
            "【VibeC - Qwen3-ASR 语音输入法】\n\n"
            "1. 鼠标点击桌面顶部的【悬浮小条】或单击快捷键 Win+Shift+H 开始录音\n"
            "2. 讲话完成后再次点击悬浮小条或单击 Win+Shift+H，自动上屏\n"
            "3. 鼠标可随意按住悬浮小条拖动到屏幕任意位置\n"
            "4. 按快捷键 Ctrl+Shift+Q 即可彻底退出程序"
        )
    act_info.triggered.connect(show_info)
    menu.addAction(act_info)

    def toggle_theme():
        floating_bar.theme = "light" if floating_bar.theme == "dark" else "dark"
        floating_bar.update()
    act_theme = QAction("🎨 切换主题", menu)
    act_theme.triggered.connect(toggle_theme)
    menu.addAction(act_theme)

    menu.addSeparator()
    act_exit = QAction("❌ 完全退出", menu)
    def on_exit():
        tray.hide()
        os._exit(0)
    act_exit.triggered.connect(on_exit)
    menu.addAction(act_exit)

    tray.setContextMenu(menu)
    tray.show()
    return tray

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    floating_bar = FloatingBarUI()
    tray = create_tray_icon(app, floating_bar)

    t = threading.Thread(target=background_task, daemon=True)
    t.start()

    sys.exit(app.exec_())
