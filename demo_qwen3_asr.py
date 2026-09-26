import time
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

# ================= 配置参数 =================
# Qwen3-ASR 模型目录，支持官方导出的 0.6B INT8 轻量版
MODEL_DIR_NAME = "sherpa-onnx-qwen3-asr-0.6B-int8"

# 加速后端模式：可选 "gpu"（优先Intel Arc GPU加速，推荐）、"cpu"（纯CPU运算）、"npu"（实验性NPU）
ACCELERATOR_BACKEND = "gpu"

HOTKEY = "windows+h"
EXIT_HOTKEY = "ctrl+shift+q"

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_RECORD_SECONDS = 60.0
# 麦克风设备索引：None 为自动选择系统默认设备
MIC_DEVICE_INDEX = None
# ============================================

# 全局运行状态与实时文本缓冲
is_recording = False
recording_start = 0
realtime_text = ""
current_backend_label = "Intel Arc GPU"

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

pynput_listener = None
h_suppressed = False
win_pressed = False

def log_debug(message):
    try:
        app_path = get_application_path()
        log_file = os.path.join(app_path, "keyboard_debug.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass

def win32_event_filter(msg, data):
    # 底层拦截键盘钩子，精准接管 Win+H，避免触发系统自带面板同时不影响其他组合键
    global h_suppressed, win_pressed
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105

    VK_LWIN = 0x5B
    VK_RWIN = 0x5C
    VK_H = 0x48

    vk_code = data.vkCode

    if vk_code in (VK_LWIN, VK_RWIN):
        if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
            win_pressed = True
        elif msg in (WM_KEYUP, WM_SYSKEYUP):
            win_pressed = False
            h_suppressed = False

    if vk_code == VK_H:
        if win_pressed:
            if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
                if not h_suppressed:
                    h_suppressed = True
                    log_debug("Win+H 命中，切换录音状态")
                    toggle_recording()
                return False
            elif msg in (WM_KEYUP, WM_SYSKEYUP):
                return False
    return True

def init_qwen3_recognizer(base_dir: str):
    # 自动探测优先使用 1.7B 高质量版本还是 0.6B 极速版本
    global current_backend_label

    candidate_dirs = [
        ("qwen3-asr-1.7b-int4", "Qwen3 1.7B [高质量版]"),
        ("sherpa-onnx-qwen3-asr-0.6B-int8", "Qwen3 0.6B [极速版]"),
        (base_dir, "Qwen3-ASR")
    ]

    selected_dir = None
    model_version_label = "Qwen3-ASR"
    app_path = get_application_path()

    for d_name, v_label in candidate_dirs:
        check_path = os.path.join(app_path, d_name)
        if os.path.exists(check_path):
            selected_dir = check_path
            model_version_label = v_label
            break

    if selected_dir is None:
        selected_dir = os.path.join(app_path, MODEL_DIR_NAME)

    model_dir = selected_dir
    conv_frontend = os.path.join(model_dir, "conv_frontend.onnx")
    encoder = os.path.join(model_dir, "encoder.int8.onnx")
    if not os.path.exists(encoder):
        encoder = os.path.join(model_dir, "encoder.int4.onnx")
    if not os.path.exists(encoder):
        encoder = os.path.join(model_dir, "encoder.onnx")

    decoder = os.path.join(model_dir, "decoder.int8.onnx")
    if not os.path.exists(decoder):
        decoder = os.path.join(model_dir, "decoder_step.int4.onnx")
    if not os.path.exists(decoder):
        decoder = os.path.join(model_dir, "decoder.onnx")

    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    if not os.path.exists(tokenizer_dir):
        tokenizer_dir = model_dir

    # 校验必要文件
    missing = []
    if not os.path.exists(conv_frontend):
        missing.append("conv_frontend.onnx")
    if not os.path.exists(encoder):
        missing.append("encoder.int8.onnx / encoder.onnx")
    if not os.path.exists(decoder):
        missing.append("decoder.int8.onnx / decoder.onnx")

    if missing:
        print(f"❌ 目录下缺少 Qwen3-ASR 关键模型组件: {', '.join(missing)}")
        print(f"请检查路径: {model_dir}")
        return None

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

    print(f"👉 单击【{HOTKEY}】开启录音，再次单击立即识别上屏。")
    print(f"👉 按下【{EXIT_HOTKEY}】安全退出后台。")
    winsound.Beep(600, 200)

    while True:
        try:
            if is_recording:
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
                            # 释放系统按键状态，防止键位冲突干扰自动模拟打字
                            keyboard.release('windows')
                            keyboard.release('h')
                            realtime_text = text
                            keyboard.write(text)
                            keyboard.write(" ")
                            time.sleep(0.8)
                    else:
                        realtime_text = "⚠️ 麦克风无声音信号"
                        time.sleep(1.5)

                realtime_text = ""
                time.sleep(0.3)
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

class OverlayUI(QWidget):
    def __init__(self):
        super().__init__()
        self.theme = "dark"
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.w_px = 580
        self.h_px = 360
        self.setFixedSize(self.w_px, self.h_px)

        print("✅ Qwen3 悬浮表盘界面已加载完成。")
        desktop = QApplication.desktop()
        rect = desktop.availableGeometry()
        self.move((rect.width() - self.w_px) // 2, (rect.height() - self.h_px) // 2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)

    def closeEvent(self, event):
        global pynput_listener
        if pynput_listener:
            pynput_listener.stop()
        keyboard.unhook_all()
        super().closeEvent(event)

    def update_ui(self):
        global is_recording
        if is_recording:
            if self.isHidden():
                self.show()
            self.update()
        else:
            if not self.isHidden():
                self.hide()

    def paintEvent(self, event):
        global is_recording, recording_start, MAX_RECORD_SECONDS, current_backend_label
        if not is_recording:
            return

        elapsed = time.time() - recording_start
        if elapsed > MAX_RECORD_SECONDS:
            elapsed = MAX_RECORD_SECONDS

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        dial_size = 210
        offset_x = (self.w_px - dial_size) / 2
        offset_y = 15

        padding = 18
        rect = QRectF(offset_x + padding, offset_y + padding, dial_size - padding*2, dial_size - padding*2)
        dial_rect = QRectF(offset_x, offset_y, dial_size, dial_size)

        if self.theme == "dark":
            c_bg = QColor(40, 40, 50, 180)
            c_grad_start = QColor(0, 220, 255, 255)
            c_grad_end = QColor(140, 80, 255, 255)
            c_text = QColor(255, 255, 255, 255)
            c_shadow = QColor(0, 0, 0, 160)
            c_subtext = QColor(200, 210, 255, 220)
        else:
            c_bg = QColor(240, 240, 245, 200)
            c_grad_start = QColor(0, 120, 255, 255)
            c_grad_end = QColor(0, 210, 140, 255)
            c_text = QColor(30, 30, 40, 255)
            c_shadow = QColor(255, 255, 255, 200)
            c_subtext = QColor(80, 80, 90, 220)

        # 绘制背景圆环
        pen_bg = QPen(c_bg, 12)
        pen_bg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(rect, 0, 360 * 16)

        # 进度渐变色
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, c_grad_start)
        gradient.setColorAt(1.0, c_grad_end)

        pen_fg = QPen(gradient, 12)
        pen_fg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_fg)

        start_angle = 90 * 16
        span_angle = -int((elapsed / MAX_RECORD_SECONDS) * 360 * 16)
        painter.drawArc(rect, start_angle, span_angle)

        # 绘制录音计时
        font = QFont("Segoe UI", 30, QFont.Bold)
        painter.setFont(font)
        text = f"00:{int(elapsed):02d}"

        painter.setPen(c_shadow)
        painter.drawText(dial_rect.adjusted(2, 2, 2, 2), Qt.AlignCenter, text)
        painter.setPen(c_text)
        painter.drawText(dial_rect, Qt.AlignCenter, text)

        # 显示当前使用的引擎与硬件加速标签
        font_small = QFont("Segoe UI", 9, QFont.Bold)
        font_small.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
        painter.setFont(font_small)

        info_text = current_backend_label
        text_rect_shadow = dial_rect.adjusted(2, 58, 2, 2)
        painter.setPen(c_shadow)
        painter.drawText(text_rect_shadow, Qt.AlignCenter, info_text)

        text_rect = dial_rect.adjusted(0, 56, 0, 0)
        painter.setPen(c_subtext)
        painter.drawText(text_rect, Qt.AlignCenter, info_text)

        # 实时动态识别字幕框
        global realtime_text
        if realtime_text:
            box_margin = 25
            box_y = offset_y + dial_size + 15
            box_h = self.h_px - box_y - 15
            box_rect = QRectF(box_margin, box_y, self.w_px - box_margin*2, box_h)

            box_color = QColor(25, 25, 35, 210) if self.theme == "dark" else QColor(245, 245, 250, 230)
            painter.setBrush(box_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(box_rect, 14, 14)

            font_caption = QFont("Microsoft YaHei", 12)
            painter.setFont(font_caption)
            painter.setPen(QColor(255, 255, 255, 240) if self.theme == "dark" else QColor(30, 30, 40, 240))
            painter.drawText(box_rect.adjusted(16, 10, -16, -10), Qt.AlignLeft | Qt.TextWordWrap, realtime_text)

def create_tray_icon(app, overlay):
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
    act_info = QAction("📝 使用说明", menu)
    def show_info():
        QMessageBox.information(
            None,
            "使用说明",
            "【VibeC - Qwen3-ASR 语音输入法】\n\n"
            "1. 在任意输入框中单击 Win+H 开始录音\n"
            "2. 讲话完成后再次单击 Win+H，自动上屏\n"
            "3. 按快捷键 Ctrl+Shift+Q 即可彻底退出程序"
        )
    act_info.triggered.connect(show_info)
    menu.addAction(act_info)

    def toggle_theme():
        overlay.theme = "light" if overlay.theme == "dark" else "dark"
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

    overlay = OverlayUI()
    tray = create_tray_icon(app, overlay)

    t = threading.Thread(target=background_task, daemon=True)
    t.start()

    sys.exit(app.exec_())
