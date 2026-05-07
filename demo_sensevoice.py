import time
import pyaudio
import keyboard
import numpy as np
import sherpa_onnx
import os
import sys
import winsound
import threading
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from PyQt5.QtWidgets import QApplication, QWidget, QSystemTrayIcon, QMenu, QAction, QMessageBox
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer, QRectF

# ================= 配置参数 =================
MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
HOTKEY = "windows+h"
EXIT_HOTKEY = "ctrl+shift+q"

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_RECORD_SECONDS = 60.0
# 麦克风设备索引：None 为自动选择（启动时会列出所有设备）。
# 如果声音全是静音，请把下面的 None 改为正确的设备编号
MIC_DEVICE_INDEX = None
# ============================================

# 全局状态变量，用于在后台线程和 UI 线程之间通信
is_recording = False
recording_start = 0
realtime_text = ""

def get_application_path():
    """获取程序运行时的绝对目录，兼容 PyInstaller 打包后的 exe"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_mic_volume_interface():
    """获取 Windows 默认麦克风的音量控制接口"""
    try:
        # 正确的 pycaw API: GetMicrophone() 获取默认录音设备
        device = AudioUtilities.GetMicrophone()
        if device is None:
            print("⚠️ 未找到默认麦克风设备")
            return None
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as e:
        print(f"⚠️ 无法访问麦克风音量控制: {e}")
        return None

def get_mic_mute() -> bool:
    """获取当前麦克风的静音状态"""
    vol = get_mic_volume_interface()
    return bool(vol.GetMute()) if vol else False

def set_mic_mute(mute: bool):
    """设置麦克风静音状态"""
    vol = get_mic_volume_interface()
    if vol:
        vol.SetMute(1 if mute else 0, None)
        print(f"🎙️ 麦克风静音已{'开启' if mute else '关闭'}")

def find_mic_device():
    """枚举并打印所有麦克风设备，返回应使用的设备索引"""
    p = pyaudio.PyAudio()
    print("\n===== 可用麦克风设备列表 =====")
    input_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info['name']))
            marker = " <-- 当前默认" if i == p.get_default_input_device_info()['index'] else ""
            print(f"  [{i}] {info['name']}{marker}")
    p.terminate()
    
    if MIC_DEVICE_INDEX is not None:
        print(f"\n✅ 使用配置的设备索引: {MIC_DEVICE_INDEX}")
        return MIC_DEVICE_INDEX
    
    default_idx = None
    p2 = pyaudio.PyAudio()
    try:
        default_idx = p2.get_default_input_device_info()['index']
    except:
        pass
    p2.terminate()
    
    print(f"\n⚠️ 正在使用系统默认设备 [索引 {default_idx}]")
    print("⚠️ 如果识别结果RMS始终为 0.0000，请修改脚本头部的 MIC_DEVICE_INDEX 为正确的设备编号")
    print("============================\n")
    return default_idx

def toggle_recording():
    """快捷键回调函数：切换录音状态"""
    global is_recording
    is_recording = not is_recording

def background_task():
    """后台任务：监听快捷键、录音以及 AI 推理"""
    global is_recording, recording_start
    
    # 注册全局快捷键并拦截系统原生按键事件 (suppress=True 代表彻底屏蔽 Windows 自身的 Win+H 语音输入面板)
    keyboard.add_hotkey(HOTKEY, toggle_recording, suppress=True)
    
    # 启动时先确认麦克风设备
    mic_device_idx = find_mic_device()
    
    app_path = get_application_path()
    model_dir = os.path.join(app_path, MODEL_DIR_NAME)
    model_file = os.path.join(model_dir, "model.onnx")
    tokens_file = os.path.join(model_dir, "tokens.txt")
    
    if not os.path.exists(model_file):
        print(f"❌ 找不到模型文件: {model_file}")
        os._exit(1)

    print(f"🔄 正在加载 SenseVoice 模型...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_file,
        tokens=tokens_file,
        num_threads=1,
        use_itn=True, 
        # provider="dml", 
    )
    print("✅ SenseVoice 加载成功！")
    print(f"👉 单击【{HOTKEY}】开始录音，再说一下结束 (已成功替代 Windows 原生面板)。")
    print(f"👉 按下【{EXIT_HOTKEY}】键退出后台程序。")
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

                # 录音前：保存原始静音状态，然后强制取消静音
                original_mute_state = get_mic_mute()
                if original_mute_state:
                    print("🎙️ 检测到麦克风被静音，自动取消静音...")
                    set_mic_mute(False)

                global realtime_text
                realtime_text = "🎤 等待语音输入..."
                last_infer_time = time.time()
                mute_warned = False

                # 持续录音，直到再次按下快捷键，或者达到 60 秒上限
                while is_recording and (time.time() - recording_start) <= MAX_RECORD_SECONDS:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)

                    # 每隔 0.4 秒做一次伪实时推理
                    if time.time() - last_infer_time > 0.4 and len(frames) > 5:
                        raw_tmp = b''.join(frames)
                        audio_tmp = np.frombuffer(raw_tmp, dtype=np.int16).astype(np.float32) / 32768.0

                        # ---- 麦克风静音检测 ----
                        rms = float(np.sqrt(np.mean(audio_tmp**2)))
                        if rms < 0.001:
                            if not mute_warned:
                                realtime_text = "⚠️ 麦克风无声音！\n请检查是否被 Mute"
                                mute_warned = True
                            last_infer_time = time.time()
                            continue  # 静音就跳过推理

                        mute_warned = False  # 有声音则清除警告
                        c_stream = recognizer.create_stream()
                        c_stream.accept_waveform(RATE, audio_tmp)
                        recognizer.decode_stream(c_stream)
                        if c_stream.result.text:
                            realtime_text = c_stream.result.text
                        last_infer_time = time.time()

                # 确保状态被重置（处理 60 秒超时自动停止的情况）
                is_recording = False
                winsound.Beep(1000, 100)
                stream.stop_stream()
                stream.close()
                p.terminate()

                # 录音结束后：恢复麦克风原始静音状态
                if original_mute_state:
                    set_mic_mute(True)
                    print("🎙️ 已恢复麦克风静音状态")

                # 最终完整推理（用于最精确的上屏结果）
                if len(frames) > 5:
                    raw_data = b''.join(frames)
                    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    rms_final = float(np.sqrt(np.mean(audio_float32**2)))
                    print(f"🔍 [诊断] 帧数:{len(frames)}, 时长:{len(frames)*CHUNK/RATE:.1f}s, RMS:{rms_final:.4f}")
                    if rms_final > 0.001:
                        c_stream = recognizer.create_stream()
                        c_stream.accept_waveform(RATE, audio_float32)
                        recognizer.decode_stream(c_stream)
                        text = c_stream.result.text
                        print(f"📝 [最终结果]: '{text}'")
                        if text:
                            keyboard.release('windows')
                            keyboard.release('h')
                            realtime_text = text  # 字幕板展示最终结果
                            keyboard.write(text)
                            keyboard.write(" ")
                            time.sleep(1.0)
                    else:
                        realtime_text = "⚠️ 麦克风无声音！\n请检查是否被 Mute"
                        time.sleep(2.0)

                realtime_text = ""
                time.sleep(0.3)


                
            if keyboard.is_pressed(EXIT_HOTKEY):
                winsound.Beep(400, 300)
                os._exit(0) # 彻底退出整个程序
                
            time.sleep(0.05) 
            
        except KeyboardInterrupt:
            os._exit(0)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

class OverlayUI(QWidget):
    def __init__(self):
        super().__init__()
        self.theme = "light" # 默认暗色主题 
        # 移除边框、置顶、不在任务栏显示
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        # 允许真正的背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.w_px = 550
        self.h_px = 350
        self.setFixedSize(self.w_px, self.h_px)
        
        # 居中显示
        desktop = QApplication.desktop()
        rect = desktop.availableGeometry()
        self.move((rect.width() - self.w_px) // 2, (rect.height() - self.h_px) // 2)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)
        
    def update_ui(self):
        global is_recording
        if is_recording:
            if self.isHidden():
                self.show()
            self.update() # 触发重绘
        else:
            if not self.isHidden():
                self.hide()

    def paintEvent(self, event):
        global is_recording, recording_start, MAX_RECORD_SECONDS
        if not is_recording:
            return
            
        elapsed = time.time() - recording_start
        if elapsed > MAX_RECORD_SECONDS:
            elapsed = MAX_RECORD_SECONDS
            
        painter = QPainter(self)
        # 开启完美的抗锯齿
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算圆环的位置，使其水平居中并靠上
        dial_size = 220
        offset_x = (self.w_px - dial_size) / 2
        offset_y = 10
        
        padding = 20
        rect = QRectF(offset_x + padding, offset_y + padding, dial_size - padding*2, dial_size - padding*2)
        dial_rect = QRectF(offset_x, offset_y, dial_size, dial_size)
        
        # 根据当前主题动态配置颜色
        if self.theme == "dark":
            c_bg = QColor(50, 50, 60, 160)
            c_grad_start = QColor(0, 255, 204, 255)  # 青色
            c_grad_end = QColor(153, 51, 255, 255)   # 紫色
            c_text = QColor(255, 255, 255, 255)
            c_shadow = QColor(0, 0, 0, 160)
            c_subtext = QColor(255, 255, 255, 200)
        else:
            c_bg = QColor(240, 240, 245, 200)        # 明亮模式：浅灰白底色
            c_grad_start = QColor(0, 102, 255, 255)  # 深海蓝
            c_grad_end = QColor(0, 204, 102, 255)    # 翡翠绿
            c_text = QColor(30, 30, 40, 255)         # 深色文字
            c_shadow = QColor(255, 255, 255, 200)    # 明亮模式下的白色高光/阴影
            c_subtext = QColor(80, 80, 90, 220)

        # 绘制底部背景圆环
        pen_bg = QPen(c_bg, 14) 
        pen_bg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(rect, 0, 360 * 16)
        
        # 创建现代感渐变色
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, c_grad_start)
        gradient.setColorAt(1.0, c_grad_end)
        
        # 绘制进度渐变圆弧
        pen_fg = QPen(gradient, 14)
        pen_fg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_fg)
        
        # startAngle: 90度(12点钟方向), spanAngle: 负数代表顺时针
        start_angle = 90 * 16
        span_angle = -int((elapsed / MAX_RECORD_SECONDS) * 360 * 16)
        painter.drawArc(rect, start_angle, span_angle)
        
        # --- 绘制中心时间文字 ---
        font = QFont("Segoe UI", 32, QFont.Bold)
        painter.setFont(font)
        text = f"00:{int(elapsed):02d}"
        
        # 1. 阴影/高光层 (偏移 2 像素)
        painter.setPen(c_shadow)
        shadow_rect = dial_rect.adjusted(2, 2, 2, 2)
        painter.drawText(shadow_rect, Qt.AlignCenter, text)
        
        # 2. 本体层
        painter.setPen(c_text)
        painter.drawText(dial_rect, Qt.AlignCenter, text)
        
        # --- 绘制底部小字 ---
        font_small = QFont("Segoe UI", 10)
        font_small.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)
        painter.setFont(font_small)
        
        text_rect_shadow = dial_rect.adjusted(2, 62, 2, 2)
        painter.setPen(c_shadow)
        painter.drawText(text_rect_shadow, Qt.AlignCenter, "MAX 01:00")
        
        text_rect = dial_rect.adjusted(0, 60, 0, 0)
        painter.setPen(c_subtext)
        painter.drawText(text_rect, Qt.AlignCenter, "MAX 01:00")
        
        # --- 绘制实时悬浮字幕板 ---
        global realtime_text
        if realtime_text:
            box_margin = 30
            box_y = offset_y + dial_size + 10
            box_h = self.h_px - box_y - 20
            box_rect = QRectF(box_margin, box_y, self.w_px - box_margin*2, box_h)
            
            # 画一个带磨砂质感的半透明圆角矩形底板
            box_color = QColor(30, 30, 40, 180) if self.theme == "dark" else QColor(240, 240, 245, 220)
            painter.setBrush(box_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(box_rect, 15, 15)
            
            # 绘制字幕文字
            painter.setPen(c_text)
            subtitle_font = QFont("Microsoft YaHei", 14) # 微软雅黑，适合展示中文
            painter.setFont(subtitle_font)
            text_rect_inner = box_rect.adjusted(20, 15, -20, -15)
            painter.drawText(text_rect_inner, Qt.AlignCenter | Qt.TextWordWrap, realtime_text)

def create_tray_icon(app, overlay):
    """创建系统托盘图标和右键菜单"""
    # 强制应用程序在所有窗口隐藏时(表盘平时是隐藏的)不退出
    app.setQuitOnLastWindowClosed(False)
    
    tray_icon = QSystemTrayIcon(app)
    
    # 代码动态生成一个简单的托盘图标（青色圆球），避免打包依赖外部图片文件
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#00FFCC"))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(4, 4, 56, 56)
    painter.end()
    tray_icon.setIcon(QIcon(pixmap))
    
    menu = QMenu()
    
    # 菜单1：使用说明
    action_help = QAction("📝 使用说明", menu)
    def show_help():
        msg = QMessageBox()
        msg.setWindowTitle("使用说明")
        msg.setText(f"🚀 Web Coding 极速语音助手\n\n1. 单击键盘 [ {HOTKEY.upper()} ] 唤醒表盘并开始录音。\n2. 说完后再次单击结束。\n3. 程序会瞬间进行 AI 推理，并在光标处自动打字。\n\n如需强制退出，请点击此处的'完全退出'。")
        msg.setIcon(QMessageBox.Information)
        msg.setWindowFlags(Qt.WindowStaysOnTopHint)
        msg.exec_()
    action_help.triggered.connect(show_help)
    menu.addAction(action_help)
    
    # 菜单2：切换主题
    action_theme = QAction("🎨 切换主题 (明亮/黑暗)", menu)
    def toggle_theme():
        if overlay.theme == "dark":
            overlay.theme = "light"
            tray_icon.showMessage("主题切换", "已切换至【明亮模式】", QSystemTrayIcon.Information, 1000)
        else:
            overlay.theme = "dark"
            tray_icon.showMessage("主题切换", "已切换至【黑暗模式】", QSystemTrayIcon.Information, 1000)
    action_theme.triggered.connect(toggle_theme)
    menu.addAction(action_theme)
    
    menu.addSeparator()
    
    # 菜单3：退出
    action_exit = QAction("❌ 完全退出", menu)
    def exit_app():
        os._exit(0)
    action_exit.triggered.connect(exit_app)
    menu.addAction(action_exit)
    
    tray_icon.setContextMenu(menu)
    tray_icon.show()
    return tray_icon

def main():
    # 启动后台监听与录音线程
    t = threading.Thread(target=background_task, daemon=True)
    t.start()
    
    # 启动 PyQt 的主事件循环
    app = QApplication(sys.argv)
    overlay = OverlayUI()
    
    # 创建并挂载系统托盘 (必须把对象保存下来)
    tray = create_tray_icon(app, overlay)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
