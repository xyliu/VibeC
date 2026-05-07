import time
import pyaudio
import keyboard
import numpy as np
import sherpa_onnx
import os
import sys
import winsound
import threading

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
# ============================================

# 全局状态变量，用于在后台线程和 UI 线程之间通信
is_recording = False
recording_start = 0

def get_application_path():
    """获取程序运行时的绝对目录，兼容 PyInstaller 打包后的 exe"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def toggle_recording():
    """快捷键回调函数：切换录音状态"""
    global is_recording
    is_recording = not is_recording

def background_task():
    """后台任务：监听快捷键、录音以及 AI 推理"""
    global is_recording, recording_start
    
    # 注册全局快捷键并拦截系统原生按键事件 (suppress=True 代表彻底屏蔽 Windows 自身的 Win+H 语音输入面板)
    keyboard.add_hotkey(HOTKEY, toggle_recording, suppress=True)
    
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
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

                # 持续录音，直到再次按下快捷键(改变了 is_recording 状态)，或者达到 60 秒上限
                while is_recording and (time.time() - recording_start) <= MAX_RECORD_SECONDS:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)

                # 确保状态被重置（处理 60 秒超时自动停止的情况）
                is_recording = False
                winsound.Beep(1000, 100)
                stream.stop_stream()
                stream.close()
                p.terminate()

                # 只有录到了有效音频才进行推理
                if len(frames) > 5:
                    raw_data = b''.join(frames)
                    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    start_time = time.time()
                    c_stream = recognizer.create_stream()
                    c_stream.accept_waveform(RATE, audio_float32)
                    recognizer.decode_stream(c_stream)
                    
                    text = c_stream.result.text
                    elapsed = time.time() - start_time
                    print(f"📝 [识别结果] (耗时 {elapsed:.2f}s): {text}")
                    
                    if text:
                        # 确保物理修饰键完全松开，防止注入文本时误触发系统级快捷键 (比如 Win + e 变成打开资源管理器)
                        keyboard.release('windows')
                        keyboard.release('h')
                        keyboard.write(text)
                        keyboard.write(" ")
                        
                time.sleep(0.3) # 防止推理结束后瞬间误触
                
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
        self.theme = "dark" # 默认暗色主题
        # 移除边框、置顶、不在任务栏显示
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        # 允许真正的背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.size_px = 220
        self.setFixedSize(self.size_px, self.size_px)
        
        # 居中显示
        desktop = QApplication.desktop()
        rect = desktop.availableGeometry()
        self.move((rect.width() - self.size_px) // 2, (rect.height() - self.size_px) // 2)
        
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
        
        padding = 20
        rect = QRectF(padding, padding, self.size_px - padding*2, self.size_px - padding*2)
        
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
        shadow_rect = self.rect().adjusted(2, 2, 2, 2)
        painter.drawText(shadow_rect, Qt.AlignCenter, text)
        
        # 2. 本体层
        painter.setPen(c_text)
        painter.drawText(self.rect(), Qt.AlignCenter, text)
        
        # --- 绘制底部小字 ---
        font_small = QFont("Segoe UI", 10)
        font_small.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)
        painter.setFont(font_small)
        
        text_rect_shadow = self.rect().adjusted(2, 62, 2, 2)
        painter.setPen(c_shadow)
        painter.drawText(text_rect_shadow, Qt.AlignCenter, "MAX 01:00")
        
        text_rect = self.rect().adjusted(0, 60, 0, 0)
        painter.setPen(c_subtext)
        painter.drawText(text_rect, Qt.AlignCenter, "MAX 01:00")

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
