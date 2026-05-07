import time
import pyaudio
import keyboard
import numpy as np
import sherpa_onnx
import os
import sys
import winsound
import threading

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient
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
        
        # 绘制底部背景圆环 (提高不透明度，使用深色半透明防止在浅色背景看不清)
        # QColor 参数为 (R, G, B, Alpha透明度：255为完全不透明)
        pen_bg = QPen(QColor(50, 50, 60, 160), 14) 
        pen_bg.setCapStyle(Qt.RoundCap) # 圆角线头
        painter.setPen(pen_bg)
        painter.drawArc(rect, 0, 360 * 16) # Qt 中的角度单位是 1/16 度
        
        # 创建现代感渐变色 (青色到紫色的赛博朋克渐变，提升纯度与完全不透明度)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor(0, 255, 204, 255))   # 亮青色，完全不透明
        gradient.setColorAt(1.0, QColor(153, 51, 255, 255))  # 亮紫色，完全不透明
        
        # 绘制进度渐变圆弧
        pen_fg = QPen(gradient, 14)
        pen_fg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_fg)
        
        # startAngle: 90度(12点钟方向)
        # spanAngle: 负数代表顺时针
        start_angle = 90 * 16
        span_angle = -int((elapsed / MAX_RECORD_SECONDS) * 360 * 16)
        painter.drawArc(rect, start_angle, span_angle)
        
        # --- 绘制中心时间文字 ---
        font = QFont("Segoe UI", 32, QFont.Bold) # 字体微调大一点，显得现代
        painter.setFont(font)
        text = f"00:{int(elapsed):02d}"
        
        # 1. 画黑色半透明阴影 (向右下偏移 2 像素)，防止在浅色 IDE 下看不清文字
        painter.setPen(QColor(0, 0, 0, 160))
        shadow_rect = self.rect().adjusted(2, 2, 2, 2)
        painter.drawText(shadow_rect, Qt.AlignCenter, text)
        
        # 2. 画纯白色文字本体
        painter.setPen(QColor(255, 255, 255, 255))
        painter.drawText(self.rect(), Qt.AlignCenter, text)
        
        # --- 绘制底部小字 ---
        font_small = QFont("Segoe UI", 10)
        font_small.setLetterSpacing(QFont.AbsoluteSpacing, 2.0) # 增加字间距更有设计感
        painter.setFont(font_small)
        
        # 1. 底部小字阴影
        text_rect_shadow = self.rect().adjusted(2, 62, 2, 2)
        painter.setPen(QColor(0, 0, 0, 160))
        painter.drawText(text_rect_shadow, Qt.AlignCenter, "MAX 01:00")
        
        # 2. 底部小字本体 (提高可见度)
        text_rect = self.rect().adjusted(0, 60, 0, 0)
        painter.setPen(QColor(255, 255, 255, 200))
        painter.drawText(text_rect, Qt.AlignCenter, "MAX 01:00")

def main():
    # 启动后台监听与录音线程
    t = threading.Thread(target=background_task, daemon=True)
    t.start()
    
    # 启动 PyQt 的主事件循环
    app = QApplication(sys.argv)
    overlay = OverlayUI()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
