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
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QTimer, QRectF

# ================= 配置参数 =================
MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
HOTKEY = "f8"
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

def background_task():
    """后台任务：监听快捷键、录音以及 AI 推理"""
    global is_recording, recording_start
    
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
    winsound.Beep(600, 200)

    while True:
        try:
            if keyboard.is_pressed(HOTKEY):
                winsound.Beep(1500, 100)
                is_recording = True
                recording_start = time.time()
                frames = []
                
                p = pyaudio.PyAudio()
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

                # 持续录音，直到松开按键，或者时间到达 60 秒上限
                while keyboard.is_pressed(HOTKEY) and (time.time() - recording_start) <= MAX_RECORD_SECONDS:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)

                # 录音结束
                is_recording = False
                winsound.Beep(1000, 100)
                stream.stop_stream()
                stream.close()
                p.terminate()

                # 将录音转为浮点格式给 AI
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
                    keyboard.write(text)
                    keyboard.write(" ")
                    
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
        
        # 绘制底部暗色圆环
        pen_bg = QPen(QColor("#333333"), 12)
        pen_bg.setCapStyle(Qt.RoundCap) # 圆角线头
        painter.setPen(pen_bg)
        painter.drawArc(rect, 0, 360 * 16) # Qt 中的角度单位是 1/16 度
        
        # 绘制进度亮色圆弧
        pen_fg = QPen(QColor("#00FFCC"), 12)
        pen_fg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_fg)
        
        # startAngle: 90度(12点钟方向)
        # spanAngle: 负数代表顺时针
        start_angle = 90 * 16
        span_angle = -int((elapsed / MAX_RECORD_SECONDS) * 360 * 16)
        painter.drawArc(rect, start_angle, span_angle)
        
        # 绘制中心时间文字
        painter.setPen(QColor("white"))
        font = QFont("Segoe UI", 28, QFont.Bold)
        painter.setFont(font)
        text = f"00:{int(elapsed):02d}"
        painter.drawText(self.rect(), Qt.AlignCenter, text)
        
        # 绘制底部小字
        painter.setPen(QColor("#AAAAAA"))
        font_small = QFont("Segoe UI", 10)
        painter.setFont(font_small)
        text_rect = self.rect().adjusted(0, 50, 0, 0)
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
