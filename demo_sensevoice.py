import time
import pyaudio
import keyboard
import numpy as np
import sherpa_onnx
import os
import sys
import winsound

# ================= 配置参数 =================
MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
HOTKEY = "f8"
EXIT_HOTKEY = "ctrl+shift+q" # 退出程序的快捷键

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# ============================================

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"\n🎤 [录音中] 请对着麦克风说话... (松开 {HOTKEY} 键结束)")
    # 播放提示音，表示开始录音
    winsound.Beep(1500, 100)
    frames = []

    while keyboard.is_pressed(HOTKEY):
        data = stream.read(CHUNK)
        frames.append(data)

    print("⏸️ [录音结束] 正在调用 SenseVoice 进行推理...")
    # 播放提示音，表示录音结束
    winsound.Beep(1000, 100)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 转换为 float32 数组，范围 [-1, 1]
    raw_data = b''.join(frames)
    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32

def get_application_path():
    """获取程序运行时的绝对目录，兼容 PyInstaller 打包后的 exe"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def main():
    app_path = get_application_path()
    model_dir = os.path.join(app_path, MODEL_DIR_NAME)
    model_file = os.path.join(model_dir, "model.onnx")
    tokens_file = os.path.join(model_dir, "tokens.txt")
    
    if not os.path.exists(model_file):
        print(f"❌ 找不到模型文件: {model_file}")
        print("请先从 GitHub 下载 SenseVoice 预编译包并解压到当前目录。")
        sys.exit(1)

    print(f"🔄 正在加载 SenseVoice 模型...")
    
    # 初始化 SenseVoice 识别器
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_file,
        tokens=tokens_file,
        num_threads=1,
        use_itn=True, # 开启逆文本正则化（将 "一百" 转为 "100" 等）
        #provider="dml", # 如果您的 sherpa-onnx 编译了 DirectML，取消注释这行即可使用 Intel GPU
    )
    print("✅ SenseVoice 加载成功！")
    print(f"👉 按住【{HOTKEY}】键说话测试。")
    print(f"👉 按下【{EXIT_HOTKEY}】键退出后台程序。")
    winsound.Beep(600, 200) # 程序启动成功的提示音

    while True:
        try:
            if keyboard.is_pressed(HOTKEY):
                audio_data = record_audio()
                
                # 创建音频流并送入模型
                start_time = time.time()
                stream = recognizer.create_stream()
                stream.accept_waveform(RATE, audio_data)
                recognizer.decode_stream(stream)
                
                text = stream.result.text
                elapsed = time.time() - start_time
                
                print(f"📝 [识别结果] (耗时 {elapsed:.2f}s): {text}")
                
                if text:
                    keyboard.write(text)
                    keyboard.write(" ")
                    
                time.sleep(0.3)
                
            if keyboard.is_pressed(EXIT_HOTKEY):
                winsound.Beep(400, 300)
                break
            time.sleep(0.05) 
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
