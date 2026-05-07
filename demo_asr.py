import time
import wave
import pyaudio
import keyboard
import openvino_genai as ov_genai
import sys
import numpy as np

# ================= 配置参数 =================
MODEL_PATH = "whisper-tiny-int8" # 使用 tiny 模型作为快速 demo
DEVICE = "NPU"                   # 注意：为了确保所有人能跑通，默认先用 CPU，如果您确定 NPU 驱动正常，可改为 "NPU"
HOTKEY = "f8"                    # 按住 F8 录音

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
    frames = []

    while keyboard.is_pressed(HOTKEY):
        data = stream.read(CHUNK)
        frames.append(data)

    print("⏸️ [录音结束] 正在进行推理识别...")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 将录音数据转为 numpy 浮点数组，WhisperPipeline 接收 [-1.0, 1.0] 的 float 序列
    raw_data = b''.join(frames)
    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    return audio_float32.tolist()

def main():
    print(f"🔄 正在加载 OpenVINO Whisper 模型到 {DEVICE}...")
    try:
        pipe = ov_genai.WhisperPipeline(MODEL_PATH, DEVICE)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请检查是否已经执行了 optimum-cli 导出了模型。")
        sys.exit(1)

    print(f"\n🚀 Web Coding 语音助手已启动！")
    print(f"👉 按住【{HOTKEY}】键说话，松开即可自动打字。")
    print(f"👉 按下【Esc】键退出程序。\n")

    while True:
        try:
            if keyboard.is_pressed(HOTKEY):
                audio_data = record_audio()
                
                start_time = time.time()
                result = pipe.generate(audio_data)
                text = str(result).strip()
                elapsed = time.time() - start_time
                
                print(f"📝 [识别结果] (耗时 {elapsed:.2f}s): {text}")
                
                if text:
                    keyboard.write(text)
                    # 额外补充一个空格，方便连续输入
                    keyboard.write(" ")
                    
                time.sleep(0.5)
                
            if keyboard.is_pressed('esc'):
                print("👋 退出程序。")
                break
                
            time.sleep(0.05) 
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
