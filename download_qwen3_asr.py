import os
import sys
import time

# 强制使用 UTF-8 编码，防止 Windows 终端显示 Emoji 发生编码异常
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def download_model(target_dir="sherpa-onnx-qwen3-asr-0.6B-int8", use_mirror=True):
    # 优先配置国内镜像站加速下载，避免境外源连接波动或限速
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("🌐 已启用国内高速镜像源: https://hf-mirror.com")
    else:
        print("🌐 正在使用 Hugging Face 官方源下载...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ 未安装 huggingface_hub，正在尝试自动安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    repo_id = "thieunv-asilla/sherpa-onnx-qwen3-asr-0.6B-int8"
    print(f"\n📥 正在下载 Qwen3-ASR 0.6B INT8 模型库: {repo_id}")
    print(f"📁 目标存储路径: {os.path.abspath(target_dir)}\n")

    start_t = time.time()
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        elapsed = time.time() - start_t
        print(f"\n🎉 模型下载完成！耗时: {elapsed:.1f} 秒")
        print(f"✅ 文件已保存在: {downloaded_path}")
        
        # 验证关键组件是否齐备
        expected_files = ["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx"]
        all_ok = True
        for f in expected_files:
            p = os.path.join(target_dir, f)
            if os.path.exists(p):
                size_mb = os.path.getsize(p) / (1024 * 1024)
                print(f"  - 校验通过: {f} ({size_mb:.2f} MB)")
            else:
                print(f"  - ⚠️ 缺少文件: {f}")
                all_ok = False
                
        if all_ok:
            print("\n🚀 所有核心推理组件校验通过，可直接运行 python demo_qwen3_asr.py 启动！")
        return all_ok
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("💡 建议：您可以手动访问 https://hf-mirror.com/thieunv-asilla/sherpa-onnx-qwen3-asr-0.6B-int8 下载文件")
        return False

if __name__ == "__main__":
    download_model()
