import os
import sys
import time

# 强制使用 UTF-8 编码，防止 Windows 终端显示 Emoji 发生编码异常
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def download_17b_model(target_dir="qwen3-asr-1.7b-int4", use_mirror=True):
    # 优先指定国内高速镜像源，加速大模型下载
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("🌐 已启用国内高速镜像源: https://hf-mirror.com")
    else:
        print("🌐 正在使用 Hugging Face 官方源下载...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ 未安装 huggingface_hub，正在自动安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    # 1.7B 旗舰高质量模型 (INT4 压缩版，兼顾顶级精度与本地 2.5GB 紧凑显存占用)
    repo_id = "andrewleech/qwen3-asr-1.7b-onnx"
    print(f"\n📥 正在下载 Qwen3-ASR 1.7B 旗舰高质量模型: {repo_id}")
    print(f"📁 目标存储路径: {os.path.abspath(target_dir)}\n")

    # 仅下载 int4 推理所需的关键子集，跳过 7GB 未量化权重以节约带宽与磁盘空间
    allow_patterns = [
        "encoder.int4.onnx",
        "encoder.onnx",
        "decoder_init.int4.onnx",
        "decoder_step.int4.onnx",
        "decoder_weights.int4.data",
        "embed_tokens.bin",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "config.json",
        "preprocessor_config.json"
    ]

    start_t = time.time()
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            resume_download=True
        )
        elapsed = time.time() - start_t
        print(f"\n🎉 1.7B 高质量模型下载完成！耗时: {elapsed:.1f} 秒")
        print(f"✅ 文件已保存至: {downloaded_path}")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print(f"💡 备选方案：您可以访问 https://hf-mirror.com/{repo_id} 进行查看")
        return False

if __name__ == "__main__":
    download_17b_model()
