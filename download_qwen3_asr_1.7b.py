import os
import sys
import time

# 强制使用 UTF-8 编码，防止 Windows 终端显示 Emoji 发生编码异常
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def download_17b_model(target_dir="sherpa-onnx-qwen3-asr-1.7B-int8", use_mirror=True):
    """
    下载 sherpa-onnx 兼容的 Qwen3-ASR 1.7B 旗舰高质量模型
    包含完整的 conv_frontend、encoder、decoder 与 tokenizer 组件
    """
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

    # sherpa-onnx 标准适配的 1.7B INT8 旗舰高质量版仓库
    repo_id = "solavr/sherpa-onnx-qwen3-asr-1.7B-int8"
    print(f"\n📥 正在下载 Qwen3-ASR 1.7B 旗舰高质量模型: {repo_id}")
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
        print(f"\n🎉 1.7B 高质量模型下载完成！耗时: {elapsed:.1f} 秒")
        print(f"✅ 文件已保存至: {downloaded_path}")

        # 校验核心推理组件是否齐全
        expected_files = [
            "conv_frontend.onnx",
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "decoder.int8.onnx.data"
        ]
        all_ok = True
        print("\n🔍 正在校验核心推理组件:")
        for f in expected_files:
            p = os.path.join(target_dir, f)
            if os.path.exists(p):
                size_mb = os.path.getsize(p) / (1024 * 1024)
                print(f"  - 校验通过: {f} ({size_mb:.2f} MB)")
            else:
                print(f"  - ⚠️ 缺少文件: {f}")
                all_ok = False

        if all_ok:
            print("\n🚀 所有核心推理组件校验通过！启动程序时将自动优先加载 1.7B 模型。")
        return all_ok
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print(f"💡 备选方案：您可以访问 https://hf-mirror.com/{repo_id} 查看或手动下载")
        return False

if __name__ == "__main__":
    download_17b_model()
