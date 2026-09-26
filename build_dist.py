import os
import sys
import shutil
import subprocess

# 适配 Windows 控制台 UTF-8 编码
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    print("=" * 60)
    print("🚀 开始自动打包 VibeC (Qwen3-ASR) 桌面独立版本...")
    print("=" * 60)

    project_root = os.path.dirname(os.path.abspath(__file__))
    spec_file = os.path.join(project_root, "demo_qwen3_asr.spec")

    if not os.path.exists(spec_file):
        print(f"❌ 找不到打包配置文件: {spec_file}")
        return

    # 1. 执行 PyInstaller 打包
    cmd = [sys.executable, "-m", "PyInstaller", spec_file, "--noconfirm"]
    print(f"\n📦 正在执行打包命令: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=project_root)
    if ret.returncode != 0:
        print("❌ 打包过程中发生错误，请检查上方日志。")
        return

    # 2. 检查生成的发布目录
    target_dist_dir = os.path.join(project_root, "dist", "demo_qwen3_asr")
    if not os.path.exists(target_dist_dir):
        print(f"❌ 未找到生成的发布目录: {target_dist_dir}")
        return

    print(f"\n✅ EXE 打包成功！发布目录: {target_dist_dir}")

    # 3. 自动拷贝 Qwen3-ASR 模型文件夹至 exe 同级目录下
    model_name = "sherpa-onnx-qwen3-asr-0.6B-int8"
    src_model_dir = os.path.join(project_root, model_name)
    dst_model_dir = os.path.join(target_dist_dir, model_name)

    if os.path.exists(src_model_dir):
        print(f"🔄 正在自动将模型文件夹部署到发布目录:\n   从: {src_model_dir}\n   到: {dst_model_dir}")
        if os.path.exists(dst_model_dir):
            shutil.rmtree(dst_model_dir)
        shutil.copytree(src_model_dir, dst_model_dir)
        print("🎉 模型文件夹自动部署完毕！")
    else:
        print(f"⚠️ 项目根目录未检测到 {model_name}，请先运行 python download_qwen3_asr.py 下载模型。")

    print("\n" + "=" * 60)
    print("✨ 全部打包与部署工作完成！")
    print(f"👉 您可以直接打开此文件夹运行: {target_dist_dir}")
    print(f"👉 双击运行: demo_qwen3_asr.exe 即可开启使用！")
    print("=" * 60)

if __name__ == "__main__":
    main()
