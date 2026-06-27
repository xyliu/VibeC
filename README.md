# VibeC - 极速语音编码助手 (SenseVoice / FireRedASR 双版本)

VibeC 是一个专为 Web Coding 场景设计的极速开源语音打字助手。
它底层基于 C++ 极速推理引擎 **sherpa-onnx**。通过接管系统原生的 `Win + H` 快捷键，实现超低延迟的“话音刚落，代码上屏”，为您带来丝滑的双手解放体验。

本项目目前支持两个卓越的离线语音模型：
1. **SenseVoice（默认版）**：速度极快，CPU 友好，适合纯中文和多语种识别。
2. **FireRedASR（中英混杂增强版，推荐）**：由小红书团队开源，对中英文混杂（Code-Switching）、编程术语、普通话方言具有卓越的识别效果。

## 🌟 核心特性
- **毫秒级推理**：得益于 `sherpa-onnx` 的底层 C++ 优化，普通的 Intel CPU 处理短语音识别几乎在瞬间完成（不到 0.1 秒）。
- **完全接管原生体验**：深度屏蔽并接管 Windows 自带的 `Win+H` 语音输入，提供更懂中文与中英代码混排的识别能力。
- **自适应模型检测**：`demo_fireredasr.py` 会自动探测模型文件夹下是更小、更快的 **CTC**（单文件）模型，还是精度更高的 **AED**（双文件）模型并自动匹配加载。
- **现代化透明 UI**：按键时在屏幕中央弹出带有“赛博朋克渐变”与“完美抗锯齿”的半透明悬浮表盘。
- **双色域主题**：支持暗色 (Dark) 和亮色 (Light) 主题，完美适配各类黑白代码编辑器背景。
- **后台免打扰**：完全托盘化运行，不占用系统任务栏。

---

## 🛠️ 1. 环境依赖安装

首先，确保您的电脑上安装了 Python 3.10 或更高版本。

在终端中执行以下命令，安装必备的 Python 第三方库：
```bash
pip install pyaudio keyboard numpy sherpa-onnx PyQt5 pyinstaller comtypes pycaw pynput
```
> **注 1：** 如果您在安装 `pyaudio` 时遇到 C++ 编译报错，可以直接下载对应的 [PyAudio 预编译 whl 轮子文件](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) 并通过 pip 安装。
> **注 2：** 在 Windows 终端运行如果遇到 `UnicodeEncodeError` (例如无法显示 Emoji)，请先执行 `$env:PYTHONUTF8=1` (PowerShell) 或 `set PYTHONUTF8=1` (CMD) 再启动。

---

## 📥 2. 下载并配置预编译模型

由于模型文件体积较大，并未直接包含在代码中。请根据您的选择下载模型：

### 选项 A：FireRedASR (强烈推荐中英编程场景)
1. **下载 CTC 模型 (运行速度最快)**:
   - 下载链接：[sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2](https://huggingface.co/csukuangfj/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25)
2. **下载 AED 模型 (识别精度最高)**:
   - 下载链接：[sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2](https://huggingface.co/csukuangfj/sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26)
3. **放置模型**：解压下载的文件夹，并将其移动到项目根目录（与 `demo_fireredasr.py` 同级）。修改 `demo_fireredasr.py` 头部的 `MODEL_DIR_NAME` 为您下载的文件夹名即可。

### 选项 B：SenseVoice (原版)
1. **下载模型包**：[点击前往 GitHub Releases 下载模型](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2) (约 300MB)
2. **放置模型**：将解压后的 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` 文件夹移动到项目根目录。

---

## 🚀 3. 如何使用

### 启动方式
在终端中执行以下命令（以 FireRedASR 版本为例）：
```bash
python demo_fireredasr.py
```
*(如果快捷键由于权限不足无法触发全局响应，请尝试以**管理员身份**运行终端或 IDE 再执行代码)*

### 交互指南：
1. **唤醒录音**：在任何软件里，将光标定位在想打字的地方，**单击一次键盘上的 `Win + H`**。此时屏幕中央会瞬间弹出一个悬浮表盘。
2. **开始讲话**：对着麦克风说出您的代码逻辑或文本。程序最长支持单次 60 秒的录音保护。
3. **结束并上屏**：**再次单击 `Win + H`**。表盘立即消失，识别出的文字会自动键入在您的鼠标光标处。

### 系统托盘功能：
程序启动后，会在 Windows 屏幕右下角的系统托盘区生成一个“青色小圆球”图标。
- **右键 -> 📝 使用说明**：查看快捷键提示。
- **右键 -> 🎨 切换主题**：在明亮模式 (深海蓝-翡翠绿渐变) 和黑暗模式 (青-紫渐变) 间一键来回切换。
- **右键 -> ❌ 完全退出**：安全干净地退出后台驻留程序。

---

## 📦 4. 打包为独立 EXE 桌面软件

如果您希望将其打包为独立可执行的 exe 软件，运行：

```bash
# 打包 FireRedASR 版
python -m PyInstaller -F -w demo_fireredasr.py
```

**打包后使用说明：**
1. 打包完成后，进入新生成的 `dist` 文件夹，您会找到 `demo_fireredasr.exe`。
2. 将您下载的模型文件夹完整复制到 `dist` 文件夹中，使其与 exe 文件处于同一级目录下。
3. 双击运行即可。
