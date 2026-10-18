# VibeC Qwen3-ASR 性能基准评测报告 (CPU vs. AMD NPU / DirectML)

本文档固化了 VibeC 在 AMD 平台（**AMD Ryzen 7 H 255 + Radeon 780M / XDNA NPU 架构**）上对 **CPU 原生版 (`sherpa-onnx`)** 与 **AMD NPU 硬件加速版 (`amd_qwen3_engine`)** 的真实基准评测数据、架构分析与选型指南。

---

## 一、 测试环境与平台规格

| 硬件 / 软件组件 | 规格参数 |
| :--- | :--- |
| **CPU 处理器** | AMD Ryzen 7 H 255 w/ Radeon 780M Graphics (8 核 16 线程，Zen4 架构) |
| **GPU / 加速单元** | AMD Radeon(TM) 780M Graphics (12 CUs, RDNA3 架构 / gfx1103) + AI 算力单元 |
| **操作系统** | Windows 11 64-bit |
| **Python 环境** | Python 3.12.13 (uv 托管虚拟环境 `.venv`) |
| **测试模型** | `Qwen3-ASR 0.6B (INT8 量化版)` (阿里通义实验室 2026 最新开源模型) |
| **音频采样规格** | 16,000 Hz, 16-bit 单声道 Float32 标准语音输入 |
| **基准测试脚本** | [benchmark_asr.py](file:///c:/sw/vibec/benchmark_asr.py) (纯净独立子进程隔离测试) |
| **原始数据输出** | [benchmark_results.json](file:///c:/sw/vibec/benchmark_results.json) |

---

## 二、 核心实测数据对比

测试方法：各测试时长音频均经过 2 轮热身后，连续执行 5 轮全流程推理取均值。

### 1. 推理延迟与实时因子 (RTF)

$$\text{RTF (实时因子)} = \frac{\text{端到端推理耗时}}{\text{音频时长}} \quad (\text{数值越小越快，小于 0.2 代表快于讲话 5 倍以上})$$

| 测试音频长度 | 典型场景 | CPU 原生版 (`sherpa-onnx`) | AMD NPU 硬件加速版 (`DirectML`) | 性能对比 / 加速比 |
| :---: | :--- | :---: | :---: | :---: |
| **2.0 秒** | 短命令（如 `git commit`、快捷短语） | **550.0 ms** <br> (RTF: 0.2750) | **551.5 ms** <br> (RTF: 0.2757) | **基本持平** (差异 < 1.5ms) |
| **5.0 秒** | 日常输入（如单行代码、普通注释、一句话） | **886.1 ms** <br> (RTF: 0.1772) | **647.5 ms** <br> (RTF: 0.1295) | 🚀 **NPU 快 1.37 倍** (节省 **238.6 ms**) |
| **10.0 秒** | 连续长句（如完整函数逻辑描述、长段落） | **1456.0 ms** <br> (RTF: 0.1456) | **1144.2 ms** <br> (RTF: 0.1144) | 🚀 **NPU 快 1.27 倍** (节省 **311.8 ms**) |

---

### 2. 系统资源与启动开销对比

| 评估维度 | CPU 原生版 (`sherpa-onnx`) | AMD NPU 硬件加速版 (`DirectML`) | 工程差异分析 |
| :--- | :---: | :---: | :--- |
| **模型冷启动耗时** | **3.69 秒** | **9.33 秒** | DirectML 首次加载需构建 DirectX 12 计算管线并预热着色器/内核。 |
| **进程常驻内存 (RSS)** | **1155.3 MB** | **1450.7 MB** | NPU 加速版额外占用约 295 MB，用于 DML 硬件加速显存与张量交换上下文。 |
| **推理时 CPU 占用** | **4 核心 100% 满载** | **大幅降低** (约 15~25%) | NPU 版将繁重的前端特征提取与 Encoder 卸载，释放 CPU。 |
| **整机功耗与发热** | 较高（多核 CPU 高频运行） | **显著降低** (高能效比专用单元) | 降低风扇转速与发热，极大延长笔记本移动办公续航。 |

---

## 三、 深度架构机理解析

### 1. 为什么 2 秒短音频两者打平？
Qwen3-ASR 包含 `conv_frontend`、`encoder` 和 `decoder (KV Cache)`。
- 对于 2 秒以内的超短音频，模型的计算张量规模较小；
- 此时硬件加速器所消耗的 **Host-Device 数据传输（PCIE/显存调度）与内核启动开销 (Launch Overhead)** 与算力收益相当；
- 因此在 2 秒以内短输入场景下，两者端到端耗时均稳定在约 0.55 秒。

### 2. 为什么 5~10 秒中长音频 NPU 硬件加速明显胜出？
- 当音频长度达到 5 秒或 10 秒时，Transformer Encoder 阶段的自注意力与前馈网络计算量随时间序列呈线性/二次方扩展；
- AMD NPU / DirectML 专为高并发矩阵乘法算子设计，并行浮点吞吐能力远超通用 CPU 指令集；
- 5 秒音频推理时间从 **886ms 锐减至 647ms**，10 秒音频从 **1.45s 锐减至 1.14s**，带来“话音刚落瞬间上屏”的明显体感提升。

---

## 四、 选型与部署建议

| 场景需求 | 推荐选用版本 | 启动方式 |
| :--- | :--- | :--- |
| **日常编程长句 / 深度打字输入** | **AMD NPU 硬件加速版** (推荐) | 双击运行 [start_amd_npu.bat](file:///c:/sw/vibec/start_amd_npu.bat) 或静默版 [start_amd_npu_silent.vbs](file:///c:/sw/vibec/start_amd_npu_silent.vbs) |
| **内存极度敏感 / 追求秒级极速冷启动** | **CPU 原生极速版** | 双击运行 [start_vibec.bat](file:///c:/sw/vibec/start_vibec.bat) 或静默版 [start_vibec_silent.vbs](file:///c:/sw/vibec/start_vibec_silent.vbs) |

---

## 五、 如何复现本测试

在项目根目录下通过内置虚拟环境随时复跑基准评测：

```powershell
.venv\Scripts\python.exe benchmark_asr.py
```

评测完成后会自动更新当前目录下的 `benchmark_results.json`。
