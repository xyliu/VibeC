"""
benchmark_asr.py - Qwen3-ASR CPU 版本 vs AMD NPU/DirectML 版本 独立性能基准测试
"""

import sys
import os
import time
import json
import subprocess
import numpy as np

# 确保 Windows 下字符输出正常
if sys.platform.startswith("win"):
    try:
        if sys.stdout:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def generate_audio_signal(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """生成带有复合谐波的合成测试音频，模拟语音信号特征"""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # 模拟人声基频及共振峰 (150Hz + 300Hz + 800Hz + 2500Hz)
    audio = (
        0.25 * np.sin(2 * np.pi * 150 * t) +
        0.20 * np.sin(2 * np.pi * 300 * t) +
        0.15 * np.sin(2 * np.pi * 800 * t) +
        0.05 * np.sin(2 * np.pi * 2500 * t)
    ).astype(np.float32)
    # 调制幅度包络
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    return (audio * envelope).astype(np.float32)


def run_benchmark_worker(engine_type: str, model_dir: str):
    """在子进程内部执行单一引擎的纯净性能测试"""
    import psutil
    process = psutil.Process(os.getpid())

    results = {
        "engine": engine_type,
        "load_time_sec": 0.0,
        "memory_mb": 0.0,
        "tests": {}
    }

    t0 = time.perf_counter()

    if engine_type == "cpu":
        import sherpa_onnx
        conv_frontend = os.path.join(model_dir, "conv_frontend.onnx")
        encoder = os.path.join(model_dir, "encoder.int8.onnx")
        decoder = os.path.join(model_dir, "decoder.int8.onnx")
        tokenizer = os.path.join(model_dir, "tokenizer")
        if not os.path.exists(tokenizer):
            tokenizer = model_dir

        recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=conv_frontend,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            num_threads=4,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu",
            max_total_len=512,
            max_new_tokens=64
        )

        def infer_fn(audio):
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            recognizer.decode_stream(stream)
            return stream.result.text

    elif engine_type == "amd_npu":
        from amd_qwen3_engine import AmdQwen3Recognizer
        recognizer = AmdQwen3Recognizer(model_dir, use_dml=True, num_threads=4)

        def infer_fn(audio):
            return recognizer.transcribe(audio, max_new_tokens=64)

    else:
        raise ValueError(f"Unknown engine: {engine_type}")

    load_time = time.perf_counter() - t0
    results["load_time_sec"] = round(load_time, 3)

    # 预热推理 2 次
    warmup_audio = generate_audio_signal(1.0)
    for _ in range(2):
        _ = infer_fn(warmup_audio)

    # 记录常驻内存
    mem_info = process.memory_info()
    results["memory_mb"] = round(mem_info.rss / 1024 / 1024, 1)

    # 针对不同音频时长进行多轮基准测试
    test_durations = [2.0, 5.0, 10.0]
    rounds_per_duration = 5

    for dur in test_durations:
        audio = generate_audio_signal(dur)
        latencies = []
        for _ in range(rounds_per_duration):
            t_start = time.perf_counter()
            _ = infer_fn(audio)
            t_end = time.perf_counter()
            latencies.append(t_end - t_start)

        avg_lat = float(np.mean(latencies))
        min_lat = float(np.min(latencies))
        max_lat = float(np.max(latencies))
        rtf = avg_lat / dur  # 实时因子

        results["tests"][f"{dur}s"] = {
            "duration": dur,
            "avg_latency_ms": round(avg_lat * 1000, 1),
            "min_latency_ms": round(min_lat * 1000, 1),
            "max_latency_ms": round(max_lat * 1000, 1),
            "rtf": round(rtf, 4),
        }

    print("__BENCHMARK_RESULT_START__")
    print(json.dumps(results))
    print("__BENCHMARK_RESULT_END__")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        engine_type = sys.argv[2]
        model_dir = sys.argv[3] if len(sys.argv) > 3 else "sherpa-onnx-qwen3-asr-0.6B-int8"
        run_benchmark_worker(engine_type, model_dir)
        return

    print("=" * 70)
    print("   VibeC: Qwen3-ASR (0.6B) CPU 版本 vs AMD NPU/DirectML 性能基准评测")
    print("=" * 70)
    print("[*] 正在启动独立子进程进行基准测试，请稍候...\n")

    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    model_dir = "sherpa-onnx-qwen3-asr-0.6B-int8"

    benchmark_data = {}

    for engine in ["cpu", "amd_npu"]:
        label = "CPU 原生多核版本 (sherpa-onnx)" if engine == "cpu" else "AMD NPU / DirectML 硬件加速版本"
        print(f"[*] 正在评测: {label} ...")
        cmd = [python_exe, "-u", script_path, "--worker", engine, model_dir]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

        output = proc.stdout
        if "__BENCHMARK_RESULT_START__" in output:
            json_str = output.split("__BENCHMARK_RESULT_START__")[1].split("__BENCHMARK_RESULT_END__")[0].strip()
            benchmark_data[engine] = json.loads(json_str)
            print(f"    [OK] 评测完成 (内存: {benchmark_data[engine]['memory_mb']} MB, 加载时间: {benchmark_data[engine]['load_time_sec']}s)")
        else:
            print(f"    [FAILED] 运行异常:\n{proc.stderr}\n{proc.stdout}")

    # 保存测试数据到本地 JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("                      基准评测结果报告")
    print("=" * 70)

    if "cpu" in benchmark_data and "amd_npu" in benchmark_data:
        cpu = benchmark_data["cpu"]
        npu = benchmark_data["amd_npu"]

        print(f"\n【1. 资源开销对比】")
        print(f"  - 模型冷启动耗时: CPU = {cpu['load_time_sec']}s  |  AMD NPU/DirectML = {npu['load_time_sec']}s")
        print(f"  - 进程内存开销 (RSS): CPU = {cpu['memory_mb']} MB  |  AMD NPU/DirectML = {npu['memory_mb']} MB")

        print(f"\n【2. 推理延迟与 RTF (实时因子) 对比】")
        print(f"{'音频长度':<10} | {'指标':<10} | {'CPU (sherpa-onnx)':<18} | {'AMD NPU (DirectML)':<20} | {'对比/差异'}")
        print("-" * 75)

        for dur_key in ["2.0s", "5.0s", "10.0s"]:
            c_test = cpu["tests"][dur_key]
            n_test = npu["tests"][dur_key]

            speedup = c_test["avg_latency_ms"] / n_test["avg_latency_ms"]
            diff_str = f"NPU 快 {speedup:.2f}x" if speedup >= 1.0 else f"CPU 快 {1.0/speedup:.2f}x"

            print(f"{dur_key:<10} | 平均耗时    | {c_test['avg_latency_ms']:>8.1f} ms        | {n_test['avg_latency_ms']:>8.1f} ms          | {diff_str}")
            print(f"{'':<10} | 实时因子RTF | {c_test['rtf']:>8.4f}           | {n_test['rtf']:>8.4f}            | (数值越低越快)")
            print("-" * 75)

    print("\n[*] 评测结果已持久化保存至 benchmark_results.json\n")


if __name__ == "__main__":
    main()
