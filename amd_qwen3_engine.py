"""
amd_qwen3_engine.py - 基于 DirectML 的 AMD NPU / GPU + CPU 异构加速 Qwen3-ASR 推理引擎
架构设计:
- 算力密集型网络 (ConvFrontend + Encoder): 卸载至 AMD NPU / DirectML 硬件加速层执行 (低功耗、高能效比)
- 序列解码器与 KV Cache (Decoder): 在 AMD 性能大核 CPU 上多线程执行 (确保 INT8 精度与极速单步循环)
- 支持预热机制与麦克风音频流无缝对接
"""

import os
import sys
import time
import numpy as np

if sys.platform.startswith("win"):
    try:
        if sys.stdout:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import onnxruntime as ort
from transformers import AutoTokenizer, WhisperFeatureExtractor


def feat_to_audio_tokens_len(feat_len: int, chunk_size: int = 100) -> int:
    def conv_out_len_3x_stride2(n: int) -> int:
        x = (int(n) + 1) // 2
        x = (x + 1) // 2
        return (x + 1) // 2

    def aftercnn(x: int) -> int:
        if x <= 0:
            return 0
        x = (x - 1) // 2 + 1
        x = (x - 1) // 2 + 1
        return (x - 1) // 2 + 1

    cs = int(chunk_size)
    full = feat_len // cs
    rem = feat_len % cs
    tn = conv_out_len_3x_stride2(cs)
    out = full * tn
    if rem > 0:
        out += aftercnn(rem)
    return max(out, 0)


class AmdQwen3Recognizer:
    """专为 AMD NPU / Ryzen AI 与 Radeon 硬件打造的 Qwen3-ASR 异构推理引擎"""

    def __init__(self, model_dir: str, use_dml: bool = True, num_threads: int = 4):
        self.model_dir = model_dir
        self.use_dml = use_dml
        self.num_threads = num_threads

        # 硬件加速 Provider 配置
        available_providers = ort.get_available_providers()
        if use_dml and "DmlExecutionProvider" in available_providers:
            self.accel_provider = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self.backend_desc = "AMD NPU / DirectML 硬件加速"
        else:
            self.accel_provider = ["CPUExecutionProvider"]
            self.backend_desc = "CPU 性能多核加速"

        print(f"[AmdQwen3] 初始化推理引擎 | 编码器加速后端: {self.backend_desc}")
        t0 = time.time()

        # Session 配置
        opt_accel = ort.SessionOptions()
        opt_accel.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        opt_cpu = ort.SessionOptions()
        opt_cpu.intra_op_num_threads = num_threads
        opt_cpu.inter_op_num_threads = 1
        opt_cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 1. ConvFrontend (AMD NPU / DirectML)
        conv_path = os.path.join(model_dir, "conv_frontend.onnx")
        self.sess_conv = ort.InferenceSession(conv_path, opt_accel, providers=self.accel_provider)

        # 2. Encoder (AMD NPU / DirectML)
        enc_name = "encoder.int8.onnx" if os.path.exists(os.path.join(model_dir, "encoder.int8.onnx")) else "encoder.onnx"
        enc_path = os.path.join(model_dir, enc_name)
        self.sess_enc = ort.InferenceSession(enc_path, opt_accel, providers=self.accel_provider)

        # 3. Decoder (CPU 性能大核，高吞吐序列生成)
        dec_name = "decoder.int8.onnx" if os.path.exists(os.path.join(model_dir, "decoder.int8.onnx")) else "decoder.onnx"
        dec_path = os.path.join(model_dir, dec_name)
        self.sess_dec = ort.InferenceSession(dec_path, opt_cpu, providers=["CPUExecutionProvider"])

        # 4. Tokenizer & Feature Extractor
        tok_dir = os.path.join(model_dir, "tokenizer") if os.path.exists(os.path.join(model_dir, "tokenizer")) else model_dir
        self.tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        self.fe = WhisperFeatureExtractor(feature_size=128)

        # 5. 模型结构常量
        self.audio_pad_id = 151676
        self.eos_id = 151645
        self.asr_text_token_id = 151704
        self.num_layers = 28
        self.kv_heads = 8
        self.head_dim = 128
        self.max_total_len = 512

        self.prompt_before = self.tok.encode(
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>",
            add_special_tokens=False
        )
        self.prompt_after = self.tok.encode(
            "<|audio_end|><|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False
        )

        # 6. KV Cache 预分配
        self.cache_k = [np.zeros((1, self.max_total_len, self.kv_heads, self.head_dim), dtype=np.float32) for _ in range(self.num_layers)]
        self.cache_v = [np.zeros((1, self.max_total_len, self.kv_heads, self.head_dim), dtype=np.float32) for _ in range(self.num_layers)]

        # 7. 预热加速器管道
        self._warmup()
        print(f"[AmdQwen3] 加载与预热完成，总耗时: {time.time()-t0:.2f}s\n")

    def _warmup(self):
        try:
            dummy_feat = np.zeros((1, 100, 128), dtype=np.float32)
            dummy_conv = self.sess_conv.run(None, {"input_features": dummy_feat})[0]
            dummy_mask = np.ones((1, dummy_conv.shape[1]), dtype=bool)
            _ = self.sess_enc.run(None, {"input_features": dummy_conv, "feature_attention_mask": dummy_mask})
        except Exception as e:
            print(f"[AmdQwen3] 预热警告 (已跳过): {e}")

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 64) -> str:
        """
        接收 float32 单声道 16000Hz 采样率的音频数组，返回识别文本
        """
        if len(audio) < 1600:
            return ""

        # 1. 音频特征提取
        feat_res = self.fe(audio, sampling_rate=16000, padding=False, return_tensors="np")
        input_features = feat_res.input_features.transpose(0, 2, 1)
        feat_len = input_features.shape[1]

        # chunk_size 对齐
        rem = feat_len % 100
        if rem != 0:
            pad_len = 100 - rem
            padded_input_features = np.pad(input_features, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
        else:
            padded_input_features = input_features

        # 2. ConvFrontend (AMD NPU / DirectML)
        conv_out = self.sess_conv.run(None, {"input_features": padded_input_features})[0]
        A_conv = conv_out.shape[1]
        expected_len = feat_to_audio_tokens_len(feat_len, 100)
        valid_frames = min(expected_len, A_conv)

        # 3. Encoder (AMD NPU / DirectML)
        tok_mask = np.zeros((1, A_conv), dtype=bool)
        tok_mask[0, :valid_frames] = True
        enc_out = self.sess_enc.run(None, {
            "input_features": conv_out,
            "feature_attention_mask": tok_mask
        })[0]
        audio_features = enc_out[:, :valid_frames, :]

        # 4. Prompt 序列构建
        input_ids = np.array([self.prompt_before + [self.audio_pad_id] * valid_frames + self.prompt_after], dtype=np.int64)
        S0 = input_ids.shape[1]

        if S0 + max_new_tokens > self.max_total_len:
            return ""

        # 重置 KV Cache
        for i in range(self.num_layers):
            self.cache_k[i].fill(0.0)
            self.cache_v[i].fill(0.0)

        # 5. Prefill 首包解码
        feed_dict = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "attention_mask": np.ones((1, S0), dtype=np.int64),
            "cache_position": np.arange(0, S0, dtype=np.int64),
        }
        for i in range(self.num_layers):
            feed_dict[f"cache_key_{i}"] = self.cache_k[i]
            feed_dict[f"cache_value_{i}"] = self.cache_v[i]

        dec_out = self.sess_dec.run(None, feed_dict)

        # 回填 Prefill 增量 KV
        for i in range(self.num_layers):
            self.cache_k[i][:, 0:S0] = dec_out[1 + i * 2]
            self.cache_v[i][:, 0:S0] = dec_out[2 + i * 2]

        logits = dec_out[0]
        next_id = int(np.argmax(logits[0, -1, :]))

        generated_tokens = []
        if next_id != self.eos_id:
            generated_tokens.append(next_id)

        # 6. 单步自回归 Decode 循环
        cur_len = S0

        for _ in range(max_new_tokens):
            if next_id == self.eos_id:
                break

            step_feed = {
                "input_ids": np.array([[next_id]], dtype=np.int64),
                "audio_features": audio_features,
                "attention_mask": np.ones((1, 1), dtype=np.int64),
                "cache_position": np.array([cur_len], dtype=np.int64),
            }
            for i in range(self.num_layers):
                step_feed[f"cache_key_{i}"] = self.cache_k[i]
                step_feed[f"cache_value_{i}"] = self.cache_v[i]

            step_out = self.sess_dec.run(None, step_feed)

            # 更新当前步的 KV Cache
            for i in range(self.num_layers):
                self.cache_k[i][:, cur_len:cur_len+1] = step_out[1 + i * 2]
                self.cache_v[i][:, cur_len:cur_len+1] = step_out[2 + i * 2]

            cur_len += 1
            next_id = int(np.argmax(step_out[0][0, -1, :]))
            if next_id == self.eos_id:
                break
            generated_tokens.append(next_id)

        # 7. 清理特殊 token 与解码文字
        cleaned_tokens = list(generated_tokens)
        if self.asr_text_token_id in cleaned_tokens:
            idx = cleaned_tokens.index(self.asr_text_token_id)
            cleaned_tokens = cleaned_tokens[idx + 1:]

        return self.tok.decode(cleaned_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("=" * 60)
    print("   AMD NPU / DirectML Qwen3-ASR 硬件加速引擎自检")
    print("=" * 60)
    model_dir = "sherpa-onnx-qwen3-asr-0.6B-int8"
    if os.path.exists(model_dir):
        rec = AmdQwen3Recognizer(model_dir, use_dml=True)
        # 生成 2 秒 440Hz 纯音测试
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        dummy_audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        print("[*] 正在测试测试音频推理...")
        t_start = time.time()
        text = rec.transcribe(dummy_audio)
        print(f"[SUCCESS] 推理成功！识别耗时: {time.time()-t_start:.3f}s，输出: '{text}'")
    else:
        print(f"[ERROR] 未找到模型文件夹: {model_dir}")
