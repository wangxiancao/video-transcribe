# video-transcribe

**零云端依赖的视频转录翻译项目。60 分钟真实视频，7.2 分钟出片。**

一条命令完成「转录 → 翻译 → 字幕 → 烧录」全流程：自研 FP16 CUDA ASR 引擎（词级时间戳）+ 本地 Q4 量化翻译模型（HY-MT2-1.8B on llama.cpp）+ Qwen3-ForcedAligner 对齐 + 关键帧并行烧录。全程无任何云端 API、无 API key、运行期零下载。

```bash
python video_transcribe.py /path/to/video.mp4      # 也可传 URL
# 产物落 runs/<视频名>/：transcript/segments/translation JSON、
# subs/{source,target,bilingual}.srt、烧录好的 output.mp4
```

## 性能（RTX 3090 / 24 GB，60 分钟真实视频冷启动实测，双跑独立验证）

| 指标 | 值 |
|---|---:|
| 端到端墙钟 | **434 s ≈ 7.2 min（8.29 RTFx）** |
| 流水线显存峰值 | **14.56 GiB** |
| 阶段分解 | ingest 7.5 s / transcribe 30.4 s / segment 0.2 s / translate 129.3 s / subtitle 0.1 s / burn 261.7 s |

## 架构

```
video_transcribe.py        # 唯一入口
pipeline/                  # 六阶段管线：ingest→transcribe→segment→translate→subtitle→burn
  asr/                     #   ASR 后端接缝：glm_asr_native（默认）/ glm_asr（transformers 回退）
  llm/                     #   托管本地 LLM 服务：多副本生命周期 + 轮询客户端（llamacpp/transformers 双 runtime）
  stages/                  #   各阶段实现（规则分段、id-free 有序数组翻译协议、并行烧录）
native/asr/                # 自研 FP16 CUDA ASR 引擎（C++/cuBLAS/wmma，无 PyTorch/vLLM）
native/llamacpp/           # Q4 翻译模型的 llama.cpp STQ 构建脚本
```

设计决策见 `docs/adr/`（自研引擎选型、对齐器留 transformers）；执行链契约见 `docs/agents/execution-chain.md`。

## 安装

**前置**：Python 3.10、NVIDIA GPU（≥16 GB 显存）、CUDA 12.x、FFmpeg、Noto CJK 字体。

1. 克隆并创建环境：

```bash
git clone <this-repo> && cd video-transcribe
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

2. 构建推理引擎：

```bash
# 自研 ASR 引擎（CMake + CUDA 12.x）
cmake -B native/asr/build -S native/asr && cmake --build native/asr/build -j
# Q4 翻译模型 runtime（llama.cpp STQ 分支）
bash native/llamacpp/build.sh
```

3. 预下载权重（一次性，运行期不下载）：

```bash
# ASR + 对齐器（ModelScope）
.venv/bin/python -c "from modelscope import snapshot_download; \
  snapshot_download('ZhipuAI/GLM-ASR-Nano-2512'); \
  snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf')"
# 翻译模型 Q4 GGUF（ModelScope）
.venv/bin/python -c "from modelscope import snapshot_download; \
  snapshot_download('Tencent-Hunyuan/HY-MT2-1.8B-GGUF', \
  local_dir='_model_cache/HY-MT2-1.8B-GGUF', \
  allow_patterns=['Hy-MT2-1.8B-Q4_K_M.gguf'])"
```

4. 配置：

```bash
cp config.example.yaml config.yaml    # 默认即全本地栈，开箱即用
```

## 用法

```bash
.venv/bin/python video_transcribe.py video.mp4                       # 全流程
.venv/bin/python video_transcribe.py video.mp4 --stages transcribe   # 只跑指定阶段
.venv/bin/python -m pipeline.runner --stages ... --validate-only ... # 只校验产物
```

`runs/<name>/` 是工作目录。只执行 `--stages` 明确选择的阶段，不自动补跑前置阶段，也不按产物存在性跳过；重跑会重新计算所选阶段，翻译响应缓存默认保留。

本地 LLM 请求不继承 Clash 等工具设置的环境代理。分段默认过滤独立填充词和病态重复；翻译器自述、无合规译文或超长译文会显式失败，不会回显源文或静默截断后烧录。

## 许可

模型权重各自遵循其官方许可（GLM-ASR-Nano: MIT；Qwen3-ForcedAligner: Apache-2.0；HY-MT2: Apache-2.0；llama.cpp: MIT）。
