# video-transcribe Agent 指南

## 项目概览

video-transcribe 是一个**零云端依赖**的视频转录、翻译与字幕项目。默认入口是 **`video_transcribe.py`**（一条命令跑六阶段管线 `pipeline/`：ingest→transcribe→segment→translate→subtitle→burn）；60 分钟真实视频约 7.2 分钟出片（8.29 RTFx，双跑验证）。

`runs/<name>/` 下的产物是跨阶段契约；可用 `--stages` 选择阶段重跑，但 runner 不按产物存在性自动跳过。

## 仓库结构

- **`docs/agents/execution-chain.md`** — 动态流程：文件驱动的数据流、跨阶段契约（路径、SRT 集合、时间戳格式）、幂等实况与门控点。
- 顶层：`video_transcribe.py` 是默认入口；`pipeline/`（runner/stages/asr/llm/codec/registry）是管线本体；`native/asr/` 是自研 FP16 CUDA ASR 引擎源码（`native/llamacpp/` 为 Q4 翻译 LLM 的 llama.cpp STQ 构建脚本）；`bench/` 是一次性基准/取证 harness；`config.example.yaml` 是被跟踪的 schema（默认自包含本地栈：native ASR 引擎 + 托管 HY-MT2 LLM），`config.yaml` 可变、被忽略、可能含密钥。

## 开发流程

1. 改动产物路径、JSON schema、时间戳、排序规则或函数签名之前，先追清它的生产者与消费者；从 `docs/agents/execution-chain.md` 与 `pipeline/registry.py` 入手。
2. 不要在同一份 checkout 里并发跑流水线：它们共享 `runs/` 与显式 `--config`，锁只在进程内有效。
3. 新配置键加到 `config.example.yaml`。管线只通过薄包装读取只读 `--config`，运行期不写 `config.yaml`；无关工作中不要覆盖本地 `config.yaml`。
4. 改动保持聚焦。汇报受影响的阶段、验证证据，以及任何未测试的模型、GPU、API 或依赖媒体文件的路径。
5. 远端为 GitHub wangxiancao/video-transcribe：改动走本地 feature 分支，验证通过后合入 `main` 并推送，提交信息聚焦可读。

## 编码准则

- LLM 调用走 `pipeline/llm/` 的客户端与托管服务（OpenAI 兼容；Q4 llama.cpp 为默认 runtime，BF16 transformers 为回退）；配置访问走 `pipeline/config.py` 薄包装。
- 把中间路径、JSON schema、SRT 顺序和时间戳格式当作 API 对待。保留有序的词级时间戳；对齐保持顺序进行、匹配不完美时尽力而为（best-effort）。
- 交互逻辑不进流水线阶段，供应商特定代码不进编排模块。

## 测试说明

`tests/` 是聚焦测试套件。默认 `.venv/bin/python -m pytest tests/ -q` 只跳过标记 `slow` 的昂贵真实模型对照；完整验收用 `.venv/bin/python -m pytest tests/ -q --run-slow`。GPU/模型相关测试在资产缺失时自动跳过。阶段级验证按 `docs/agents/verification.md`。

- Python：`.venv/bin/python -m compileall -q video_transcribe.py pipeline`
- 默认入口：`.venv/bin/python video_transcribe.py <video-or-url>`（产物落 `runs/`；阶段可 `--stages` 选择；`--validate-only` 只验产物）
- 字幕烧录质量门：`bench/v4_7_burn_gate.py`；Q4 探针：`bench/v4_7_q4_probe.py`

管线不按产物存在性跳过阶段；所选阶段会重新执行，翻译缓存默认开启——重跑不等于干净重算。缺少所需服务与资产时，绝不声称端到端成功。

## 环境搭建

- 用 Python 3.10。FFmpeg 必需；字幕渲染需要 Noto 字体；GPU 推理需要 CUDA 12.x（llama.cpp 构建按 `native/llamacpp/build.sh`）。
- 项目 Python 环境是仓库根的 `.venv`（真实目录；唯一装全依赖集的解释器）。PATH 上的系统/conda 解释器缺依赖——绝不要信任 shell PATH 里的裸 `python3`/`pip`。
- 该 venv 没有 pip；用 `uv pip install --python .venv/bin/python <pkg>` 往里装包。
- 把 `config.example.yaml` 复制成 `config.yaml`；绝不提交或暴露其本地值。
- 模型权重一次性预下载：GLM-ASR 与 ForcedAligner（ModelScope 缓存 `~/.cache/modelscope`）、HY-MT2-1.8B GGUF（`_model_cache/HY-MT2-1.8B-GGUF/`）；运行期不下载。

## 工具使用规则

- 不要直接修改 `runs/`、`output/`、`history/`、`queue_logs/`、`_model_cache/`、`bench/results/` 里的生成内容；通过产出它的那个阶段重新生成。
- 只有任务明确要求时才新增依赖。

## 已知问题 / 约束

- 无 Web UI；`video_transcribe.py` 是唯一产品入口（如需 UI 属于新特性）。
- 完整验证可能需要网络访问（仅权重预下载）、FFmpeg、可观的 GPU 显存，以及一个真实媒体样本。
- 共享 GPU 上：native ASR 引擎按空闲显存动态下预算；正式基准按惯例独占 GPU。

## Agent skills

### 领域文档

单上下文：仓库根一份 `GLOSSARY.md` 加 `docs/adr/`。见 `docs/agents/domain.md`。

### 验证

阶段级验证按 `docs/agents/verification.md`；工具层复用判定见 `docs/agents/tool-audit.md`；夹具约定见 `tests/fixtures/README.md`。

### Issue 仓库

GitHub：`wangxiancao/video-transcribe` 的 Issues，用 `gh` CLI 操作。见 `docs/agents/issue-tracker.md`。

### 分诊标签

默认五角色：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。见 `docs/agents/triage-labels.md`。
