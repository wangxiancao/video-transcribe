# Tests

测试覆盖 runner、产物契约、六阶段编排、本地 LLM 生命周期、多副本轮询、native ASR 和真实媒体烧录。单元/阶段测试使用小夹具；需要模型或 GPU 的测试在资产缺失时自动跳过。

## 默认快速回归

```bash
.venv/bin/python -m pytest tests/ -q
```

默认只跳过标记为 `slow` 的昂贵真实模型对照测试，可加 `-rs` 查看跳过原因。其余 native CUDA、权重加载、mel、SRT 和烧录测试仍会运行。查看耗时分布：

```bash
.venv/bin/python -m pytest tests/ -q --durations=15
```

## 完整验收

```bash
.venv/bin/python -m pytest tests/ -q --run-slow
```

`--run-slow` 额外执行全权重哈希/设备加载对照，以及 native FP16 对 transformers BF16 的真实转录对照，需要本地 GLM-ASR 权重、已构建的 native 二进制和可用 CUDA。完整验收可能需要数分钟；缺资产或未加该选项产生的 skip 不能算作真实模型验证通过。

## 性能基准

RTFx 基准不属于 pytest，需要 ModelScope 权重、CUDA 和 16 kHz 单声道音频：

```bash
.venv/bin/python -m bench.asr_rtfx preflight --cache-dir /root/.cache/modelscope
```

JSON I/O 基准是独立脚本：

```bash
.venv/bin/python tests/benchmark_json_io.py
```

阶段级验证约定见 `docs/agents/verification.md`，夹具约定见 `tests/fixtures/README.md`。
