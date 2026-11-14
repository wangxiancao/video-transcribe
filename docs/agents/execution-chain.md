# video-transcribe 核心执行链

当前产品默认是一条六阶段、文件驱动的单向流水线。唯一产品入口是 `video_transcribe.py`；阶段注册表是 `pipeline/registry.py`，运行器是 `pipeline/runner.py`。旧版 Whisper、Excel、spaCy 和猴补丁链路已经移除；配音（tts，dub 后续接入）按 ADR-0003/0004 重建，只在显式 `--stages` 时执行，六阶段默认行为不变。

## 数据流与产物

```
输入（本地视频 | HTTP(S) URL）
 └─ ingest     → media/source.mp4 + media/audio.wav
    └─ transcribe → transcript.json（ASR 文本 + 有序词级时间戳）
       └─ segment  → segments.json（规则断句 + 低信息片段过滤）
          └─ translate → translation.json（忠实翻译 + 润色）
             └─ subtitle → subs/{source,target,bilingual}.srt
                ├─ burn   → output.mp4（烧录 target.srt）
                └─ tts    → tts/dub_plan.json + tts/tts.wav（显式 --stages）
                   └─ dub → output_dub.mp4（整轨替换，后续票）
```

| 阶段 | 输入 | 输出 | 实现要点 |
|---|---|---|---|
| ingest | `--source` | `media/source.mp4`、`media/audio.wav` | yt-dlp 或本地文件；FFmpeg 规范化视频，提取 16 kHz、单声道、16-bit PCM |
| transcribe | `media/audio.wav` | `transcript.json` | 默认自研 FP16 CUDA GLM-ASR；Qwen3-ForcedAligner 走 transformers；保留原始 ASR 证据 |
| segment | `transcript.json` | `segments.json` | 标点、静音、词数、时长规则，不调用 LLM；默认过滤独立填充词和病态重复并重新编号 |
| translate | `segments.json` | `translation.json` | HY-MT2 两轮翻译；有序字符串数组协议；时间戳逐项复制 |
| subtitle | `translation.json` | 三份 SRT | 源文、译文按同一时间轴拆分；JSON/SRT 原子写入 |
| tts | `translation.json`、`media/audio.wav` | `tts/dub_plan.json`、`tts/tts.wav` | 逐段 F5-TTS 合成（命名音色库，ADR-0004）；槽内 ffmpeg atempo 变速上限 1.25×（ADR-0003），超长溢出记账不截断；plan 时间戳全部来自实测音频 |
| burn | 源视频、`target.srt` | `output.mp4` | 只烧录目标语言字幕；支持关键帧并行、libx264/h264_nvenc、时长校验 |

runner 只执行 `--stages` 明确选择的阶段，并始终按注册表顺序执行。它不会自动补跑前置阶段，也不会因为产物存在而跳过阶段。重跑会重新计算所选阶段；翻译 LLM 响应缓存默认开启，因此重跑不等于完全无缓存计算。

## 跨阶段契约

所有路径相对于 `--workdir`。JSON 当前 `schema_version` 均为 1。阶段执行前校验输入，执行后校验产出；缺输入返回 2，阶段或契约失败返回 1。

- `transcript.json`：`language` + `segments[]`；每段含 `text/start/end/words[]`，词含 `text/start/end`。词序和开始时间不得回退，结束时间必须大于开始时间。
- `segments.json`：在上述词级契约上增加从 1 开始的连续 `id`。过滤会删除整段，因此这里不再保证包含原始 transcript 的每一个词。
- `translation.json`：`source_language/target_language` + `segments[]`；每段含 `id/start/end/source/target`。段数、ID 和时间戳必须与 `segments.json` 一致。
- SRT：`HH:MM:SS,mmm --> HH:MM:SS,mmm`，编号连续、时间正时长且不得重叠。三份 SRT cue 数和时间轴一致；bilingual 每 cue 恰好源文、译文两行。
- `tts/dub_plan.json`：`input/voice/sample_rate/source_duration_s/track_duration_s` + `segments[]`（每段 `id/slot_start/slot_end/text/synthesized_duration_s/speed_factor/method/placed_start/placed_end/overflow_s`）+ `overflow`（`count/total_s/ratio`）。段数、ID、槽时间戳与 `translation.json` 一致（`tts.input` 决定 `text` 对应 `target` 还是 `source` 字段）；`method ∈ {none, atempo}`；`speed_factor ≤ 1.25`（固定契约，不进配置）；`placed_start = slot_start`（不截断不位移）；`overflow_s = max(0, placed_end - slot_end)`。
- `tts/tts.wav`：24 kHz 单声道 PCM16 整轨；静音底长度等于源音频，末段溢出可延长整轨；时长与 plan `track_duration_s` 相差不超过 50 ms；非语音区间保持静音。
- 视频：烧录输出必须同时含音视频流，时长与源视频相差不超过 0.5 秒。

## 文本质量门

`pipeline/text_quality.py` 是当前确定性质量规则的唯一实现。

1. `transcribe` 不过滤原始文本，便于审计 ASR。
2. `segment` 默认开启 `segmentation.filter_low_information`：删除独立“嗯、呃、啊、um、uh”等填充段，以及至少四个相同字或短语连续重复至少三次的病态片段。“对、没有、是”等短回答不因长度被删除。
3. `translate` 拒绝含“自然流畅、目标语言、副标题、将原文润色”等翻译器自述/指令泄漏的响应，并进行重试和必要时的批次拆分。
4. 润色失败或超过 88 字符时，只能回退到已经通过质量门的忠实译文并记录警告。忠实翻译也失败则整个阶段失败；不得回显源文，不得静默截断，不得继续生成成功字幕。

这些规则不能识别所有语法通顺但内容错误的 ASR 结果；真实样本验收仍需检查识别和翻译质量。

## 本地 LLM 与缓存

只要本次选择了 `translate` 且 `llm.local.enabled=true`，runner 就创建托管服务，并在 `finally` 中停止全部副本。服务支持 llama.cpp Q4 和 transformers BF16 两种 runtime；多副本通过轮询客户端分担并发请求。

托管服务客户端使用 `trust_env=False`，Clash 等工具设置的 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 不会介入本地 HTTP 请求。只有显式禁用本地服务后使用的外部 `api.*` 客户端才继承环境代理。

翻译缓存位于 `<workdir>/.llm-cache/`，键由 system prompt 和请求正文计算。解析失败的缓存响应会在重试前删除。

TTS 合成缓存在 `<workdir>/.tts-cache/`，键 = 音色 + 段文本 + 合成速度（ADR-0004：拟合发生在合成之后，合成始终按原生速度）；缓存命中时重跑不加载模型、不访问 GPU。

## 门控与失败行为

- 媒体缺失、模型缺失、 native 二进制缺失、LLM 服务启动失败均显式报错，不自动切换云端。
- ASR/对齐器输出必须满足词级时间戳契约；异常越界词按现有 best-effort 规则钳制并警告，严重回退或空结果失败。
- 分段全部过滤、忠实翻译无合规结果、SRT 契约破坏或烧录时长漂移都会使运行非零退出。
- tts 的变速上限 1.25× 是固定契约；超长溢出不判阶段失败，逐段记账进 plan 并在 stderr 告警，累计溢出（≤总时长 2%）由 bench 验收门量化判定。
- `output.mp4` 使用临时文件，验证通过后才原子替换。不要在同一 workdir 并发运行流水线。
