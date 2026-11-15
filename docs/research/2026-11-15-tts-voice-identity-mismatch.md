# 调研：配音音色与原片说话人身份脱节（P2）

日期：2026-11-15。触发条件：ADR-0004 的 reopening condition 之一——"voice cloning 成为一等需求"（`docs/adr/0004-f5-tts-with-named-voice-library.md`）。

本文区分两类事实：**仓库事实**（给文件路径，均已在本次调研中复读核实）与**外部调研结论**（给来源 URL，优先官方 repo / 模型卡 / issue / arXiv）。无法从 primary source 验证的论断一律标注"未验证"。

---

## ① 问题定义

**要解决的问题：配音的"人"和原片说话人不是一个人/一种气质。** 三个具体表现：

1. **固定库音色与原片说话人身份无关。** `tts.voice` 是全局唯一音色名（`config.example.yaml` L81–93），`pipeline/stages/tts.py` 的 `_voice()` 在段循环外只解析一次；整条音轨从头到尾是同一个库音色（`zh-female-01` 等，ADR-0004）。
2. **多说话人视频也只有单一配音音色。** `translation.json`/`dub_plan.json` 契约里没有任何 speaker 字段（`docs/agents/execution-chain.md` L38、L40；`pipeline/stages/tts.py` 写 plan 处只记录顶层单一 `voice`）。
3. **气质（沉稳/激昂/年龄感）不匹配。** 音色库只有 4 个最低配置音色，manifest（`voices.json`）只有 `name/ref_audio/ref_text/language` 四个字段，没有性别、年龄、语速、气质等可检索元数据（`pipeline/tts/voices.py`）。ADR-0005 已实测 F5 会继承参考片段的语速（`zh-female-01` 2.4–2.6 字/s vs `zh-male-01` 5.2–5.5 字/s）——气质同样随参考片段走，但库里没有按气质选音色的通道。

**已被满足的前置条件（仓库事实）：** ASR/对齐阶段已产出有序词级时间戳（`transcript.json` 词含 `text/start/end`，`docs/agents/execution-chain.md` L36）——说话人归属可以直接落在已有词级时间轴上，不需要重做 ASR。合成缓存键已含音色名（`pipeline/tts/cache.py`：voice+text+speed），多音色合成天然不撞缓存。引擎接缝已存在（`pipeline/tts/factory.py`：为"未来引擎选择不触碰 stage"而设）。

**不可破坏的约束（仓库事实）：** 零云端（ADR-0004 排除 Edge TTS；`docs/agents/execution-chain.md` L68 "不自动切换云端"）；验收门 TTS E2E RTF ≤ 0.5（ADR-0004 记 F5 stock PyTorch RTF 0.147 达标）；1 小时无音色漂移门；slot fitting atempo ≤ 1.25×、累计溢出 ≤ 2%（ADR-0003/0005）；`dub_plan.json`/`tts.wav` 是跨阶段契约（AGENTS.md：中间 JSON schema 当 API 对待）。

---

## ② 解决方案分路线

### 路线 A：不换引擎 —— 说话人区分 + 音色库映射 + 自动匹配

**机制：** 离线说话人日志（diarization）跑在 `media/audio.wav` 上 → 用已有的词级时间戳把 speaker 标签映射到 `segments.json` 每一段（overlap-aware，whisperX 的 `assign_word_speakers` 就是这个先例）→ 按说话人从音色库选音色（人工映射或自动匹配）→ tts stage 按段选音色合成。气质匹配靠：扩充音色库 + 给 manifest 加气质元数据 + 用 speaker embedding（原片说话人 vs 库内参考片段的余弦相似度）或音高/语速特征做最近邻匹配。

**关键组件（外部调研，primary 来源）：**

| 组件 | 结论 | 许可证 | 来源 |
|---|---|---|---|
| pyannote.audio 4.x + `speaker-diarization-community-1` | 当前开源最优离线 pipeline，比 3.1 在说话人计数/归属上显著改进（DER：AISHELL-4 11.7、DIHARD-3 20.2、VoxConverse 11.2，不跳过重叠语音）；powerset 分割原生建模重叠说话人；**本地运行**（precision-2 才是云服务） | 代码 MIT；community-1 权重 **CC-BY-4.0**；模型 gated（一次性 HF 接受条款 + token + 预下载，运行期完全本地） | https://github.com/pyannote/pyannote-audio ；https://huggingface.co/pyannote/speaker-diarization-community-1 |
| pyannote `speaker-diarization-3.1`（备选，生态最熟） | 兼容 pyannote.audio 3.1+，纯 PyTorch；DER 略差于 community-1 | 模型卡明言 MIT + gated | https://huggingface.co/pyannote/speaker-diarization-3.1 |
| whisperX（集成先例，非依赖项） | `assign_word_speakers(diarize_segments, result)` 把 pyannote 结果按词级时间戳合并；官方自述 "Diarization is far from perfect"、重叠语音处理弱 | BSD-2-Clause（LICENSE 已核） | https://github.com/m-bain/whisperX |
| diart | overlap-aware **在线** diarization（IEEE ASRU 2021 官方实现）；但我们管线是离线批处理，在线框架无优势；README 建议 pyannote.audio<3.1（issue #214），与 4.x 生态有脱节 | MIT | https://github.com/juanmc2005/diart |
| NVIDIA NeMo Sortformer | 端到端 diarization、免聚类、支持流式；上限 4 说话人、以英语为主 | 框架 Apache-2.0；权重 CC-BY-4.0（经检索，未逐卡核验） | https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1 |
| speechbrain ECAPA-TDNN | speaker embedding + 余弦相似度验证（VoxCeleb1-O EER 0.80%）；做"原片说话人↔库音色"相似度匹配的成熟件 | Apache-2.0（模型卡） | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb |
| Resemblyzer | 256 维 GE2E 嵌入，GTX 1080 上约 1000× 实时，CPU 可跑；官方自述对英语最稳 | Apache-2.0 | https://github.com/resemble-ai/Resemblyzer |
| F5-TTS 官方参考片段建议 | 参考音频 <12 s、结尾留约 1 s 静音；单次生成（含参考）上限 30 s；数字需预处理成中文读法——扩充音色库时照此执行 | — | https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md |

**集成点（仓库）：**
- 新阶段 `diarize`（transcribe 之后、segment 之前；或并入 transcribe 可选步骤）：产物如 `speakers.json`；`segments.json`/`translation.json` schema 增 `speaker` 字段（**契约变更**，schema_version 与 validator 需同步，AGENTS.md 视 schema 为 API）。
- `pipeline/stages/tts.py`：把 `_voice()` 从循环外移到按段解析（依据 `segment["speaker"]` + 映射）；`dub_plan.json` 每段增 `voice` 字段（契约变更）。缓存键无需改。
- 新配置键（沿用 `tts.*` 扁平与 `translation.term_map` 的 map 风格）：`diarization.enabled`、`diarization.backend: 'pyannote'`、`diarization.max_speakers: 0`（0=自动）、`tts.voice_map`（speaker→voice 名）、`tts.voice_matching.method: 'embedding'|'pitch'|'off'`。
- 音色库扩充：`_model_cache/tts_voices/voices.json` 条目加 `tags`（性别/年龄/气质/语速）元数据——ADR-0005 已证语速是可测量属性（字/s），气质标签可先人工标注。

**零云合规：** 合规。pyannote/ECAPA 均本地推理；唯一边界是 pyannote 模型 gated——需要**一次性**联网接受条款、建 token 并预下载权重，与仓库"权重一次性预下载、运行期不下载"的既有政策（AGENTS.md）同构；运行期无任何网络调用。

**RTF/显存影响：** diarization 是对整条 audio.wav 的一次性前向（GPU 上远快于实时——具体数值**未验证**）；ECAPA/Resemblyzer 嵌入提取是毫秒级（Resemblyzer 官方 ~1000× 实时）。TTS 合成本身不变，不触碰 RTF ≤ 0.5 门。

**1h 长跑风险：** 无新增漂移风险——F5 逐段从固定参考片段合成（ADR-0004 的设计本来就是抗漂移的），多音色只是多几个固定参考。风险在 diarization 的一致性（同一说话人在长视频中被分裂/合并），community-1 的计数改进正是针对此，但 1h 实测**未做**。

---

### 路线 B：换/增零样本跨语言克隆引擎（原片说话人几秒语音 → 直接合成中文）

**机制：** 从原片抽取说话人参考音频（几秒干净语音，可复用 diarization 结果挑最干净片段），用零样本克隆 TTS 直接以该音色合成中文。以下逐一核实（primary 来源）：

| 引擎 | 克隆/跨语言声明 | 已知 issue | 许可证 | RTF | ADR-0004 拒绝理由现状 |
|---|---|---|---|---|---|
| **F5-TTS**（现役，换参考即克隆） | 官方不主打跨语言；Emilia 中英码切换样本多为口音较轻的 YouTube 内容 | **issue #315（已核实）**：跨语言克隆携带参考音频口音（EN ref→其他语言带英语口音）；维护者原话 "Accent is just part of speech style, which is somehow different from timbre"，建议**用目标语言的参考音频**；已关 | 代码 MIT；Emilia 权重 CC-BY-NC | PyTorch 0.147（官方 L20 基准 0.1467） | 结论：**EN ref→ZH out 直接克隆有官方确认的口音泄漏**，"换参考片段"路线在跨语言方向不成立，需目标语言参考或换引擎。https://github.com/SWivid/F5-TTS/issues/315 |
| **Cross-Lingual F5-TTS**（第三方变体） | 论文《Towards Language-Agnostic Voice Cloning and Speech Synthesis》（arXiv 2509.14579，2025-09）；HF 卡含中/英/多语变体与语速预测器 | 质量与 1h 漂移未评测 | HF 卡标 Apache-2.0；**但训练数据是 Emilia（数据集本身 CC BY-NC-4.0，F5 官方因此把自家权重定为 NC）**，该标注与官方立场矛盾，需自查 | 未提供 | 第三方权重，质量/长跑全部**未验证**。https://huggingface.co/QingyuLiu1/Cross-Lingual_F5-TTS ；https://arxiv.org/abs/2509.14579 |
| **IndexTTS-2 / 2.5**（bilibili） | 单参考片段零样本克隆；中/英/日/西/阿 5 语言；官方声明保留 "cross-lingual and timbre-emotion disentanglement"；**官方跨语言评测方向是"中文 prompt→其他语言"**，EN ref→ZH out 方向 README 未列表（**未验证**）；IndexTTS-2 有 8 维情绪向量 + emo_alpha(0–1) + 情绪参考音频/文本 | 官方 README 注意事项：随机情绪采样降低克隆保真度、拼音控制不覆盖所有音节组合、CUDA 12.8+ | **bilibili Model Use License Agreement（自定义，商用需联系 indexspeech@bilibili.com）**；安装链最重（ADR-0004 拒绝理由之一，未变） | 2.5 bf16 ≈ **0.21**（RTX 4090，官方）——满足 ≤0.5 门 | RTF 拒绝理由不成立（0.21 达标）；"安装链重 + 自定义许可证"仍在。https://github.com/index-tts/index-tts |
| **CosyVoice2 / Fun-CosyVoice3-0.5B-2512** | 官方：跨语言零样本克隆、9 语言 + 18 中文方言、instruct 控制（语言/方言/情绪/语速/音量）、150ms 流式延迟 | **issue #1420（已核实）**：cosyvoice2-0.5B 乱读/重复；社区结论 >15 s 生成会加重重复，建议换 inference_instruct2 路径；2025-07 关闭，无代码修复。**注意：本管线 `segmentation.max_duration_s: 8.0`（config.example.yaml L37）逐段合成本来就 <15 s，ADR 拒绝理由与现管线设计存在张力** | repo Apache-2.0；Fun-CosyVoice3 权重经检索称 Apache-2.0（ModelScope 页未直接核验，**未验证**） | 仓库实测 ~0.70（ADR-0004）——**仍超 0.5 门**；`_model_cache/cosyvoice3/` 有未用遗留权重 | RTF 拒绝理由（未变）成立；长重复问题被 8s 分段缓解但漂移门仍需实测。https://github.com/FunAudioLLM/CosyVoice/issues/1420 ；https://github.com/FunAudioLLM/CosyVoice |
| **Qwen3-TTS 系列**（0.6B/1.7B × Base/CustomVoice/VoiceDesign） | Base：**3 秒**参考快速克隆（+转写文本；`x_vector_only_mode` 可无转写但质量下降）；10 语言含中文；CustomVoice 9 高级音色 + 指令风格控制；VoiceDesign 从文字描述设计音色 | vLLM 仅离线推理；FA2 依赖；社区报 unwanted laughter 类 artifacts（**未验证**） | **Apache-2.0** | 官方 README 无 RTF/VRAM；仓库实测（ADR-0004）无 FA2 时 RTF 2.0——**远超 0.5 门**；社区 Rust/ONNX 实现报 0.6（issue #187，**未验证**），仍超标 | RTF 拒绝理由（未变）成立。https://github.com/QwenLM/Qwen3-TTS |
| **GPT-SoVITS**（V5/V2ProPlus） | "5 秒样本即时 TTS"零样本；官方支持跨数据集语言推理（英/日/韩/粤/中）→ EN ref→ZH out 是声明特性；1 分钟微调提相似度 | **issue #1519（已核实）**：2024-08 报逐段 VRAM 累积（2GB→29GB）；维护者 2025-06 复现结论是"推理 batch 过大（batch=20 可塞进 24GB）"，2025-06-27 关闭，**无代码修复**。本管线逐段合成即 batch=1，命中该失败模式的概率低，但 1h 连续运行仍无实测 | **MIT** | v2ProPlus：0.014（4090）/0.028（4060Ti，官方）——最快 | **OOM 拒绝理由已被官方关闭结论削弱**（归因 batch 配置而非固有泄漏）；MIT + 最快 RTF，成为重开候选，但 1h 漂移/显存需专项 bench。https://github.com/RVC-Boss/GPT-SoVITS/issues/1519 ；https://github.com/RVC-Boss/GPT-SoVITS |
| **XTTS-v2**（Coqui） | 17 语言含 zh-cn；6 秒片段跨语言克隆 | 公司 2024 年初关停 | **CPML（非商用，且连输出音频也受限）**；代码 MPL；无新商用授权渠道（issue #3490） | 官方无 RTF | **许可证即否决**，零云/许可双查不通过。https://huggingface.co/coqui/XTTS-v2 ；https://github.com/coqui-ai/TTS/issues/3490 |

**集成点（仓库）：** `pipeline/tts/factory.py` 接缝即为此设计——加 `tts.engine: 'f5'|'gpt_sovits'|'indextts'` 选择键 + 各引擎资产预下载路径键（沿用 `_model_cache/` 惯例）；克隆参考来源 `tts.clone_reference: 'library'|'source_speaker'`。注意 `dub_plan.json` 需记录引擎与参考来源（契约变更）；ADR-0004 需走新 ADR 修订（其 reopening condition 已被本次触发）。

**零云合规：** 上表引擎全部本地推理，合规（纯云 API 的 SiliconFlow CosyVoice2、ElevenLabs 等一律不符约束，不展开）。

**RTF/显存：** 见表。达标者仅 IndexTTS-2.5（0.21）与 GPT-SoVITS（0.014）；两者均为**自回归**架构——ADR-0004 选 F5 的核心理由是"非自回归对重复/漂移累积免疫"。但本管线逐段独立调用 + 每段重置上下文（与 F5 相同的逐段设计）会削弱该风险，需 1h 实测而非纸面推断。

**1h 长跑风险：** 自回归引擎的最大不确定点：GPT-SoVITS 的历史 OOM（已削弱未消除）、IndexTTS 无公开 1h 证据、Qwen3-TTS RTF 不达标。全部标注：**需实测**。

---

### 路线 C：TTS + 音色转换（VC）两段式 —— 保 F5 节奏稳定性，后置换音色

**机制：** F5 按现有流程逐段合成中文（节奏/语速/落槽全部沿用 ADR-0003/0005 机制）→ VC 模型把合成语音的音色转换为原片说话人音色（源=F5 输出，参考=原片说话人几秒语音），保留源语音的节奏与内容。最贴近该机制的是 OpenVoice v2：官方架构即 "base speaker（负责节奏/情绪/口音）+ tone color converter（只换音色）"。

| VC 方案 | 机制与质量 | 许可证 | 状态 | 来源 |
|---|---|---|---|---|
| **OpenVoice v2**（MyShell/MIT） | 零样本 tone color conversion；v2 原生多语言**含中文**（英/西/法/中/日/韩）+ 音质改进；v2 base speaker 是 MeloTTS——**本管线用 F5 替代 base speaker，converter 单独用**；官方无 RTF 数（社区共识 converter 轻量、GPU 亚实时、CPU 可用，**未验证**）；NAACL 2024 论文 | **MIT，"Free for both commercial and research use"** | 活跃 | https://github.com/myshell-ai/OpenVoice ；论文 https://arxiv.org/abs/2312.01479 |
| **Seed-VC** | 1–30 s 参考零样本 VC + 实时流式 + 歌声 VC；V2 "best in suppressing source speaker traits"，有 similarity-cfg-rate 与口音/情绪转换；官方无整体 RTF 数（流式 chunk 推理 150ms 级） | **GPL-3.0**（传染性：若分发衍生代码需开源；仅内部使用不触发分发义务，需自查） | **仓库页面显示已归档（2025-11-21，read-only）**——维护风险 | https://github.com/Plachtaa/seed-vc |
| **kNN-VC** | 非参数：WavLM 特征 k 近邻匹配 + HiFi-GAN；无需训练；官方建议参考音频累计**最长 5 分钟**才达最佳质量——与"几秒参考"场景相悖；评测仅英语 | MIT（LICENSE 已核，Stellenbosch） | 学术 | https://github.com/bshall/knn-vc |
| **so-vits-svc** | 歌声转换框架，**需逐音色训练**、无零样本；条款限学术用途；不面向配音 | AGPL-3.0 | **2023-11 归档** | https://github.com/svc-develop-team/so-vits-svc |

**两段式 vs 直接克隆的 tradeoff（分析，非引用）：**
- 优点：F5 的全部既有资产保留——RTF 0.147、抗漂移（非自回归）、逐段落槽逻辑、合成缓存；VC 是无状态逐段后处理，1h 长跑漂移风险低；许可证最优组合（F5 CC-BY-NC 输出 + OpenVoice MIT 工具，个人场景已接受 NC）。
- 缺点：音色相似度通常低于直接克隆（converter 只换音色不重建发声细节）；converter 对 F5 24 kHz 中文输出的实际效果、对情绪起伏语音的保真度**未验证**；总 RTF = F5 RTF + converter RTF（converter 未知，需 bench，预算上有 F5 0.147 的余量：0.5−0.147≈0.35）。

**集成点（仓库）：** `pipeline/stages/tts.py` 段循环内、`engine.synth` 之后 `plan_fit` 之前插 VC pass（缓存键需加入"是否 VC + VC 参考标识"，`pipeline/tts/cache.py` 契约变更）；新配置键 `tts.voice_convert.enabled`、`tts.voice_convert.backend: 'openvoice_v2'`、`tts.voice_convert.target: 'source_speaker'`；`dub_plan.json` 每段记录 VC 决策（契约变更）。VC 参考音频由 diarization 产物挑取（依赖路线 A 的 `speakers.json`）。

**零云合规：** OpenVoice/Seed-VC/kNN-VC 均本地。**许可证：** 见表。**1h 长跑：** VC 逐段无状态，理论风险最低；F5 部分已有 1h 证据（ADR-0004 双跑验证），VC 部分需实测（音色一致性 across 段）。

---

### 路线 D：气质/情绪控制

**机制与来源：**

1. **按气质选参考片段（F5 现役即可用）：** F5 继承参考的语速与风格是仓库已测事实（ADR-0005 语速实测；官方 infer README 的参考片段选择指南：<12 s、结尾留静音）。做法：音色库扩容 + `voices.json` 加气质标签（沉稳/激昂/年龄感/字每秒），人工或自动选片段。零成本、零风险。
2. **说话人→音色自动匹配：** 用 ECAPA-TDNN/Resemblyzer 嵌入对"原片说话人 vs 库内参考片段"算余弦相似度最近邻（两模型卡均以余弦相似度为官方用法），辅以音高（F0）/语速聚类兜底（嵌入模型以英语/VoxCeleb 为主，中文跨语言嵌入质量**未验证**，音高兜底更稳）。
3. **情绪控制引擎（需路线 B 前置）：** IndexTTS-2 官方情绪控制——8 维情绪向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]、`emo_alpha` 强度 0–1、情绪参考音频/文本四通道（https://github.com/index-tts/index-tts）；CosyVoice instruct 模式官方语义——"languages, dialects, emotions, speed, volume" 指令（https://github.com/FunAudioLLM/CosyVoice）。
4. **本地 voice design（从文字描述生成音色）：** **Qwen3-TTS-12Hz-1.7B-VoiceDesign** 官方支持 free-form 自然语言音色描述（"what you imagine is what you hear"），Apache-2.0，10 语言含中文（https://github.com/QwenLM/Qwen3-TTS）——是唯一核实的中文可用本地 voice design；但 Base 系 RTF 2.0 超门（ADR-0004 仓库实测），VoiceDesign 变体的 RTF/质量**未验证**，且描述生成的是"新音色"而非"原片说话人音色"，解决的是气质不匹配而非身份克隆。Parler-TTS 同为文字描述控制但**仅英语**（https://github.com/huggingface/parler-tts，Apache-2.0，经检索）——不适用于中文输出。

**集成点：** 主要是数据与配置而非代码——`voices.json` schema 扩展（`tags`/`pace_chars_per_s`）+ `tts.voice_matching.method`；情绪向量若走 IndexTTS 需路线 B 的引擎键 + `tts.emotion.*`。

---

### 先例（secondary 来源，标注：非 primary 验证）

- **SoniTranslate**（Apache-2.0）：pyannote diarization + "TTS now supports up to 12 speakers" + 手动说话人编辑；音色模仿用 XTTS/OpenVoiceV2/freevc/RVC，TTS 还含 Edge TTS（云，不符本仓库约束）。多说话人配音的完整先例。https://github.com/R3gm/SoniTranslate
- **VideoLingo**（Apache-2.0）：README 限制一节明言 "The dubbing workflow does not automatically assign a separate voice to each speaker"——即同类产品同样卡在多说话人分配，无现成方案可抄。TTS 后端含 GPT-SoVITS/F5-TTS；SiliconFlow CosyVoice2 是云 API（不符约束）。https://github.com/Huanshere/VideoLingo

---

## ③ 路线对比表

| 维度 | A. diarization+音色映射 | B. 换克隆引擎 | C. F5+VC 两段式 | D. 气质/情绪控制 |
|---|---|---|---|---|
| 解决"多说话人单一音色" | **是（根因）** | 是（附带） | 是（需 A 提供参考） | 部分（仅气质） |
| 解决"音色=原片说话人" | 否（只能近似匹配库音色） | **是（直接克隆）** | **是（VC 换音色）** | 否 |
| 解决"气质匹配" | 部分（标签+嵌入匹配） | 部分（情绪控制可叠加） | 部分（继承 F5 风格） | **是（选片/描述/情绪）** |
| 引擎变更 | 无 | 换/增引擎（factory 接缝） | 无（加后处理） | 无（或随 B） |
| 契约变更 | transcript/segments/translation + dub_plan 增 speaker/voice | dub_plan 增 engine/reference | 缓存键 + dub_plan 增 VC 字段 | voices.json schema |
| RTF 影响 | ≈0（TTS 不变） | IndexTTS 0.21 / GPT-SoVITS 0.014 达标；Qwen 2.0、CosyVoice 0.70 超标 | +converter RTF（未知，预算余量 0.35） | 0 |
| 1h 漂移风险 | 无新增 | 自回归架构，需实测（ADR-0004 核心担忧） | 低（VC 无状态；F5 部分已验证） | 无 |
| 许可证 | pyannote CC-BY-4.0(gated)/MIT；ECAPA Apache-2.0 | GPT-SoVITS MIT > IndexTTS 自定义 > Qwen Apache-2.0(RTF 不达标) > XTTS CPML(否决) > CosyVoice(未验证) | **OpenVoice MIT 最优**；Seed-VC GPL/归档 | 全部宽松 |
| 零云合规 | 合规（pyannote 需一次性 gated 预下载） | 合规 | 合规 | 合规 |
| 工程量 | 中（新阶段+契约+匹配器） | 高（新引擎资产+bench+ADR 修订） | 中（后处理 pass+bench） | 低 |
| 主要未知 | diarization 1h 一致性；中文嵌入质量 | EN ref→ZH 质量（除 F5 #315 外均缺官方方向数据）；1h 显存 | converter 对 F5 中文的实际效果与 RTF | VoiceDesign RTF/质量 |

---

## ④ 建议优先级

1. **先做 A（diarization + 按说话人映射音色 + 库扩容/气质标签 + 嵌入自动匹配）。** 理由：直击"多说话人单一音色"这一最尖锐表现；不动引擎、不碰 RTF/漂移门；pyannote community-1 的词级归属有 whisperX 先例可直接抄；音色库扩容与 F5 官方参考片段指南完全兼容。A 同时是 B/C 的前置（克隆/VC 都需要"哪段是谁"的答案与干净参考片段）。
2. **其次做 C（OpenVoice v2 converter 后置）。** 理由：许可证（MIT）与零云最优；保留 F5 全部已验证资产（RTF、抗漂移、落槽、缓存）；一次性解决"音色=原片说话人"；工程量集中在 bench（converter 对 F5 中文输出的相似度/RTF 实测）。
3. **B 仅在 C 实测音色相似度不达标时启动**，首选候选 **GPT-SoVITS**（MIT、RTF 0.014、#1519 已被官方关闭结论削弱为 batch 配置问题——ADR-0004 拒绝理由需重审）与 **IndexTTS-2.5**（RTF 0.21 达标、克隆口碑好；自定义许可证与重安装链是既有反对项）；无论选谁都必须过 1h 漂移/显存专项 bench + 新 ADR。**Qwen3-TTS（RTF 2.0）、CosyVoice3（RTF 0.70）、XTTS-v2（CPML）维持拒绝。**
4. **D 与 A 合并交付**（气质标签 + 匹配），情绪向量控制仅在 B 落地后追加；VoiceDesign 记为探索项不排期。

---

## ⑤ 未验证/存疑项清单

1. pyannote community-1 在 1h 真实视频上的说话人一致性/DER/耗时/显存（官方基准是数据集 DER，非长视频稳定性）——未验证。
2. ECAPA-TDNN/Resemblyzer 嵌入在中文语音上的区分度（两模型均为英语/VoxCeleb 训练）——未验证；建议音高/语速特征兜底。
3. Fun-CosyVoice3-0.5B-2512 权重许可证（检索称 Apache-2.0，ModelScope 模型卡原文未核验成功）——未验证。
4. IndexTTS-2.5 "EN ref→ZH out" 方向的官方质量数据（README 跨语言表为 ZH prompt→其他语言）——未验证。
5. IndexTTS-2.5 README News 所列 2.5 发布日期（抓取为 2026-08-10）未二次核验；不影响能力/许可证结论。
6. OpenVoice v2 tone color converter 的 RTF（官方无数字）与对 F5-TTS 24 kHz 中文输出的音色相似度/情绪保真——未验证，是路线 C 的核心待 bench 项。
7. Seed-VC 归档状态（仓库页面显示 2025-11-21 起只读）与 GPL-3.0 对本仓库分发场景的实际约束——未验证/需自查。
8. Qwen3-TTS 社区 Rust/ONNX 实现 RTF 0.6（issue #187）与 "unwanted laughter" artifacts——均未验证。
9. GPT-SoVITS 在本管线逐段（batch=1）调用下的 1h 显存曲线与音色稳定性——未验证（#1519 官方关闭结论是 batch=20 场景的复现）。
10. Cross-Lingual F5-TTS（arXiv 2509.14579）第三方权重的质量、1h 漂移，以及"Apache-2.0 checkpoint 却用 CC BY-NC 的 Emilia 数据训练"的许可证矛盾——未验证，采用前需法律与质量双查。
11. NeMo Sortformer 各权重卡是否 gated、中文会议语料表现——未验证。
12. diart 对 pyannote.audio 4.x 的兼容（README 建议 <3.1）与维护活跃度——未验证。
13. F5-TTS Emilia 权重 CC-BY-NC-4.0 与商业化的关系为 ADR-0004 已接受项；若未来商用触发 reopening（本文不重开该议题）。
