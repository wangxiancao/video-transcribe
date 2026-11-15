# TTS 配音节奏/能量不匹配（P1）调研

日期：2026-11-15。作者：后台调研 agent。性质：桌面调研（未运行流水线、未做任何音频实验）。

本文区分两类论断：

- **[仓库事实]** — 来自本仓库已提交的代码/ADR/实测记录，给文件路径。
- **[外部结论]** — 来自外部 primary source（官方 repo/docs/issues、arXiv 论文、FFmpeg 官方文档/本机 ffmpeg 二进制），给 URL。二手转述一律标注。

---

## ① 问题定义（仓库事实）

源片英文原声从容（慢语速、有留白），中文配音听感赶稿（用户主观感受约 20 字/秒的赶稿感）。ADR-0005 已记录 speed 2.0 时合成语速约 10 字/秒，再叠加 atempo 1.25 封顶与短槽压缩，有效语速可达约 12.5 字/秒，而自然中文语速约 4–6 字/秒（`docs/adr/0005-tts-synthesis-speed-knob.md` 第 11 行给出"natural Mandarin ≈ 4–6 chars/s"）。

相关仓库事实链：

- 管线与契约：tts 阶段逐段合成、ffmpeg atempo 封顶 1.25×、溢出记账不截断；验收门为累计溢出 ≤ 源音频 2%、TTS E2E RTFx ≤ 0.5（RTF 0.147 stock PyTorch，见 `docs/adr/0004-f5-tts-with-named-voice-library.md`）、1 小时无音色漂移（`docs/agents/execution-chain.md`）。
- ADR-0005 实测：默认 `zh-female-01` + speed 1.0 时 1h 真实英语播客 EN→ZH 累计溢出 65.44%；换 `zh-male-01`（参考片段语速 4.7 字/秒，合成实测 5.2–5.5 字/秒）+ `pause_s: 0.8`/`max_words: 14` + speed 2.0 后降到 0.33%；speed 2.0 下 97% 的残余溢出集中在 219 个 <1s 的槽（每段合成固定开销 ~0.5–1s）。ADR-0005 自己的 reopening condition 第一条就是"合成语音在人耳听感门里不可接受地赶"（`docs/adr/0005-tts-synthesis-speed-knob.md` 第 33 行）——本次调研正是对该触发条件的响应。
- 翻译阶段没有任何"时长/长度预算"概念：`pipeline/stages/translate.py` 的 SYSTEM_FAITHFUL / SYSTEM_REFINE 提示只约束忠实与流畅，唯一的长度约束是输出质量门的 88 字符上限（超限回退忠实稿）。
- 分段阶段：`pipeline/stages/segment.py` 以 `pause_s`（默认 0.45）/`max_words`（默认 20）/`max_duration_s`（8.0）做规则断句；ADR-0005 的配音配置把 `pause_s` 提到 0.8 合并短槽。
- 能量/响度：全链路无任何响度处理——`pipeline/stages/tts.py` 把合成段原样叠加进静音底，`pipeline/stages/dub.py` 直接 `aac 96k` 封装（`_mux`，无 `-af`）。
- 引擎封装：`pipeline/tts/engine.py` 的 `synth()` 只透传 `speed/seed/nfe_step/cross_fade_duration` 给 `f5_tts.api.F5TTS.infer`；`fix_duration` 等参数未被使用。
- 本机安装包：f5-tts **1.1.22**（`.venv/lib/python3.10/site-packages/f5_tts/`，MIT 代码许可）；本机 ffmpeg **4.2.7**。

---

## ② 根因分解

四条根因假设全部成立，且相互叠加。前两条有时序证据，后两条是机制推断（能量部分无仓库实测）。

### 根因 1：EN→ZH 翻译文本时长膨胀，译文自然语速所需时长 > 源槽时长

[仓库事实+推导] ADR-0005 的实测可反推净膨胀率：`zh-male-01` 参考语速 4.7 字/秒、speed 1.0 合成约 5.2–5.5 字/秒仍溢出 65.44%；要到 speed 2.0（约 10 字/秒，即自然上限的 ~2 倍）才把溢出压到 2% 量级。这说明按自然语速（4–6 字/秒）朗读译文所需的时长约为源语音槽的 **~2 倍**。这是"翻译文本膨胀 × 引擎合成速率"的联合效应，仓库未单独测量纯文本膨胀分量（见 ⑤）。

[外部结论] 该现象是配音行业与学术界的已知核心问题，有专门文献（isometric MT / dubbing-aware MT，见 ③-A）。

### 根因 2：靠 speed/atempo 硬压音频，而不是控制文本长度

[仓库事实] 现有手段全是"音频侧压缩"：`tts.speed`（合成时长估计的除数）+ atempo 1.25 封顶（`pipeline/tts/fitting.py` 的 `plan_fit`）；文本侧零干预。

[外部结论]
- F5 的 `speed` 官方语义是**时长估计公式的除数**，不是后处理变速：`duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)`（官方 main 分支 `src/f5_tts/infer/utils_infer.py`，https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/utils_infer.py ；本机 1.1.22 同公式）。把时长压到自然时长的一半以下，模型只能在同样的声学容量里塞进全部音素，issue #1343 报告了由此产生的吞音/压缩伪声，维护者关闭时只确认"它确实改总时长"，未否认伪声（https://github.com/SWivid/F5-TTS/issues/1343）。
- 行业反证：对 319.57 小时、54 部专业译制片的大规模测量发现，专业配音**宁可破坏时间约束也不接受不自然的语速**——"When forced to pick one or the other, human dubbers appear more willing to break timing constraints than vary speaking rate"，且译制版语速标准差向 ~1.25 词/秒收窄（源语 ~1.46–1.47 词/秒）；同时字符等长（isometry）与等时性（isochrony）只有弱相关 r=0.279（Dubbing in Practice，arXiv:2212.12137，https://arxiv.org/abs/2212.12137 ，全文 https://arxiv.org/html/2212.12137 ）。
- Amazon 配音架构的韵律对齐打分函数里，目标语速 >2 直接给最低分（"too high speaking rates will result in low quality TTS speech"）（Prosodic Alignment for off-screen automatic dubbing，Interspeech 2022，https://www.isca-archive.org/interspeech_2022/virkar22_interspeech.pdf ）。

本管线 speed 2.0 + atempo 1.25 的有效倍率落在这些文献判定的"低质量区"内——赶稿感是结构性的，不是调参能消除的。

### 根因 3：分段/停顿粒度——ADR-0005 的合并策略吃掉了留白

[仓库事实+机制推断] `segment.py` 的 `_rule_boundaries` 在词间静音 > `pause_s` 处断句；把 `pause_s` 从 0.45 提到 0.8，意味着 **≤0.8s 的自然停顿被并入段内**：合并段的 `slot_start..slot_end` 覆盖原停顿区间（`segment.py` 第 70–76 行取组内首词 start、末词 end），而 F5 对整段文本合成不会在原停顿位置还原停顿——留白变成了连续语流。源片的"从容"很大一部分正是这些 0.3–0.8s 的呼吸位；合并策略用溢出指标换了时序，代价是听感的从容度，此代价未进入任何验收门（ADR-0005 只量化了溢出）。

[外部结论] 行业做法正相反：Amazon 的 prosodic alignment 把译文**切分成短语并在其中插入停顿**去匹配源的短语/停顿结构（源停顿检测阈值 ≥300ms），而不是把停顿吞进语流（同上 Interspeech 2022 论文 §4；其 2020 年前作首次提出该模型）。Netflix 非虚构类配音准则同样要求对齐"start and finish of each dialogue event (including internal pauses)"（https://partnerhelp.netflixstudios.com/hc/en-us/articles/11220999686163-Dubbing-Creative-Guidelines-Nonfiction-Unscripted-English ）。

### 根因 4：无能量/响度匹配——"赶"的一部分是能量密度

[仓库事实] 全链路无响度处理：tts 段原始振幅直接相加（`pipeline/stages/tts.py` 第 203–205 行），dub 直接 AAC（`pipeline/stages/dub.py` `_mux`）。F5 内部唯一的响度逻辑是 `target_rms=0.1`：仅当参考音频 RMS 低于 0.1 时先放大参考、生成后再按同因子回乘（本机 1.1.22 `utils_infer.py` `infer_batch_process` 第 463–465、526–527 行）——它不是输出响度归一，各段输出响度取决于模型采样。

[外部结论] "能量密度贡献赶稿感"有间接文献支撑：跨语言韵律迁移（含能量）机器配音工作把能量/音高随源语音迁移后，主观自然度差距向人类表现收窄 11.2%（arXiv:2306.11658，https://arxiv.org/abs/2306.11658 ）。但"逐槽响度对齐能降低赶稿感"这一具体因果在公开文献中没有直接对照实验——标注为**机制推断，未验证**。

---

## ③ 解决方案

按机制分五组。集成点的配置键命名均参照 `config.example.yaml` 现有风格（小写点分路径、嵌套 map）。

### A. 翻译阶段字符预算（dubbing-aware MT / isometric MT 落到 prompt）

- **机制**：在 translate 阶段为每段计算 `budget_chars = (end - start) × chars_per_second`（自然语速锚点，中文建议 4–5 字/秒，见下），把预算随请求发给 LLM 并在输出质量门校验；超预算段要求复译或按预算再压缩。译文变短后，speed 可从 2.0 回落、atempo 触发率下降，三条根因（1/2/3）同时减压。
- **Primary 来源**：
  - Isometric MT（±10% 字符内译文，自学习长度控制，ICASSP 2022）：https://arxiv.org/abs/2112.08682
  - Isochrony-aware MT（以源语音段时长为条件信号做翻译，Interspeech 2022）：https://arxiv.org/abs/2112.08548
  - VideoDubber（按 token 语音时长而非字数控制译文长度；含 En↔Zh 与真实电影测试集；En→Zh 上 SLC0.2 从基线 24.09% 提到 60.05%，BLEU 不降反升 14.12→14.32；并指出**中文字数控制与实际语音时长错位最严重**——中文一个词跨多字，字数预算要按"词/音节"折算）：https://arxiv.org/abs/2211.16934
  - **LLM prompt 长度控制**（与本管线 HY-MT2 LLM 路线直接同构：8 个开源 LLM、few-shot + 指令措辞对长度遵循度的研究；结论：指令措辞与示例性质必须一致，等长示例反而常被无视；采样多候选后按长度挑选可拿到 SOTA 的长度-质量折中）：https://arxiv.org/abs/2506.04855
  - 行业佐证：Netflix 配音准则"Translate and synchronize dialogue to match original audio timing"（https://partnerhelp.netflixstudios.com/hc/en-us/articles/11220999686163-Dubbing-Creative-Guidelines-Nonfiction-Unscripted-English ）；Dubbing in Practice 提醒**自然度优先于严格等长**（多数专业译配行超过 ±10% 字符变化），预算应是软约束 + 允许溢出兜底（https://arxiv.org/abs/2212.12137 ）。
  - 中文语速锚点（预算公式的 chars_per_second）：新闻播音 90 年代以来约 280 字/分 ≈ 4.7 字/秒（人民日报海外版，http://paper.people.com.cn/rmrbhwb/html/2016-06/17/content_1688313.htm ）；纪录片/专题片解说约 200–220 字/分 ≈ 3.3–3.7 字/秒（配音平台二手资料，https://yueyin.zhipianbang.com/news/detail-9756.html ，**非 primary，标注存疑**）；与 ADR-0005 仓库实测的"自然 4–6 字/秒"一致，建议默认取 4.5–5。
- **集成点**：`pipeline/stages/translate.py`——SYSTEM_FAITHFUL/SYSTEM_REFINE 请求体加 `budget` 字段（或专门的 dubbing 提示变体）；输出门加每段预算校验。新配置键：`translation.max_chars_per_second`（float，0=关闭，默认 0 保持现行为；配音跑法设 4.5–5.0）。`translation.json` schema 不变（target 只是变短），subtitle 75 字符契约与 88 字符门天然兼容。翻译缓存键由 system prompt + 请求体构成（`docs/agents/execution-chain.md`），改提示即自动失效旧缓存，无迁移问题。
- **零云合规**：完全合规——用的就是现有本地 HY-MT2 托管服务，零新依赖。
- **许可证**：论文方法不受许可约束；无新代码依赖。
- **RTF/成本**：0 额外算力（同一次 LLM 调用加字段）；若采用"多候选挑长度"变体，翻译调用 ×2–3，翻译阶段远非瓶颈（整体 8.29 RTFx 中 TTS/ASR 占大头），但注意 Q4 单副本 batch_segments=1 的吞吐限制。
- **风险**：HY-MT2-1.8B（Q4）对字符预算的遵循度未知——Javorský 论文显示小开源 LLM 的长度遵循是主要失败点（**未验证，见 ⑤**）；过度压缩导致漏译（VideoDubber 用 BLEU 监控，本管线现无译文回译校验，只能靠 88 字符门兜底下限、预算门兜底上限之间的窗口）；预算应作用于两轮（忠实+润色），否则润色轮可能把长度加回去。

### B. 停顿保留与 off-screen 松弛（prosodic alignment 思路）

- **机制**：两个正交子项。(1) **停顿保留**：配音跑法把 `segmentation.pause_s` 回调到 0.45 甚至更低，让自然停顿留在段间静音里（源有停顿→配音轨也有静音），溢出问题交给 A 的文本预算解决，而不是靠合并吞停顿。(2) **槽尾留白 / off-screen 松弛**：本产品是整轨替换的画外音式配音（ADR-0003），观感上属 Amazon 论文里的 off-screen 场景——该论文证明对 off-screen 句放松等时性（全局动态规划分配松弛量、允许语音越界进句间静音）显著提升主观评分（Wins 提升 107%–297%，p<0.01）。仓库现状：溢出进相邻静音本就允许且记账（ADR-0003），缺的是把它从"被动兜底"变成"主动分配"——例如槽尾预留 `min_tail_s`、短槽允许顺延。
- **Primary 来源**：Interspeech 2022 论文全文 https://www.isca-archive.org/interspeech_2022/virkar22_interspeech.pdf （含 2020 前作与 ICASSP 2021 改进版的机制描述：短语切分+停顿插入+语速罚分+时间边界松弛）；Netflix 准则（internal pauses 对齐）https://partnerhelp.netflixstudios.com/hc/en-us/articles/11220999686163-Dubbing-Creative-Guidelines-Nonfiction-Unscripted-English 。
- **集成点**：(1) 只改 `config.yaml` 的 `segmentation.pause_s`（已有键），无需代码——但必须与 A 同步上线，否则溢出门回炸（ADR-0005 记录 0.45 下 1102 段、speed 2.0 才 2.13%）。(2) 若做主动松弛：`pipeline/stages/tts.py` 的放置逻辑 + 新键 `tts.min_tail_s`（float，默认 0）与溢出门交互方式需按 ADR-0003 重新定义（建议先不动 2% 门）。
- **零云合规**：合规，纯本地逻辑。
- **许可证**：无新增。
- **RTF/成本**：(1) 为零；(2) 段数增多会略增 TTS 段数与 atempo 次数，量级 <10%。
- **风险**：pause_s 回调后短槽（<1s）的每段固定开销问题回到台面（ADR-0005：97% 残余溢出在短槽），需与 D-1 的 fix_duration 配套；主动松弛会动 `dub_plan.json` 的 `placed_start = slot_start` 固定契约（`docs/agents/execution-chain.md`），波及 validator/bench，是一次真正的契约变更，应走 ADR。

### C. 能量/响度匹配（ffmpeg 三件套 + 每槽 RMS 对齐）

- **机制**：三层递进。(1) **整轨两遍 loudnorm**（EBU R128）：第一遍 `print_format=json` 测 `measured_I/LRA/TP/thresh`，第二遍带 measured_values 做线性归一到目标 LUFS——消除"整条配音忽轻忽响"。(2) **每槽响度向源语音对齐**：对每个槽测源音频（`media/audio.wav`）与合成段的积分响度/RMS，逐段乘 gain 对齐——配音的"音量存在感"与源一致，避免整轨恒定高能量密度的疲劳感。(3) **dynaudnorm/acompressor**：动态归一/压缩只作补救（见风险）。源能量包络逐帧调制（envelope matching）属研究级（arXiv:2306.11658 的韵律迁移含能量成分，https://arxiv.org/abs/2306.11658 ），不建议首期做。
- **Primary 来源**：
  - FFmpeg 官方 loudnorm 文档（EBU R128；I 默认 -24、LRA 7、TP -2、linear 默认 true、print_format json/summary）：https://ffmpeg.org/ffmpeg-filters.html#loudnorm ；本机 ffmpeg 4.2.7 `ffmpeg -h filter=loudnorm` 输出与之一致（参数面已逐项核对）。
  - 两遍线性法官方推荐：FFmpeg wiki "it is recommended to run the normalization with two passes"（https://trac.ffmpeg.org/wiki/AudioVolume ）。
  - EBU R128 目标 -23 LUFS / 真峰 -1 dBTP：https://tech.ebu.ch/docs/r/r128.pdf （广播目标；播客/流媒体惯例 -16 LUFS，二手共识，未从 primary 逐平台核实）。
  - dynaudnorm（帧 500ms/滤波 31 帧/峰值 0.95/最大增益 ×10 等默认值）与 acompressor（threshold 0.125、ratio 2、attack 20ms、release 250ms、detection rms 等默认值）参数面来自本机 `ffmpeg -h filter=dynaudnorm`、`ffmpeg -h filter=acompressor`（= 官方 4.2.7 二进制自述）。
- **集成点**：(1) 整轨 loudnorm 放 `pipeline/stages/dub.py` 的 `_mux`（AAC 编码前 `-af`），新键 `dub.loudnorm: {enabled, i_lufs, tp_dbtp, lra}`——注意 loudnorm 上采样到 192kHz 内部处理，24kHz mono 输入需验证 AAC 96k 输出无异常（**未验证**）；或放 tts 阶段写 `tts/tts.wav` 前。(2) 每槽对齐放 `pipeline/stages/tts.py` 的放置循环（槽区间从 `media/audio.wav` 读 RMS，对 samples 乘 gain，plan 里记 `gain_db`），新键 `tts.slot_gain_match: true` + `tts.slot_gain_target_offset_db`。
- **零云合规**：合规，ffmpeg 已是必需依赖，soundfile/numpy 已在用。
- **许可证**：FFmpeg LGPL/GPL（系统包已存在，无新增）。
- **RTF/成本**：loudnorm 两遍 ≈ 2× 实时以下的 CPU，对 8.29 RTFx 的管线可忽略；每槽 RMS 对齐是逐段 numpy 运算，近零。
- **风险**：loudnorm 的动态模式（linear 不可行时）会引入时变增益"泵感"，配音场景应坚持 linear+两遍；每槽对齐在源槽极安静（耳语/远处人声）时会错误拉高合成音或放大底噪——需夹紧 gain 范围（如 ±6dB）；响度归一解决的是"存在感一致性"，对根因 1/2 的语速赶稿**无直接作用**，预期收益独立且有限（对应根因 4）。

### D. F5-TTS 原生参数用满（不动引擎）

调研要点先给结论：**官方 infer API 没有 `n_best`，也没有 `target_r`**——代码搜索与 main 分支源码逐文件核对均为零命中（https://github.com/SWivid/F5-TTS 搜索 `n_best`/`target_r`；main 分支 `utils_infer.py`、`api.py`、`infer_gradio.py` 全文核对）。"n_best 多候选按目标时长挑选"**不是官方功能**；best-of-N 选时长只出现在第三方 wrapper / 测试时扩展研究（如 2026 年 flow-TTS 测试时扩展论文，未取到稳定 URL，**未验证**）。`target_r` 的真名是 `target_rms`（响度参数，见根因 4）。官方参数实测语义（本机 1.1.22 包源码 + main 分支，二者公式一致）：

| 参数 | 默认 | 语义 |
|---|---|---|
| `speed` | 1.0 | 时长估计除数（非后处理变速；范围即本管线 0.5–2.0） |
| `fix_duration` | None | 强制总时长（**含参考段**；`duration = int(fix_dur*sr/hop)`，生成后裁掉 ref；多 chunk 时按 UTF-8 字节权重分摊——1.1.22 与 main 均已含分摊修复，issue #1308 的 N 倍膨胀已修） |
| `nfe_step` | 32 | 去噪步数（质量/速度，与时长无关） |
| `cfg_strength` | 2.0 | classifier-free guidance 强度（与时长无关） |
| `sway_sampling_coef` | -1 | flow 步采样策略系数（与时长无关） |
| `target_rms` | 0.1 | 参考/输出响度下限归一（见根因 4） |
| `cross_fade_duration` | 0.15 | chunk 间交叉淡化 |

来源：本机 `.venv/.../f5_tts/api.py` 与 `infer/utils_infer.py`（一手，官方发布包）；main 分支 https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/utils_infer.py ；issue #1308（fix_duration 分摊修复的来历）https://github.com/SWivid/F5-TTS/issues/1308 。

#### D-1 短槽 fix_duration 修补

- **机制**：对 <1s 槽（ADR-0005：97% 残余溢出所在），显式传 `fix_duration = ref_sec + slot_duration - tail_margin`，绕开两个坑：(a) 每段 ~0.5–1s 的合成下限来自时长估计公式与参考段拼接开销，fix_duration 直接指定 mel 帧数；(b) 1.1.22 对 <10 UTF-8 字节文本（中文 ≤3 字）**静默把 speed 覆盖为 0.3**（`local_speed = 0.3`，本机 1.1.22 源码第 490–492 行；main 已改为 `speed * 0.3`，issue #1331，https://github.com/SWivid/F5-TTS/issues/1331 ），短中文叹词因此被拉长——fix_duration 可完全绕开该分支。
- **Primary 来源**：官方源码（上表）；issue #1331（含官方推荐的 fix_duration 替代算式）；issue #1155（维护者确认短文本时长是"core part"问题并指向时长预测研究 arXiv:2507.14988 DMOSpeech 2，https://github.com/SWivid/F5-TTS/issues/1155 、https://arxiv.org/abs/2507.14988 ）。
- **集成点**：`pipeline/tts/engine.py` 的 `synth()` 增加 `fix_duration` 透传；`pipeline/stages/tts.py` 对 `slot_duration < 阈值` 的段计算并传入；`ref_sec` 可从参考片段实测（`pipeline/tts/voices.py` 的音频路径）。新键：`tts.short_slot_fix_duration: true`（或内部阈值常量）。注意 fix_duration 进合成缓存键（现在键=音色+文本+speed，ADR-0004/0005 惯例：拟合后于合成、合成恒按原生速度——fix_duration 属合成参数，必须入键）。
- **零云合规/许可证/RTF**：合规；无新增；0 额外算力。
- **风险**：#1343 已示警——把时长强压到模型自然值以下会有吞音/压缩伪声风险，短槽文本本就短（3–4 字），风险可控但**未实测**；ADR-0005 明言"F5 infer speed 参数语义若在升级中改变"是重开条件——1.1.22 与 main 的 `local_speed` 行为已分叉（0.3 vs speed×0.3），升级包会改变短段行为，需钉版本。

#### D-2 参考音色库重录（快语速参考）

- **机制**：F5 时长估计 = 参考段"字节/秒"外推（上表公式），参考语速直接决定基础合成语速（ADR-0005 实测 zh-female-01 2.2 字/秒 → 合成 2.4–2.6 字/秒；zh-male-01 4.7 → 5.2–5.5）。换成 ~5 字/秒的从容中文参考后，speed 可回到 1.0–1.2 的自然区间。
- **Primary 来源**：时长公式（官方源码，上表）；官方 ref 选择建议：参考音频 <12s（超长自动剪到 ~12s）、结尾留 ~1s 静音、参考文本须与音频严格一致（官方 infer README，https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md ）。**官方文档未给出"参考语速应匹配目标语速"的明文建议**——该结论由公式与 ADR-0005 实测推出。
- **Cross-lingual 备注**（任务清单项）：英文 ref → 中文输出时，时长估计按 ref 的"字节/秒"外推——英文参考的慢节奏会同样传染中文输出；且官方 issue 确认风格（含口音、语速习惯）随参考迁移，跨语言克隆口音明显（维护者："the model uses the same style of speech as used in the reference audio"；Emilia 训练集 code-switch 样本口音偏软，https://github.com/SWivid/F5-TTS/issues/315 ）。本管线 zh 配音用 zh 参考，不受影响；若未来做 en 输出需配 en 参考。
- **集成点**：纯资产变更（`_model_cache/tts_voices/` 重录 + `voices.json`），代码零改动；ADR-0005 已把它列为"right long-term fix"，是维护者决策（NC 许可资产不进 git）。
- **零云合规/许可证**：资产须自备（参考 ADR-0004 的 NC 音频不入库约定）；模型权重仍 CC-BY-NC-4.0（ModelScope SWivid/F5-TTS_Emilia-ZH-EN，ADR-0004）。
- **RTF/成本**：一次性录音成本；运行期 0。
- **风险**：参考越快，中文短段合成越容易触发 <10 字节的 0.3 慢速分支以外的失稳（#1155 描述短文本本就不稳）；与 A 组合后 speed 回落，收益最大。

### E. 备选本地引擎的时长/语速控制（仅"能否直接解决 P1"视角，不做 ADR-0004 式重评）

| 引擎 | 原生时长/语速控制（primary 核实） | 许可证 | RTF | 对 P1 的直接价值 |
|---|---|---|---|---|
| Qwen3-TTS（12Hz-1.7B） | **无显式 speed/duration 参数**；风格走自然语言 `instruct`（如"用愤怒的语气说"），未文档化语速指令（README，https://github.com/QwenLM/Qwen3-TTS ） | Apache-2.0 | ADR-0004 记录原生 RTF 2.0（除非 FlashAttention2 顺利装上） | 低：无时长旋钮 + RTF 超 0.5 门 |
| IndexTTS-2 / 2.5 | 2.5 有 `duration_factor`（0.5–2.0，>1 放慢 <1 加快——相对速率，**非绝对目标时长**）；2 号称"首个具备精确合成时长控制的自回归 TTS"但 README 明言"not yet enabled in this release"（https://github.com/index-tts/index-tts ） | bilibili 自定义许可（商用需联系） | 官方 README：2.5 RTF 0.2065（4090, bf16）；ADR-0004 记 0.21 | 中：duration_factor ≈ 另一个 speed 旋钮，仍属根因 2 的"硬压"路线；真正的绝对时长控制未开源 |
| CosyVoice2/3 | `speed=1.0` 参数存在，但实现是 **mel 谱事后线性插值**（`F.interpolate`），仅非流式模式；instruct 模型支持自然语言语速/音量指令（README"languages, dialects, emotions, speed, volume"；源码 https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/model.py ；论文 https://arxiv.org/abs/2506.02489 ） | 代码 Apache-2.0（模型许可另计） | ADR-0004 记 ~0.70（超 0.5 门） | 低：speed 本质等同 atempo，且 RTF 不达标 |
| XTTS-v2 | `speed` 在 model.inference 层，实现为 **GPT latent 线性插值**（`length_scale=1/max(speed,0.05)`，事后缩放；https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/models/xtts.py ） | 代码 MPL-2.0；**权重 CPML（非商用，https://huggingface.co/coqui/XTTS-v2 ）**；上游仓库已归档 | 未官方公布 | 低：speed 等同后处理变速；权重许可与归档状态都比现状差 |

结论：**没有任何一个候选提供"绝对目标时长"的原生开源控制**（IndexTTS-2 有但未启用）；CosyVoice/XTTS 的 speed 本质与 atempo 同类，换引擎不解决根因 2，且各有 RTF/许可/维护硬伤。E 整体不推荐作为 P1 的解法。

---

## ④ 建议优先级（按性价比）

1. **A 翻译字符预算**（`translation.max_chars_per_second`）——唯一直击根因 1+2 的手段；零额外算力、零新依赖、schema 不变、缓存自动失效；primary 文献与行业实践最厚（isometric MT → isochrony-aware → VideoDubber → LLM 长度控制一路成熟）。与 Dubbing in Practice 的"自然度优先"结论对齐：预算做软约束（超预算重试/警告），不拿 88 字符门式的硬失败换长度。
2. **D-2 参考音色库重录（~5 字/秒中文参考）**——ADR-0005 已认定的 long-term fix；A 落地后 speed 可回 1.0–1.2，"赶稿感"的最大倍率来源（2.0×）随之消失。成本是一次性资产决策。
3. **C-1 整轨两遍 loudnorm**（`dub.loudnorm.*`）——一行滤镜的工程量，独立改善能量一致性；目标值建议 -16 LUFS（流媒体惯例）或对齐源轨实测 I 值（更贴"向源对齐"思想）。
4. **B-1 pause_s 回调 0.45 + A 联动**——恢复留白只需改配置，但必须在 A 生效后做，否则溢出门回炸；ADR-0005 的 0.33% 成绩是在吞停顿的前提下拿到的，需要重新验收。
5. **C-2 每槽 RMS 对齐**（`tts.slot_gain_match`）——中等工程量，收益限于根因 4，且需 gain 夹紧防底噪放大。
6. **D-1 短槽 fix_duration**——解决 0.33% 残余溢出的最后一块（<1s 槽），对整体听感影响小；注意缓存键与包版本分叉。
7. **B-2 主动松弛/off-screen 分配** 与 **E 换引擎**——前者要动 dub_plan 固定契约（应走新 ADR）；后者经核实不提供所缺能力，均不建议现在做。

验收注意（仓库约束）：以上任何一项上线后都需重跑 1h 验收内容（issue #5 语料）核 2% 溢出门、RTFx ≤ 0.5、1h 无漂移，并补一个人耳听感门记录（ADR-0005 已要求人耳门持续判定"是否仍然赶"）。

---

## ⑤ 未验证 / 存疑项清单

1. **HY-MT2-1.8B Q4 对字符预算提示的遵循度**——未测试。Javorský 等（arXiv:2506.04855）显示小开源 LLM 长度遵循是主要失败模式（等长示例常被无视），1.8B Q4 属于该风险区。需先做离线探针再定默认值。
2. **EN→ZH 纯文本膨胀率**——仓库 65.44% 溢出是"文本膨胀 × 参考语速"联合值，未单独测量文本分量；外部亦无 En→Zh 字符/秒 膨胀率的 primary 数据（VideoDubber 给的是控制后指标）。VideoDubber 明示中文"字数 ≠ 时长"（词跨多字），预算按字算存在系统性偏差，更准的锚是音节数（中文≈字数）。
3. **中文语速基准的纪录片 200–220 字/分**——来自配音接单平台（zhipianbang 等），非 primary；新闻 280 字/分有人民日报海外版背书（http://paper.people.com.cn/rmrbhwb/html/2016-06/17/content_1688313.htm ）。
4. **Coupé 等 2019 的中文 ~5.18 音节/秒、英文 ~6.3 音节/秒**——来自搜索结果二手转述，primary 页面（https://www.science.org/doi/10.1126/sciadv.aaw2594 ）只核实了总均值 6.63 音节/秒与 ~39.15 bits/秒收敛结论，分语言数值**未验证**。
5. **`n_best`**——官方代码确认不存在（代码搜索 + main 分支逐文件核对）；其"多候选按目标时长选择"的传闻出处为第三方 wrapper / 测试时扩展研究，效果**未验证**。
6. **fix_duration 短槽实测效果与伪声风险**——未运行任何合成实验；#1343 的吞音报告提示强压时长的伪声风险。
7. **f5-tts 1.1.22 与 main 的短文本行为分叉**——1.1.22 `local_speed=0.3`（完全覆盖 speed），main `speed*0.3`（相对放缓）：升级会静默改变短段时长行为，ADR-0005 的重开条件之一即此类语义变化；建议钉死 1.1.22 并记录该差异。
8. **loudnorm 在本链路（24kHz mono → AAC 96k）的听感**——未验证；loudnorm 内部升采样至 192kHz，输出重采样回 24kHz 后再 AAC 的质量未测。
9. **每槽 RMS 对齐的收益**——"能量密度贡献赶稿感"只有间接文献（韵律迁移含能量的整体收益 11.2%），无逐槽响度对齐的对照实验，属机制推断。
10. **IndexTTS-2 绝对时长控制**——README 明言未在开源版启用；2.5 `duration_factor` 实测效果未验证。
11. **Qwen3-TTS 语速类 instruct 指令**——README 只演示情绪/方言指令，未文档化"说慢点"类指令的可靠性，未验证。
12. **用户"约 20 字/秒"的赶稿感**——主观感受，仓库无客观测量；客观上界为 speed 2.0（≈10 字/秒）× atempo 1.25 ≈ 12.5 字/秒，任何"20 字/秒"的说法无法从产物指标证实。
13. **纯云 API 配音服务**（ElevenLabs dubbing 等）——不符零云约束，按任务要求一笔带过，不纳入方案。

---

## 附：本文引用的仓库内文件

- `docs/adr/0003-dub-replaces-full-audio-track.md`、`docs/adr/0004-f5-tts-with-named-voice-library.md`、`docs/adr/0005-tts-synthesis-speed-knob.md`
- `docs/agents/execution-chain.md`
- `pipeline/stages/tts.py`、`pipeline/stages/dub.py`、`pipeline/stages/translate.py`、`pipeline/stages/segment.py`
- `pipeline/tts/engine.py`、`pipeline/tts/fitting.py`、`pipeline/tts/voices.py`
- `config.example.yaml`
- `.venv/lib/python3.10/site-packages/f5_tts/api.py`、`.venv/lib/python3.10/site-packages/f5_tts/infer/utils_infer.py`（f5-tts 1.1.22 官方发布包源码，本机一手核对）
