import re


FILLER_ONLY = frozenset({
    "啊", "啊啊", "哎", "唉", "呃", "呃呃", "额", "嗯", "嗯嗯", "唔", "哦", "哦哦", "噢",
    "erm", "hmm", "mhm", "mm", "uh", "um",
})
FILLER_PUNCTUATION = " \t\r\n.,!?;:'\"，。！？；：、…~～-_"
REPEATED_CHAR_RE = re.compile(r"([A-Za-z\u4e00-\u9fff])\1{3,}")


def has_pathological_repetition(text):
    compact = re.sub(r"\s+", "", str(text))
    if REPEATED_CHAR_RE.search(compact):
        return True
    for size in range(2, 9):
        repeats = size * 3
        for start in range(0, len(compact) - repeats + 1):
            unit = compact[start:start + size]
            if compact[start + size:start + repeats] == unit * 2:
                return True
    return False


def is_low_information(text):
    normalized = str(text).strip().lower().strip(FILLER_PUNCTUATION)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized in FILLER_ONLY or has_pathological_repetition(text)


UNIT_REPEAT_RE = re.compile(r"(.{1,8}?)(\1{2,})")


def collapse_pathological_repetition(text):
    """Collapse unit-x3+ runs (per has_pathological_repetition) to one unit.

    Companion rescue for is_low_information: a spoken stutter translated
    faithfully ("更更更加" -> "更加更加更加可靠") is pathological in writing;
    collapsing the run keeps the meaning. Best-effort — callers must
    re-validate the result.
    """
    value = str(text)
    for _ in range(3):
        collapsed = UNIT_REPEAT_RE.sub(r"\1", value)
        if collapsed == value:
            break
        value = collapsed
    return value


def has_translation_meta(text):
    value = str(text).strip()
    lowered = value.lower()
    if any(marker in value for marker in (
        "自然流畅", "目标语言", "副标题", "将原文", "优化后的翻译", "完成了翻译",
    )):
        return True
    if "润色" in value and any(marker in lowered for marker in (
        "原文", "表达", "翻译", "subtitle", "标题",
    )):
        return True
    return "简体中文" in value and any(marker in value for marker in (
        "表达", "版本", "翻译",
    ))


# The refine prompt's two-string example leaks back as translated variants
# ("优化后的第一句翻译。", "Refined first sentence.", "refin 了 第一句 话。"),
# as whole-prompt echoes ("refine the subtitle translation to sound natural
# and fluent in the target language.") and as self-narration ("因此翻译为：").
# Observed on the Q4 runtime.
EXAMPLE_LEAK_RE = re.compile(
    r"优化后|精炼|翻译为|因此翻译|refin(e|ed)?\s*[了后：:]"
    r"|first sentence|first subtitle|second sentence|second subtitle"
    r"|second translation|natural and fluent|target language"
    r"|subtitle translation|refine the",
    re.IGNORECASE,
)


def has_example_leak(text):
    """Refine-pass variant of has_translation_meta for example echoes.

    Rejected refine responses only fall back to the faithful draft, so the
    broader substring set is safe here: a false positive costs polish, not
    the run.
    """
    return bool(EXAMPLE_LEAK_RE.search(str(text)))
