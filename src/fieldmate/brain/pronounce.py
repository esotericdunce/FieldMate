"""
Pronunciation Hardening & Text Normalization Engine for Rime TTS in FieldMate.

Ported and enhanced from Rime voice agent primitives:
- Converts technical acronyms (BSOD -> B S O D, WHEA -> W H E A, NVMe -> N V M e, etc.).
- Normalizes unit & math symbols (°C -> degrees Celsius, °F -> degrees Fahrenheit, % -> percent, etc.) so TTS never says "degree sign".
- Replaces em-dashes (—) with commas to prevent TTS stalls.
- Collapses repeated letters (e.g. "soooo" -> "so") and stacked punctuation ("!!" -> "!").
- Cleans markdown tags, code blocks, backticks, and stray asterisks.
"""

import re

_EM_DASH = re.compile(r"\s*—\s*")
_REPEAT = re.compile(r"([A-Za-z])\1{2,}")
_STACKED_PUNCT = re.compile(r"([!?])[!?]+")
_STRAY_ASTERISK = re.compile(r"(?<!\*)\*(?!\*)")
_MULTISPACE = re.compile(r"\s{2,}")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BACKTICKS = re.compile(r"`([^`]+)`")

# Symbol, unit & math normalizations for TTS
_SYMBOL_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # Temperature & Degrees
    (re.compile(r"(\d+)\s*°\s*C\b", re.IGNORECASE), r"\1 degrees Celsius"),
    (re.compile(r"(\d+)\s*°\s*F\b", re.IGNORECASE), r"\1 degrees Fahrenheit"),
    (re.compile(r"°\s*C\b", re.IGNORECASE), " degrees Celsius"),
    (re.compile(r"°\s*F\b", re.IGNORECASE), " degrees Fahrenheit"),
    (re.compile(r"°"), " degrees "),

    # Hardware & Performance units
    (re.compile(r"\b(\d+)\s*GB/s\b", re.IGNORECASE), r"\1 gigabytes per second"),
    (re.compile(r"\b(\d+)\s*MB/s\b", re.IGNORECASE), r"\1 megabytes per second"),
    (re.compile(r"\b(\d+)\s*GHz\b", re.IGNORECASE), r"\1 gigahertz"),
    (re.compile(r"\b(\d+)\s*MHz\b", re.IGNORECASE), r"\1 megahertz"),
    (re.compile(r"\b(\d+)\s*ms\b", re.IGNORECASE), r"\1 milliseconds"),
    (re.compile(r"\b(\d+)\s*RPM\b", re.IGNORECASE), r"\1 R. P. M."),
    (re.compile(r"(\d+)\s*%"), r"\1 percent"),

    # Math & Symbols
    (re.compile(r"~"), " approximately "),
    (re.compile(r"(?<=\d)\s*x\s*(?=\d)", re.IGNORECASE), " by "),  # 1920x1080 -> 1920 by 1080
    (re.compile(r"\s*=\s*"), " equals "),
    (re.compile(r"&"), " and "),
]

# Common technical PC troubleshooting pronunciations
_TECH_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bBSOD\b", re.IGNORECASE), "B. S. O. D."),
    (re.compile(r"\bWHEA\b", re.IGNORECASE), "W. H. E. A."),
    (re.compile(r"\bNVMe\b", re.IGNORECASE), "N. V. M. E."),
    (re.compile(r"\bSATA\b", re.IGNORECASE), "say-tah"),
    (re.compile(r"\bUEFI\b", re.IGNORECASE), "U. E. F. I."),
    (re.compile(r"\bBIOS\b", re.IGNORECASE), "bye-oss"),
    (re.compile(r"\bMDSCHED\b", re.IGNORECASE), "M. D. sched"),
    (re.compile(r"\bSFC\b", re.IGNORECASE), "S. F. C."),
    (re.compile(r"\bRAM\b", re.IGNORECASE), "RAM"),
    (re.compile(r"\bGPU\b", re.IGNORECASE), "G. P. U."),
    (re.compile(r"\bCPU\b", re.IGNORECASE), "C. P. U."),
    (re.compile(r"\bHDD\b", re.IGNORECASE), "H. D. D."),
    (re.compile(r"\bSSD\b", re.IGNORECASE), "S. S. D."),
]


def _collapse_repeat(m: re.Match) -> str:
    run, ch = m.group(0), m.group(1)
    return run[:3] if ch.lower() in ("m", "h") else ch


def tts_pronounce(text: str) -> str:
    """Transform raw LLM technical text into clean phonetic text for Rime TTS."""
    if not text:
        return ""

    # 1. Clean Markdown formatting
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _BACKTICKS.sub(r"\1", text)
    text = _STRAY_ASTERISK.sub("", text)
    text = text.replace("**", "")

    # 2. Em dashes -> commas
    text = _EM_DASH.sub(", ", text)

    # 3. Unit, degree & symbol normalizations
    for pattern, replacement in _SYMBOL_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # 4. PC Technical terms pronunciation hardening
    for pattern, replacement in _TECH_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # 5. Collapse elongated words ("sooooo" -> "so") & stacked punctuation
    text = _REPEAT.sub(_collapse_repeat, text)
    text = _STACKED_PUNCT.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text)

    return text
