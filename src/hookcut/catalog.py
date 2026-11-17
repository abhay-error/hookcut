"""A curated list of open models that run locally through Ollama.

Every tag here was checked against the Ollama library. Sizes are download sizes
for the default quantisation. Cloud-only tags are excluded on purpose: hookcut
is local-first.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import asdict, dataclass

#: Working memory on top of the weights: a 16k-token context plus runtime.
OVERHEAD_GB = 1.5
#: Share of total RAM a model may use, leaving room for Whisper, video and the OS.
RAM_BUDGET = 0.75


@dataclass(frozen=True)
class CatalogModel:
    tag: str
    family: str
    maker: str
    size_gb: float
    notes: str
    #: Reasons step by step before answering (supports the "think" toggle).
    reasoning: bool = False
    recommended: bool = False

    @property
    def needs_gb(self) -> float:
        return round(self.size_gb + OVERHEAD_GB, 1)


CATALOG: list[CatalogModel] = [
    # Qwen (Alibaba)
    CatalogModel("qwen3:4b", "Qwen", "Alibaba", 2.5, "Fast, good for 8 GB machines", True),
    CatalogModel("qwen3:8b", "Qwen", "Alibaba", 5.2, "Balanced speed and quality", True),
    CatalogModel("qwen3:14b", "Qwen", "Alibaba", 9.3, "Best pick for 16 GB", True, True),
    CatalogModel("qwen3:30b", "Qwen", "Alibaba", 19, "Mixture-of-experts: big brain, quick", True),
    CatalogModel("qwen3.5:9b", "Qwen", "Alibaba", 6.6, "Newer generation, strong reasoning", True),
    CatalogModel("qwen3.5:27b", "Qwen", "Alibaba", 17, "Top quality for 24 GB+", True),
    CatalogModel("qwen3.6:35b-a3b", "Qwen", "Alibaba", 23, "Latest Qwen, mixture-of-experts", True),
    # Gemma (Google)
    CatalogModel("gemma4:e4b", "Gemma", "Google", 6.6, "Efficient, great writing"),
    CatalogModel("gemma4:12b", "Gemma", "Google", 7.7, "Sharp titles and hooks"),
    CatalogModel("gemma4:26b-a4b", "Gemma", "Google", 16, "Mixture-of-experts, fast for its size"),
    CatalogModel("gemma3:12b", "Gemma", "Google", 8.2, "Previous generation, very stable"),
    # Llama (Meta)
    CatalogModel("llama3.2:3b", "Llama", "Meta", 2.0, "Tiny and quick"),
    CatalogModel("llama3.1:8b", "Llama", "Meta", 4.9, "Reliable all-rounder"),
    # Nemotron (NVIDIA)
    CatalogModel("nemotron-3-nano:4b", "Nemotron", "NVIDIA", 2.8, "Small and fast", True),
    CatalogModel("nemotron-3.5-bolt:e4b", "Nemotron", "NVIDIA", 7.2, "Efficient reasoning", True),
    CatalogModel("nemotron-3-nano:30b", "Nemotron", "NVIDIA", 18,
                 "Mixture-of-experts, excellent judgement", True),
    CatalogModel("nemotron-3.5-lightning:30b", "Nemotron", "NVIDIA", 19, "Newest Nemotron", True),
    # GLM (Z.ai)
    CatalogModel("glm4:9b", "GLM", "Z.ai", 5.5, "Solid mid-size model"),
    CatalogModel("glm-4.7-flash:q4", "GLM", "Z.ai", 19, "Fast flagship-class model", True),
    # DeepSeek
    CatalogModel("deepseek-r1:8b", "DeepSeek", "DeepSeek", 5.2, "Reasoning model", True),
    CatalogModel("deepseek-r1:14b", "DeepSeek", "DeepSeek", 9.0, "Reasoning, 16 GB friendly", True),
    CatalogModel("deepseek-r1:32b", "DeepSeek", "DeepSeek", 20, "Strong reasoning", True),
    # Mistral
    CatalogModel("mistral-nemo:12b", "Mistral", "Mistral AI", 7.1, "Good instruction following"),
    CatalogModel("mistral-small3.2:24b", "Mistral", "Mistral AI", 15, "High quality, 24 GB+"),
    # Phi (Microsoft)
    CatalogModel("phi4-mini:3.8b", "Phi", "Microsoft", 2.5, "Compact and capable"),
    CatalogModel("phi4:14b", "Phi", "Microsoft", 9.1, "Strong at structured answers"),
    # gpt-oss (OpenAI open weights)
    CatalogModel("gpt-oss:20b", "gpt-oss", "OpenAI", 14, "Open-weight reasoning model", True),
    # Granite (IBM) / OLMo (AI2)
    CatalogModel("granite4.2:8b", "Granite", "IBM", 5.3, "Business-grade, concise"),
    CatalogModel("olmo-3.1:32b", "OLMo", "Ai2", 19, "Fully open (data + weights)", True),
]

BY_TAG = {m.tag: m for m in CATALOG}


def total_ram_gb() -> float:
    """Physical memory in GB, or 0 if it can't be determined."""
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return status.ullTotalPhys / 1024**3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return 0.0


def fit(model: CatalogModel, ram_gb: float) -> str:
    """'good', 'tight' (will swap / be slow), or 'too-big'. 'unknown' without RAM info."""
    if ram_gb <= 0:
        return "unknown"
    if model.needs_gb <= ram_gb * RAM_BUDGET:
        return "good"
    if model.needs_gb <= ram_gb * 0.95:
        return "tight"
    return "too-big"


def as_dicts(installed: set[str], ram_gb: float) -> list[dict]:
    return [
        {**asdict(m), "needs_gb": m.needs_gb, "installed": m.tag in installed,
         "fit": fit(m, ram_gb)}
        for m in CATALOG
    ]


def is_reasoning(tag: str) -> bool:
    model = BY_TAG.get(tag)
    if model:
        return model.reasoning
    return any(k in tag for k in ("qwen3", "deepseek-r1", "gpt-oss", "nemotron", "-think", "qwq"))
