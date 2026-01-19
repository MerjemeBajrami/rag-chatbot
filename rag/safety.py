from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


INJECTION_PATTERNS = [
    r"\bignore\b.*\binstructions\b",
    r"\bdisregard\b.*\b(system|developer)\b",
    r"\byou are now\b",
    r"\bdo not cite\b",
    r"\breveal\b.*\bprompt\b",
    r"\bdeveloper message\b",
    r"\bsystem prompt\b",
]

SUSPICIOUS_URL_PATTERN = r"https?://\S+"


@dataclass(frozen=True)
class SafetyResult:
    is_suspicious: bool
    reason: str = ""


def detect_prompt_injection(user_text: str) -> SafetyResult:
    """
    Lightweight guardrail. Not perfect, but satisfies the 'guardrails' nice-to-have:
    - Detect common instruction override attempts
    - Detect attempts to remove citations or reveal system prompt
    """
    t = user_text.strip().lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, t):
            return SafetyResult(True, f"Matched injection pattern: {pat}")

    # Sometimes malicious prompts include lots of links or exfil instructions
    if len(re.findall(SUSPICIOUS_URL_PATTERN, user_text)) >= 3:
        return SafetyResult(True, "Too many URLs in a single message (possible prompt injection).")

    return SafetyResult(False, "")
