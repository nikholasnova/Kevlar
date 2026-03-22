from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from kevlar.preprocessing.patterns import SYSTEM_REMINDER_PATTERN, VOLATILE_PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class NormalizedPrompt:
    system: str
    messages: list[dict]
    stable_hash: str
    volatile_sections: list[str] = field(default_factory=list)


def _extract_volatile(text: str) -> tuple[str, list[str]]:
    """Pull volatile content out of text, return (cleaned_text, extracted_sections)."""
    extracted = []
    cleaned = text
    for pattern in VOLATILE_PATTERNS:
        matches = pattern.findall(cleaned)
        extracted.extend(matches)
        cleaned = pattern.sub("", cleaned)
    # collapse multiple blank lines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, extracted


def _build_volatile_block(sections: list[str]) -> str:
    if not sections:
        return ""
    joined = "\n\n".join(s.strip() for s in sections if s.strip())
    if not joined:
        return ""
    return f"\n\n[Dynamic Context]\n{joined}"


def normalize(
    system: str,
    messages: list[dict],
    enabled: bool = True,
) -> NormalizedPrompt:
    if not enabled:
        h = hashlib.sha256(system.encode()).hexdigest()[:16]
        return NormalizedPrompt(system=system, messages=messages, stable_hash=h)

    stable_system, sys_volatile = _extract_volatile(system)
    all_volatile = list(sys_volatile)

    normalized_messages = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            clean, vol = _extract_volatile(content)
            all_volatile.extend(vol)
            normalized_messages.append({**msg, "content": clean})
        elif isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    clean, vol = _extract_volatile(block["text"])
                    all_volatile.extend(vol)
                    new_blocks.append({**block, "text": clean})
                else:
                    new_blocks.append(block)
            normalized_messages.append({**msg, "content": new_blocks})
        else:
            normalized_messages.append(msg)

    volatile_block = _build_volatile_block(all_volatile)
    final_system = stable_system + volatile_block

    stable_hash = hashlib.sha256(stable_system.encode()).hexdigest()[:16]

    if all_volatile:
        logger.debug(
            "Normalized prompt: extracted %d volatile sections, stable_hash=%s",
            len(all_volatile),
            stable_hash,
        )

    return NormalizedPrompt(
        system=final_system,
        messages=normalized_messages,
        stable_hash=stable_hash,
        volatile_sections=all_volatile,
    )
