import re
from typing import List

def semantic_chunk_text(
    text: str,
    max_len: int = 750,
    overlap: int = 140,
    min_chunk_len: int = 100
) -> List[str]:
    """
    Sentence-aware chunking with overlap.
    - Cleans whitespace
    - Avoids tiny chunks
    - Adds bounded overlap for continuity
    """
    if not text:
        return []

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)

    base_chunks = []
    current = ""

    for s in sentences:
        if not s:
            continue

        candidate = f"{current} {s}".strip() if current else s
        if len(candidate) <= max_len:
            current = candidate
        else:
            if len(current) >= min_chunk_len:
                base_chunks.append(current.strip())
            elif base_chunks:
                base_chunks[-1] = f"{base_chunks[-1]} {current}".strip()
            current = s

    if current:
        if len(current) >= min_chunk_len:
            base_chunks.append(current.strip())
        elif base_chunks:
            base_chunks[-1] = f"{base_chunks[-1]} {current}".strip()

    if not base_chunks:
        return []

    safe_overlap = max(0, min(overlap, max_len // 2))

    final_chunks = []
    for i, chunk in enumerate(base_chunks):
        if i == 0 or safe_overlap == 0:
            final_chunks.append(chunk)
            continue

        prev_tail = base_chunks[i - 1][-safe_overlap:]
        merged = f"{prev_tail} {chunk}".strip()
        final_chunks.append(merged)

    return final_chunks