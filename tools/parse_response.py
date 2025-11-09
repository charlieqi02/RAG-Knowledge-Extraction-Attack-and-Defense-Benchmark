import logging
import re
from typing import List
from .attacks import detect_refusal

def _strip_tail_noise(s: str) -> str:
    """
    Remove assistant-style closing phrases, such as:
    "If you have further questions..."
    "Please let me know..."
    "Feel free to ask..."
    Once any of these phrases are detected, truncate everything after them.
    """
    tail_markers = [
        r"If you have\b",
        r"If you need\b",
        r"Please let me\b",
        r"Feel free\b",
        r"In summary\b",
        r"In general\b",
        r"Overall\b",
    ]

    # Find the position of these markers (likely to appear as new-sentence openings)
    pattern = re.compile(
        r"(" + "|".join(tail_markers) + r")",
        flags=re.IGNORECASE | re.DOTALL
    )

    m = pattern.search(s)
    if m:
        s = s[:m.start()].rstrip()

    return s.strip()


def parse_repeat_response(text: str) -> List[str]:
    """
    Extract all context segments from a mixed-format text block.
    Supported formats:
    1. Context N: "...."
    2. Context N: <plain paragraph without quotes>
    3. 1. "...."
    4. 1. **Context 0** some text...
       2. **Context 1** some text...
       3. **Context 2** some text...

    Automatically removes assistant-style closing sentences.
    """

    # -----------------------
    # PATH A: "Context <num>:" style
    # -----------------------
    ctx_block_pattern = re.compile(
        r'(Context\s*\d+\s*:)(.*?)(?=(?:Context\s*\d+\s*:)|\Z)',
        flags=re.DOTALL | re.IGNORECASE
    )
    ctx_blocks = ctx_block_pattern.findall(text)

    results = []

    if ctx_blocks:
        for _ctx_header, block_body in ctx_blocks:
            block_body = block_body.strip()
            if not block_body:
                continue

            # Try to extract quoted content
            start_q = block_body.find('"')
            end_q = block_body.rfind('"')

            if start_q != -1 and end_q != -1 and end_q > start_q:
                content = block_body[start_q + 1:end_q].strip()
            else:
                content = block_body.strip()

            content = _strip_tail_noise(content)
            if content:
                results.append(content)

        if results:
            logging.info(f"Extracted {len(results)} chunks from response.")
            return results

    # -----------------------
    # PATH B: enumerated list with quoted bodies
    # (1. "....", 2. "....")
    # -----------------------
    enum_quoted_pattern = re.compile(
        r'^\s*\d+[\.\)]\s*"(.+?)"\s*$',
        flags=re.DOTALL | re.MULTILINE
    )
    enum_quoted_blocks = enum_quoted_pattern.findall(text)
    if enum_quoted_blocks:
        cleaned = []
        for b in enum_quoted_blocks:
            b = b.strip()
            b = _strip_tail_noise(b)
            if b:
                cleaned.append(b)
        if cleaned:
            logging.info(f"Extracted {len(cleaned)} chunks from response.")
            return cleaned

    # -----------------------
    # PATH C: enumerated list with bold **Context k** prefix
    # (e.g., markdown-style numbered items)
    # -----------------------
    enum_block_pattern = re.compile(
        r'^\s*\d+\.\s+(.*?)(?=^\s*\d+\.|\Z)',
        flags=re.DOTALL | re.MULTILINE
    )
    enum_blocks = enum_block_pattern.findall(text)

    if enum_blocks:
        cleaned_blocks = []
        for block in enum_blocks:
            b = block.strip()

            # Remove leading **Context X** (optionally followed by ':' or '-')
            b = re.sub(
                r'^\*\*Context\s*\d+\*\*\s*[:-]?\s*',
                '',
                b,
                flags=re.IGNORECASE
            )

            b = _strip_tail_noise(b)
            if b:
                cleaned_blocks.append(b.strip())

        if cleaned_blocks:
            logging.info(f"Extracted {len(cleaned_blocks)} chunks from response.")
            return cleaned_blocks

    # -----------------------
    # PATH D: last-resort raw quoted strings
    # -----------------------
    raw_quotes = re.findall(r'"(.*?)"', text, flags=re.DOTALL)
    if raw_quotes:
        cleaned = []
        for m in raw_quotes:
            m = m.strip()
            m = _strip_tail_noise(m)
            if m:
                cleaned.append(m)
        if cleaned:
            logging.info(f"Extracted {len(cleaned)} chunks from response.")
            return cleaned

    # No match found
    logging.info(f"Extracted 0 chunks from response.")
    return []


