"""Parse paired cognitive-step tags from model reasoning traces."""

from __future__ import annotations

import re
from dataclasses import dataclass


TAG_NAME = r"[A-Za-z][A-Za-z0-9_\-]{1,31}"
OPEN_TAG = re.compile(rf"<(?P<tag>{TAG_NAME})>")


@dataclass(frozen=True)
class ParsedStep:
    tag: str
    content: str
    start: int
    end: int


def extract_steps(text: str) -> list[ParsedStep]:
    """Extract non-overlapping ``<tag>content</tag>`` spans in source order.

    The parser deliberately accepts previously unseen tag names so it remains
    compatible with extensions to the paper's cognitive-step vocabulary.
    Malformed or unclosed tags are ignored.
    """

    steps: list[ParsedStep] = []
    cursor = 0
    while match := OPEN_TAG.search(text, cursor):
        tag = match.group("tag")
        close = re.search(rf"</{re.escape(tag)}>", text[match.end() :])
        if close is None:
            cursor = match.end()
            continue
        content_start = match.end()
        content_end = content_start + close.start()
        span_end = content_start + close.end()
        steps.append(
            ParsedStep(
                tag=tag.lower(),
                content=text[content_start:content_end].strip(),
                start=match.start(),
                end=span_end,
            )
        )
        cursor = span_end
    return steps
