"""Deterministic MGR mention-verifier policy (stage 1, preregistered).

This module never imports CHAIR vocabulary, COCO annotations, or any evaluator.
It only defines the frozen self-list prompt, output parsing and span alignment,
grounding-output parsing, region cropping, and the fixed evidence integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


SELFLIST_PROMPT = """The following text is a description of an image. List every concrete physical thing the text mentions as being in the image: objects, people, animals, vehicles, food, furniture, plants, and similar visible entities.
Rules:
- Copy each item exactly as it is written in the text (same words, same spelling), without articles such as "a", "an", or "the".
- One item per line, no numbering, no bullets, no explanations.
- Do not list abstract words (such as atmosphere, mood, scene, setting, design, detail), colors, materials, actions, or whole places such as room or street.
- Do not add, merge, or paraphrase items. Do not list anything the text does not mention.

Text:
{caption}"""

GROUNDING_PROMPT = "Point to the {phrase} in the image."

DETERMINERS = ("a", "an", "the", "this", "that", "these", "those")
MAX_ITEMS = 40
MAX_ITEM_WORDS = 8
CROP_MARGIN = 0.10
MIN_CROP_SIDE = 28  # Qwen2.5-VL smart_resize requires both sides >= 28 px
TRIGGER_THRESHOLD = 0.1
SELFLIST_MAX_NEW_TOKENS = 160
GROUNDING_MAX_NEW_TOKENS = 64

_LINE_PREFIX = re.compile(r"^\s*(?:[-*•]+|\(?\d+[.)]?)\s*")
_BOX_TAG = re.compile(
    r"<box>\s*\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]\s*</box>"
)
_BBOX_2D = re.compile(
    r"[\"']bbox_2d[\"']\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_POINT_ONLY = re.compile(r"point_2d|<points")


def format_selflist_prompt(caption: str) -> str:
    if not caption.strip():
        raise ValueError("self-list caption must be non-empty")
    return SELFLIST_PROMPT.format(caption=caption)


def strip_leading_determiner(item: str) -> str:
    words = item.split()
    while words and words[0].lower() in DETERMINERS:
        words = words[1:]
    return " ".join(words)


def parse_selflist_output(text: str) -> tuple[list[str], dict]:
    """Apply the frozen line rules; return kept items and drop counts."""

    stats = {"raw_lines": 0, "empty": 0, "too_long": 0, "beyond_max": 0, "kept": 0}
    items = []
    for raw_line in text.splitlines():
        stats["raw_lines"] += 1
        line = _LINE_PREFIX.sub("", raw_line).strip()
        line = line.rstrip(".").strip()
        line = strip_leading_determiner(line)
        if not line:
            stats["empty"] += 1
            continue
        if len(line.split()) > MAX_ITEM_WORDS:
            stats["too_long"] += 1
            continue
        if len(items) >= MAX_ITEMS:
            stats["beyond_max"] += 1
            continue
        items.append(line)
    stats["kept"] = len(items)
    return items, stats


@dataclass(frozen=True)
class Occurrence:
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class Mention:
    mention: str
    mention_key: str
    listed_item: str
    alignment: str
    occurrences: tuple[Occurrence, ...]


def _find_all(caption: str, phrase: str) -> list[Occurrence]:
    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    return [
        Occurrence(match.start(), match.end(), caption[match.start() : match.end()])
        for match in pattern.finditer(caption)
    ]


def _plural_variants(item: str) -> list[str]:
    words = item.split()
    if not words:
        return []
    last = words[-1]
    variants = [last + "s", last + "es"]
    if last.lower().endswith("s") and len(last) > 2:
        variants.append(last[:-1])
    return [" ".join(words[:-1] + [variant]) for variant in variants]


def align_items(caption: str, items: list[str]) -> tuple[list[Mention], list[str]]:
    """Whole-word case-insensitive alignment with last-word plural tolerance."""

    mentions: list[Mention] = []
    seen_keys: set[str] = set()
    unaligned: list[str] = []
    for item in items:
        occurrences = _find_all(caption, item)
        alignment = "exact"
        if not occurrences:
            for variant in _plural_variants(item):
                occurrences = _find_all(caption, variant)
                if occurrences:
                    alignment = "plural"
                    break
        if not occurrences:
            unaligned.append(item)
            continue
        key = occurrences[0].text.lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        mentions.append(
            Mention(
                mention=occurrences[0].text,
                mention_key=key,
                listed_item=item,
                alignment=alignment,
                occurrences=tuple(occurrences),
            )
        )
    return mentions, unaligned


def mention_to_dict(mention: Mention) -> dict:
    return asdict(mention)


def parse_grounding_box(raw_text: str) -> tuple[list[float] | None, str | None]:
    """Prereg v2 parser: first <box>[[…]]</box> or first "bbox_2d": [...]."""

    match = _BOX_TAG.search(raw_text) or _BBOX_2D.search(raw_text)
    if match:
        x1, y1, x2, y2 = (float(value) for value in match.groups())
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return None, "degenerate_box"
        return [x1, y1, x2, y2], None
    if _POINT_ONLY.search(raw_text):
        return None, "point_only"
    return None, "no_box_output"


def crop_box(box: list[float], width: int, height: int) -> tuple[list[int] | None, str | None]:
    """Expand each side by 10% of box width/height, clamp, reject degenerate crops."""

    x1, y1, x2, y2 = box
    margin_x = CROP_MARGIN * (x2 - x1)
    margin_y = CROP_MARGIN * (y2 - y1)
    left = max(0, int(round(x1 - margin_x)))
    top = max(0, int(round(y1 - margin_y)))
    right = min(width, int(round(x2 + margin_x)))
    bottom = min(height, int(round(y2 + margin_y)))
    if right - left < MIN_CROP_SIDE or bottom - top < MIN_CROP_SIDE:
        return None, "degenerate_box"
    return [left, top, right, bottom], None


def combine_scores(s_whole: float, s_region: float | None) -> float:
    """Two-witness absence: region evidence can only veto a whole-image trigger."""

    if s_region is None:
        return float(s_whole)
    return float(max(s_whole, s_region))


def is_triggered(score: float) -> bool:
    return score <= TRIGGER_THRESHOLD


def cascade_needs_region(s_whole: float) -> bool:
    """Method cost accounting: region queries are issued only when whole says absent."""

    return s_whole <= TRIGGER_THRESHOLD


def spacy_candidate_overlaps(candidate: dict, selflist_mentions: list[dict]) -> bool:
    """Candidate-level overlap: any occurrence touching any self-list span."""

    spans = [
        (int(occ["start_char"]), int(occ["end_char"]))
        for mention in selflist_mentions
        for occ in mention["occurrences"]
    ]
    for occurrence in candidate["occurrences"]:
        start = int(occurrence["start_char"])
        end = int(occurrence["end_char"])
        if any(start < span_end and span_start < end for span_start, span_end in spans):
            return True
    return False
