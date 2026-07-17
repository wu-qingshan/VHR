"""Core deterministic policy for VERA-VHR local caption revision."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


REVISION_PROMPT = """Original sentence: {sentence}
The following phrases are not clearly supported by the image: {phrases}.
Rewrite only this sentence using details clearly visible in the image. Preserve every other supported object and fact, do not add new objects, and do not mention the unsupported phrases.
Answer with exactly one revised sentence and no explanation."""

ASCII_LETTER = re.compile(r"[A-Za-z]")


def valid_visual_candidate(candidate: dict) -> bool:
    """Reject parser artifacts such as Markdown '*' before visual verification."""

    return bool(
        ASCII_LETTER.search(candidate.get("root_lemma", ""))
        and ASCII_LETTER.search(candidate.get("phrase", ""))
    )


def sanitize_extraction(extraction: dict) -> dict:
    return {
        **extraction,
        "candidates": [
            candidate
            for candidate in extraction["candidates"]
            if valid_visual_candidate(candidate)
        ],
    }


def format_revision_prompt(sentence: str, phrases: Iterable[str]) -> str:
    values = [phrase.strip() for phrase in phrases if phrase.strip()]
    if not sentence.strip():
        raise ValueError("revision sentence must be non-empty")
    if not values:
        raise ValueError("revision requires at least one unsupported phrase")
    return REVISION_PROMPT.format(sentence=sentence, phrases="; ".join(values))


def exact_cycle(text: str, max_period=16, repeats=3):
    tokens = text.split()
    for end in range(repeats, len(tokens) + 1):
        max_candidate_period = min(max_period, end // repeats)
        for period in range(1, max_candidate_period + 1):
            block = tokens[end - period : end]
            if all(
                tokens[
                    end - (repeat_index + 1) * period : end - repeat_index * period
                ]
                == block
                for repeat_index in range(1, repeats)
            ):
                return {
                    "period": period,
                    "block": block,
                    "end_token_index": end,
                }
    return None


def levenshtein_words(left: str, right: str) -> int:
    left_words = left.split()
    right_words = right.split()
    previous = list(range(len(right_words) + 1))
    for left_index, left_word in enumerate(left_words, start=1):
        current = [left_index]
        for right_index, right_word in enumerate(right_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class CandidateAudit:
    passed: bool
    checks: dict[str, bool]
    candidate_roots: tuple[str, ...]
    retained_protected_root_count: int
    missing_protected_roots: tuple[str, ...]
    retained_trigger_roots: tuple[str, ...]
    low_score_query_ids: tuple[str, ...]
    minimum_object_score: float
    edit_distance: int
    word_count: int
    cycle: dict | None


def audit_revision_candidate(
    original_sentence: str,
    trigger_roots: Iterable[str],
    protected_roots: Iterable[str],
    candidate_text: str,
    extraction: dict,
    scores: dict[str, float],
) -> CandidateAudit:
    trigger_roots = set(trigger_roots)
    protected_roots = set(protected_roots)
    candidate_roots = {row["root_lemma"] for row in extraction["candidates"]}
    query_ids = [row["query_id"] for row in extraction["candidates"]]
    missing_scores = [query_id for query_id in query_ids if query_id not in scores]
    if missing_scores:
        raise KeyError(f"missing candidate verifier scores: {missing_scores[:5]}")
    low_score_query_ids = tuple(
        sorted(query_id for query_id in query_ids if float(scores[query_id]) <= 0.1)
    )
    missing_protected = tuple(sorted(protected_roots - candidate_roots))
    retained_triggers = tuple(sorted(trigger_roots & candidate_roots))
    word_count = len(candidate_text.split())
    original_words = len(original_sentence.split())
    cycle = exact_cycle(candidate_text)
    checks = {
        "exactly_one_nonempty_sentence": len(extraction["sentences"]) == 1
        and bool(candidate_text.strip()),
        "no_high_confidence_absent_object": not low_score_query_ids,
        "all_protected_roots_retained": not missing_protected,
        "all_trigger_roots_removed": not retained_triggers,
        "length_between_half_and_1_25": 0.5 * original_words
        <= word_count
        <= 1.25 * original_words,
        "no_exact_cycle": cycle is None,
    }
    candidate_scores = [float(scores[query_id]) for query_id in query_ids]
    return CandidateAudit(
        passed=all(checks.values()),
        checks=checks,
        candidate_roots=tuple(sorted(candidate_roots)),
        retained_protected_root_count=len(protected_roots & candidate_roots),
        missing_protected_roots=missing_protected,
        retained_trigger_roots=retained_triggers,
        low_score_query_ids=low_score_query_ids,
        minimum_object_score=min(candidate_scores, default=1.0),
        edit_distance=levenshtein_words(original_sentence, candidate_text),
        word_count=word_count,
        cycle=cycle,
    )


def choose_revision(original_sentence: str, candidates: dict[str, dict], allowed_modes):
    """Choose a passing candidate by the frozen VERA lexicographic policy."""

    mode_preference = {"base": 0, "vhr": 1}
    passing = []
    audits = {}
    for mode in allowed_modes:
        candidate = candidates[mode]
        audit = audit_revision_candidate(
            original_sentence=original_sentence,
            trigger_roots=candidate["trigger_roots"],
            protected_roots=candidate["protected_roots"],
            candidate_text=candidate["text"],
            extraction=candidate["extraction"],
            scores=candidate["scores"],
        )
        audits[mode] = audit
        if audit.passed:
            passing.append((mode, candidate, audit))
    if not passing:
        return None, audits
    passing.sort(
        key=lambda row: (
            -row[2].retained_protected_root_count,
            -row[2].minimum_object_score,
            row[2].edit_distance,
            mode_preference[row[0]],
        )
    )
    mode, candidate, audit = passing[0]
    return {"mode": mode, "text": candidate["text"], "audit": audit}, audits


def apply_sentence_revisions(source_text: str, source_extraction: dict, replacements):
    """Replace selected sentence spans while preserving every untouched byte."""

    replacement_by_index = {int(key): value for key, value in replacements.items()}
    pieces = []
    cursor = 0
    seen = set()
    for sentence in source_extraction["sentences"]:
        index = int(sentence["index"])
        start = int(sentence["start_char"])
        end = int(sentence["end_char"])
        if source_text[start:end] != sentence["text"]:
            raise AssertionError("source sentence span is not exact")
        pieces.append(source_text[cursor:start])
        if index in replacement_by_index:
            replacement = replacement_by_index[index].strip()
            if not replacement:
                raise ValueError("VERA replacement must be non-empty")
            pieces.append(replacement)
            seen.add(index)
        else:
            pieces.append(source_text[start:end])
        cursor = end
    pieces.append(source_text[cursor:])
    unknown = set(replacement_by_index) - seen
    if unknown:
        raise KeyError(f"replacement sentence indices not found: {sorted(unknown)}")
    output = "".join(pieces)
    if not output.strip():
        raise ValueError("VERA produced an empty caption")
    return output
