"""Architecture-independent extraction helpers for preregistered Qwen SEAR.

This module is intentionally separate from the historical VHR generation path.
It only parses already-generated captions and never imports or mutates VHR state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import time
from typing import Iterable


DETERMINERS = frozenset({"a", "an", "the", "this", "that", "these", "those"})
DISCOURSE_STOPLIST = frozenset(
    {
        "image",
        "photo",
        "picture",
        "scene",
        "view",
        "background",
        "foreground",
        "setting",
        "area",
        "side",
        "color",
        "lighting",
        "atmosphere",
        "impression",
        "focus",
        "design",
        "sense",
        "way",
        "part",
        "center",
        "top",
        "bottom",
    }
)


@dataclass(frozen=True)
class Sentence:
    index: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class CandidateOccurrence:
    root_lemma: str
    phrase: str
    sentence_index: int
    start_char: int
    end_char: int
    normalized_surface: str
    normalized_lemmas: str


@dataclass(frozen=True)
class Candidate:
    """One verifier query, deduplicated by noun-chunk root lemma."""

    root_lemma: str
    phrase: str
    normalized_surface: str
    normalized_lemmas: str
    occurrences: tuple[CandidateOccurrence, ...]


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def normalize_match_text(text: str) -> str:
    """Normalize punctuation and whitespace without using CHAIR vocabulary."""

    return normalize_space(re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _strip_leading_determiner(chunk) -> tuple[int, int]:
    tokens = list(chunk)
    start = 0
    while start < len(tokens) and tokens[start].lower_ in DETERMINERS:
        start += 1
    return start, len(tokens)


def _candidate_occurrence(chunk, sentence_index: int) -> CandidateOccurrence | None:
    root_lemma = chunk.root.lemma_.lower().strip()
    if chunk.root.pos_ == "PRON" or root_lemma in DISCOURSE_STOPLIST:
        return None

    relative_start, relative_end = _strip_leading_determiner(chunk)
    tokens = list(chunk)[relative_start:relative_end]
    if not tokens:
        return None
    phrase = chunk.doc[tokens[0].i : tokens[-1].i + 1].text
    phrase = phrase.strip()
    if not phrase:
        return None
    normalized_lemmas = normalize_match_text(
        " ".join(token.lemma_.lower() for token in tokens if not token.is_punct)
    )
    return CandidateOccurrence(
        root_lemma=root_lemma,
        phrase=phrase,
        sentence_index=sentence_index,
        start_char=tokens[0].idx,
        end_char=tokens[-1].idx + len(tokens[-1]),
        normalized_surface=normalize_match_text(phrase),
        normalized_lemmas=normalized_lemmas,
    )


def extract_caption(text: str, nlp) -> dict:
    """Extract sentences, noun occurrences, and root-deduplicated queries."""

    doc = nlp(text)
    sentences: list[Sentence] = []
    by_root: dict[str, list[CandidateOccurrence]] = {}
    for sentence_index, sent in enumerate(doc.sents):
        sentences.append(
            Sentence(
                index=sentence_index,
                text=sent.text,
                start_char=sent.start_char,
                end_char=sent.end_char,
            )
        )
        for chunk in sent.noun_chunks:
            occurrence = _candidate_occurrence(chunk, sentence_index)
            if occurrence is None:
                continue
            by_root.setdefault(occurrence.root_lemma, []).append(occurrence)

    candidates = []
    for root_lemma, occurrences in by_root.items():
        representative = occurrences[0]
        candidates.append(
            Candidate(
                root_lemma=root_lemma,
                phrase=representative.phrase,
                normalized_surface=representative.normalized_surface,
                normalized_lemmas=representative.normalized_lemmas,
                occurrences=tuple(occurrences),
            )
        )
    return {
        "sentences": [asdict(sentence) for sentence in sentences],
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def mention_is_captured(mention: str, candidates: Iterable[dict]) -> bool:
    """Return whether a CHAIR mention is contained in any generic noun phrase.

    CHAIR is used only by the audit caller. Matching accepts either surface or
    spaCy-lemma containment so ordinary plurals do not create false misses.
    """

    normalized = normalize_match_text(mention)
    if not normalized:
        return False
    padded_mention = f" {normalized} "
    for candidate in candidates:
        representations = [candidate]
        representations.extend(candidate.get("occurrences", ()))
        for representation in representations:
            for key in ("normalized_surface", "normalized_lemmas"):
                value = f" {representation[key]} "
                if padded_mention in value:
                    return True
    return False


def filter_caption(text: str, extraction: dict, scores: dict[str, float], threshold=0.1):
    """Apply the frozen sentence-retention rule to one extracted caption."""

    start = time.perf_counter()
    sentences = extraction["sentences"]
    if not sentences:
        if not text:
            raise ValueError("SEAR cannot return an empty caption")
        return text, {
            "triggered": False,
            "deleted_sentence_indices": [],
            "retained_sentence_indices": [],
            "fallback_used": True,
            "filter_elapsed_seconds": time.perf_counter() - start,
        }

    per_sentence_scores = {sentence["index"]: [] for sentence in sentences}
    per_sentence_queries = {sentence["index"]: [] for sentence in sentences}
    for candidate in extraction["candidates"]:
        query_id = candidate.get("query_id")
        if not query_id or query_id not in scores:
            raise KeyError(f"missing verifier score for {query_id!r}")
        score = float(scores[query_id])
        for occurrence in candidate["occurrences"]:
            sentence_index = int(occurrence["sentence_index"])
            per_sentence_scores[sentence_index].append(score)
            per_sentence_queries[sentence_index].append(query_id)

    keep = []
    for sentence in sentences:
        sentence_scores = per_sentence_scores[sentence["index"]]
        keep.append(not any(score <= threshold for score in sentence_scores))

    fallback_used = False
    if not any(keep):
        fallback_used = True
        minima = [
            min(per_sentence_scores[sentence["index"]], default=float("inf"))
            for sentence in sentences
        ]
        fallback_index = max(range(len(sentences)), key=lambda index: minima[index])
        keep[fallback_index] = True

    retained = [sentence for sentence, should_keep in zip(sentences, keep) if should_keep]
    deleted = [sentence for sentence, should_keep in zip(sentences, keep) if not should_keep]
    pieces = []
    previous_index = None
    previous_end = None
    for sentence in retained:
        exact = text[sentence["start_char"] : sentence["end_char"]]
        if exact != sentence["text"]:
            raise AssertionError("retained sentence is not an exact source substring")
        if previous_index is not None:
            if sentence["index"] == previous_index + 1:
                pieces.append(text[previous_end : sentence["start_char"]])
            else:
                pieces.append("\n\n")
        pieces.append(exact)
        previous_index = sentence["index"]
        previous_end = sentence["end_char"]
    filtered = "".join(pieces).strip()
    if not filtered:
        raise ValueError("SEAR produced an empty caption")
    elapsed = time.perf_counter() - start
    return filtered, {
        "triggered": bool(deleted),
        "deleted_sentence_indices": [row["index"] for row in deleted],
        "retained_sentence_indices": [row["index"] for row in retained],
        "deleted_query_ids": sorted(
            {
                query_id
                for row in deleted
                for query_id in per_sentence_queries[row["index"]]
                if scores[query_id] <= threshold
            }
        ),
        "fallback_used": fallback_used,
        "filter_elapsed_seconds": elapsed,
    }
