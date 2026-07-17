#!/usr/bin/env python3
"""Deterministic CPU checks for the isolated SEAR noun-phrase extractor."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy

from qwen_sear import extract_caption, filter_caption, mention_is_captured


def main():
    nlp = spacy.load("en_core_web_sm")
    text = (
        "The image shows a man in a blue shirt beside two parked cars. "
        "In the background, the man is near a traffic light."
    )
    first = extract_caption(text, nlp)
    second = extract_caption(text, nlp)
    assert first == second

    by_root = {candidate["root_lemma"]: candidate for candidate in first["candidates"]}
    assert "image" not in by_root
    assert "background" not in by_root
    assert by_root["man"]["phrase"] == "man"
    assert len(by_root["man"]["occurrences"]) == 2
    assert by_root["shirt"]["phrase"] == "blue shirt"
    assert by_root["car"]["phrase"] == "two parked cars"
    assert text[0 : first["sentences"][0]["end_char"]] == first["sentences"][0]["text"]

    candidates = first["candidates"]
    assert mention_is_captured("car", candidates)
    assert mention_is_captured("traffic light", candidates)
    assert not mention_is_captured("elephant", candidates)

    repeated_root = extract_caption("A light is on. A traffic light is visible.", nlp)
    assert len([row for row in repeated_root["candidates"] if row["root_lemma"] == "light"]) == 1
    assert mention_is_captured("traffic light", repeated_root["candidates"])

    filtered_source = "A dog is visible. An elephant stands nearby. The sky is blue."
    filtered_extraction = extract_caption(filtered_source, nlp)
    scores = {}
    for index, candidate in enumerate(filtered_extraction["candidates"]):
        candidate["query_id"] = f"query-{index}"
        scores[candidate["query_id"]] = 0.01 if candidate["root_lemma"] == "elephant" else 0.99
    filtered, stats = filter_caption(filtered_source, filtered_extraction, scores)
    assert filtered == "A dog is visible.\n\nThe sky is blue."
    assert stats["deleted_sentence_indices"] == [1]
    assert stats["triggered"]

    all_bad = "An elephant is visible. A unicorn is nearby."
    all_bad_extraction = extract_caption(all_bad, nlp)
    all_bad_scores = {}
    for index, candidate in enumerate(all_bad_extraction["candidates"]):
        candidate["query_id"] = f"bad-{index}"
        all_bad_scores[candidate["query_id"]] = (
            0.08 if candidate["root_lemma"] == "elephant" else 0.02
        )
    fallback, fallback_stats = filter_caption(all_bad, all_bad_extraction, all_bad_scores)
    assert fallback == "An elephant is visible."
    assert fallback_stats["fallback_used"]
    print("Qwen SEAR extractor checks passed")


if __name__ == "__main__":
    main()
