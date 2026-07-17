#!/usr/bin/env python3
"""CPU unit checks for the frozen VERA-VHR acceptance and ranking policy."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy

from qwen_sear import extract_caption
from qwen_vera import (
    apply_sentence_revisions,
    audit_revision_candidate,
    choose_revision,
    exact_cycle,
    format_revision_prompt,
    levenshtein_words,
    sanitize_extraction,
    valid_visual_candidate,
)


def candidate(nlp, text, scores_by_root, trigger_roots=("unicorn",), protected=("man", "umbrella")):
    extraction = extract_caption(text, nlp)
    scores = {}
    for index, row in enumerate(extraction["candidates"]):
        row["query_id"] = f"candidate-{index}-{row['root_lemma']}"
        scores[row["query_id"]] = scores_by_root.get(row["root_lemma"], 0.99)
    return {
        "text": text,
        "extraction": extraction,
        "scores": scores,
        "trigger_roots": list(trigger_roots),
        "protected_roots": list(protected),
    }


def main():
    nlp = spacy.load("en_core_web_sm")
    original = "A man holds a red umbrella beside a unicorn."
    prompt = format_revision_prompt(original, ["a unicorn"])
    assert prompt == format_revision_prompt(original, ["a unicorn"])
    assert "Original sentence:" in prompt and "a unicorn" in prompt
    assert levenshtein_words("a b c", "a x c") == 1
    assert exact_cycle("a dog runs a dog runs a dog runs") is not None
    assert exact_cycle("a dog runs near a tree") is None

    markdown = extract_caption("This is a list:\n\n1. **Bed**: A bed is visible.", nlp)
    assert any(not valid_visual_candidate(row) for row in markdown["candidates"])
    cleaned = sanitize_extraction(markdown)
    assert all(valid_visual_candidate(row) for row in cleaned["candidates"])
    assert all(row["root_lemma"] != "*" for row in cleaned["candidates"])

    good = candidate(nlp, "A man holds a red umbrella.", {"man": 0.99, "umbrella": 0.95})
    good_audit = audit_revision_candidate(
        original, good["trigger_roots"], good["protected_roots"], good["text"], good["extraction"], good["scores"]
    )
    assert good_audit.passed

    missing = candidate(nlp, "A man stands nearby.", {"man": 0.99})
    missing_audit = audit_revision_candidate(
        original, missing["trigger_roots"], missing["protected_roots"], missing["text"], missing["extraction"], missing["scores"]
    )
    assert not missing_audit.passed
    assert not missing_audit.checks["all_protected_roots_retained"]

    retained = candidate(
        nlp,
        "A man holds a red umbrella beside a unicorn.",
        {"man": 0.99, "umbrella": 0.95, "unicorn": 0.9},
    )
    retained_audit = audit_revision_candidate(
        original, retained["trigger_roots"], retained["protected_roots"], retained["text"], retained["extraction"], retained["scores"]
    )
    assert not retained_audit.checks["all_trigger_roots_removed"]

    low_score = candidate(
        nlp,
        "A man holds a red umbrella beside a horse.",
        {"man": 0.99, "umbrella": 0.95, "horse": 0.01},
    )
    low_score_audit = audit_revision_candidate(
        original, low_score["trigger_roots"], low_score["protected_roots"], low_score["text"], low_score["extraction"], low_score["scores"]
    )
    assert not low_score_audit.checks["no_high_confidence_absent_object"]

    base = candidate(nlp, "A man holds a red umbrella.", {"man": 0.99, "umbrella": 0.8})
    vhr = candidate(nlp, "A man carries a red umbrella.", {"man": 0.99, "umbrella": 0.95})
    selected, audits = choose_revision(original, {"base": base, "vhr": vhr}, ("base", "vhr"))
    assert selected["mode"] == "vhr"
    assert audits["base"].passed and audits["vhr"].passed

    tied_base = candidate(nlp, "A man holds a red umbrella.", {"man": 0.99, "umbrella": 0.95})
    tied_vhr = candidate(nlp, "A man holds a red umbrella.", {"man": 0.99, "umbrella": 0.95})
    tied, _ = choose_revision(original, {"base": tied_base, "vhr": tied_vhr}, ("base", "vhr"))
    assert tied["mode"] == "base"

    source = "First sentence. A man holds a red umbrella beside a unicorn. Final sentence."
    source_extraction = extract_caption(source, nlp)
    replaced = apply_sentence_revisions(source, source_extraction, {1: good["text"]})
    assert replaced == "First sentence. A man holds a red umbrella. Final sentence."
    assert apply_sentence_revisions(source, source_extraction, {}) == source
    print("Qwen VERA policy checks passed")


if __name__ == "__main__":
    main()
