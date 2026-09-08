#!/usr/bin/env python3
"""Unit tests for the preregistered MGR mention-verifier policy."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_mgr import (
    align_items,
    combine_scores,
    crop_box,
    is_triggered,
    parse_grounding_box,
    parse_selflist_output,
    spacy_candidate_overlaps,
    strip_leading_determiner,
)


def test_parse_selflist_output():
    text = (
        "1. bus\n"
        "- The dog\n"
        "* an extremely long description item that goes on for many many words here\n"
        "cat.\n"
        "\n"
        "2) person\n"
    )
    items, stats = parse_selflist_output(text)
    assert items == ["bus", "dog", "cat", "person"], items
    assert stats["too_long"] == 1
    assert stats["empty"] == 1
    assert stats["kept"] == 4

    many = "\n".join(f"item{i}" for i in range(50))
    items, stats = parse_selflist_output(many)
    assert len(items) == 40
    assert stats["beyond_max"] == 10


def test_strip_leading_determiner():
    assert strip_leading_determiner("the dog") == "dog"
    assert strip_leading_determiner("a red car") == "red car"
    assert strip_leading_determiner("dog") == "dog"


def test_align_items_exact_and_plural():
    caption = "A bus is next to the BUS depot. The stickers on the bus are new."
    mentions, unaligned = align_items(caption, ["bus", "sticker", "moon rock"])
    keys = [m.mention_key for m in mentions]
    assert keys == ["bus", "stickers"], keys
    bus = mentions[0]
    assert bus.alignment == "exact"
    assert len(bus.occurrences) == 3
    assert bus.occurrences[0].text == "bus"
    assert bus.occurrences[1].text == "BUS"
    assert bus.occurrences[2].text == "bus"
    stickers = mentions[1]
    assert stickers.alignment == "plural"
    assert stickers.occurrences[0].text == "stickers"
    assert unaligned == ["moon rock"]


def test_align_items_word_boundaries():
    caption = "There is a classic car in the carport."
    mentions, unaligned = align_items(caption, ["car"])
    assert len(mentions[0].occurrences) == 1
    assert mentions[0].occurrences[0].text == "car"
    assert unaligned == []


def test_parse_grounding_box():
    box, reason = parse_grounding_box('```json\n[{"bbox_2d": [46, 0, 489, 375]}]\n```')
    assert box == [46.0, 0.0, 489.0, 375.0] and reason is None
    box, reason = parse_grounding_box("<ref>cat</ref><box>[[1,2,30,40]]</box>")
    assert box == [1.0, 2.0, 30.0, 40.0] and reason is None
    box, reason = parse_grounding_box('[{"point_2d": [5, 6]}]')
    assert box is None and reason == "point_only"
    box, reason = parse_grounding_box("There are none.")
    assert box is None and reason == "no_box_output"
    box, reason = parse_grounding_box('[{"bbox_2d": [10, 10, 10, 40]}]')
    assert box is None and reason == "degenerate_box"


def test_crop_box():
    crop, reason = crop_box([100, 100, 200, 140], 640, 480)
    assert crop == [90, 96, 210, 144] and reason is None
    crop, reason = crop_box([0, 0, 640, 480], 640, 480)
    assert crop == [0, 0, 640, 480]
    crop, reason = crop_box([10.0, 10.0, 10.5, 10.5], 640, 480)
    assert crop is None and reason == "degenerate_box"
    crop, reason = crop_box([600, 400, 700, 500], 640, 480)
    assert crop == [590, 390, 640, 480] and reason is None


def test_combine_scores():
    assert combine_scores(0.02, 0.9) == 0.9
    assert combine_scores(0.02, 0.02) == 0.02
    assert combine_scores(0.02, None) == 0.02
    assert combine_scores(0.7, 0.1) == 0.7
    assert is_triggered(combine_scores(0.02, 0.9)) is False
    assert is_triggered(combine_scores(0.02, None)) is True


def test_spacy_overlap():
    selflist = [
        {
            "occurrences": [
                {"start_char": 10, "end_char": 14},
                {"start_char": 40, "end_char": 44},
            ]
        }
    ]
    overlapping = {"occurrences": [{"start_char": 13, "end_char": 20}]}
    disjoint = {"occurrences": [{"start_char": 15, "end_char": 20}]}
    assert spacy_candidate_overlaps(overlapping, selflist) is True
    assert spacy_candidate_overlaps(disjoint, selflist) is False


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"pass {name}")
    print("all tests passed")
