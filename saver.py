"""Pure helpers for isolated SAVER spatial diagnostics.

This module intentionally contains no generation or evaluation logic.  It maps
the square image-token grid used by LLaVA-1.5 back through the CLIP
shortest-edge resize and center crop to a raw PIL image.
"""

import math
from typing import Dict, Sequence, Tuple


def clip_center_crop_geometry(
    image_size: Tuple[int, int],
    shortest_edge: int,
    crop_size: Tuple[int, int],
) -> Dict[str, object]:
    """Reproduce CLIPImageProcessor resize and center-crop geometry.

    ``image_size`` follows PIL's ``(width, height)`` convention, while
    ``crop_size`` is ``(width, height)``.  Transformers resizes the shortest
    edge exactly and truncates the longer edge with ``int`` before applying a
    centered crop.
    """
    width, height = (int(value) for value in image_size)
    crop_width, crop_height = (int(value) for value in crop_size)
    shortest_edge = int(shortest_edge)
    if min(width, height, crop_width, crop_height, shortest_edge) <= 0:
        raise ValueError("Image, resize, and crop dimensions must be positive")

    if width <= height:
        resized_width = shortest_edge
        resized_height = int(shortest_edge * height / width)
    else:
        resized_height = shortest_edge
        resized_width = int(shortest_edge * width / height)

    crop_left = (resized_width - crop_width) // 2
    crop_top = (resized_height - crop_height) // 2
    if crop_left < 0 or crop_top < 0:
        raise ValueError(
            "SAVER raw-coordinate mapping does not support processor padding"
        )

    return {
        "raw_size": [width, height],
        "resized_size": [resized_width, resized_height],
        "crop_size": [crop_width, crop_height],
        "crop_offset": [crop_left, crop_top],
        "scale": [resized_width / width, resized_height / height],
    }


def patch_bbox_to_raw_bbox(
    patch_bbox: Sequence[int],
    grid_side: int,
    image_size: Tuple[int, int],
    shortest_edge: int,
    crop_size: Tuple[int, int],
) -> Dict[str, object]:
    """Map a half-open patch box ``[x0,y0,x1,y1]`` to a raw-image box.

    The returned raw box is also half-open and uses floor for its leading
    edges and ceil for its trailing edges so that it contains the complete
    processor-space patch region.
    """
    if len(patch_bbox) != 4:
        raise ValueError("patch_bbox must contain [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (int(value) for value in patch_bbox)
    grid_side = int(grid_side)
    if grid_side <= 0:
        raise ValueError("grid_side must be positive")
    if not (0 <= x0 < x1 <= grid_side and 0 <= y0 < y1 <= grid_side):
        raise ValueError(
            f"Invalid half-open patch bbox {list(patch_bbox)} for {grid_side}x{grid_side} grid"
        )

    geometry = clip_center_crop_geometry(image_size, shortest_edge, crop_size)
    raw_width, raw_height = geometry["raw_size"]
    resized_width, resized_height = geometry["resized_size"]
    crop_width, crop_height = geometry["crop_size"]
    crop_left, crop_top = geometry["crop_offset"]

    processor_x0 = x0 * crop_width / grid_side
    processor_y0 = y0 * crop_height / grid_side
    processor_x1 = x1 * crop_width / grid_side
    processor_y1 = y1 * crop_height / grid_side

    resized_x0 = crop_left + processor_x0
    resized_y0 = crop_top + processor_y0
    resized_x1 = crop_left + processor_x1
    resized_y1 = crop_top + processor_y1

    raw_x0 = max(0, min(raw_width, math.floor(resized_x0 * raw_width / resized_width)))
    raw_y0 = max(0, min(raw_height, math.floor(resized_y0 * raw_height / resized_height)))
    raw_x1 = max(0, min(raw_width, math.ceil(resized_x1 * raw_width / resized_width)))
    raw_y1 = max(0, min(raw_height, math.ceil(resized_y1 * raw_height / resized_height)))
    if raw_x1 <= raw_x0 or raw_y1 <= raw_y0:
        raise ValueError("Mapped raw-image crop is empty")

    return {
        **geometry,
        "grid_side": grid_side,
        "patch_bbox": [x0, y0, x1, y1],
        "processor_bbox": [processor_x0, processor_y0, processor_x1, processor_y1],
        "resized_bbox": [resized_x0, resized_y0, resized_x1, resized_y1],
        "raw_bbox": [raw_x0, raw_y0, raw_x1, raw_y1],
        "raw_area_fraction": (
            (raw_x1 - raw_x0) * (raw_y1 - raw_y0) / (raw_width * raw_height)
        ),
    }

