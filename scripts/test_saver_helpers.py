import unittest

from saver import clip_center_crop_geometry, patch_bbox_to_raw_bbox


class SaverSpatialHelperTests(unittest.TestCase):
    def test_landscape_geometry_matches_clip_center_crop(self):
        geometry = clip_center_crop_geometry((640, 480), 336, (336, 336))
        self.assertEqual(geometry["resized_size"], [448, 336])
        self.assertEqual(geometry["crop_offset"], [56, 0])
        full = patch_bbox_to_raw_bbox(
            [0, 0, 24, 24], 24, (640, 480), 336, (336, 336)
        )
        self.assertEqual(full["raw_bbox"], [80, 0, 560, 480])

    def test_portrait_geometry_matches_clip_center_crop(self):
        geometry = clip_center_crop_geometry((480, 640), 336, (336, 336))
        self.assertEqual(geometry["resized_size"], [336, 448])
        self.assertEqual(geometry["crop_offset"], [0, 56])
        full = patch_bbox_to_raw_bbox(
            [0, 0, 24, 24], 24, (480, 640), 336, (336, 336)
        )
        self.assertEqual(full["raw_bbox"], [0, 80, 480, 560])

    def test_real_diagnostic_bbox_is_valid_and_local(self):
        mapped = patch_bbox_to_raw_bbox(
            [0, 18, 3, 21], 24, (640, 412), 336, (336, 336)
        )
        self.assertEqual(mapped["resized_size"], [521, 336])
        self.assertEqual(mapped["crop_offset"], [92, 0])
        self.assertEqual(mapped["raw_bbox"], [113, 309, 165, 361])
        self.assertLess(mapped["raw_area_fraction"], 0.02)

    def test_invalid_patch_bbox_is_rejected(self):
        with self.assertRaises(ValueError):
            patch_bbox_to_raw_bbox(
                [0, 0, 25, 24], 24, (640, 480), 336, (336, 336)
            )


if __name__ == "__main__":
    unittest.main()
