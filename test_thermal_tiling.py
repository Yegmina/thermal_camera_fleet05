import unittest
from unittest.mock import patch

import numpy as np
import torch

from thermal_detector import ROOT, ThermalDetector, merge_detections, tile_windows


class TilingTests(unittest.TestCase):
    def test_every_native_pixel_is_covered_in_both_orientations(self):
        for width, height in ((640, 512), (512, 640)):
            covered = np.zeros((height, width), dtype=bool)
            windows = tile_windows(width, height)
            self.assertEqual(len(windows), 4)
            for x1, y1, x2, y2 in windows:
                self.assertLessEqual(x2 - x1, 384)
                self.assertLessEqual(y2 - y1, 384)
                covered[y1:y2, x1:x2] = True
            self.assertTrue(covered.all())

    def make_fake_detector(self, simulate_oom=False):
        image = np.zeros((512, 640, 3), dtype=np.uint8)
        detector = object.__new__(ThermalDetector)
        detector.image_size = 640
        detector.batch_size = 5
        calls = []
        global_box = np.array([280, 200, 340, 300, 0.9], dtype=np.float32)

        def predict(crops, confidence):
            calls.append(len(crops))
            if simulate_oom and len(crops) > 2:
                raise torch.cuda.OutOfMemoryError('Simulated allocator budget exhaustion')
            outputs = []
            for crop in crops:
                self.assertTrue(np.shares_memory(image, crop))
                byte_offset = crop.ctypes.data - image.ctypes.data
                y, x_bytes = divmod(byte_offset, image.strides[0])
                x = x_bytes // image.strides[1]
                box = global_box.copy()
                box[[0, 2]] -= x
                box[[1, 3]] -= y
                outputs.append(box[None])
            return outputs

        detector.predict_batch = predict
        return detector, image, calls, global_box

    def test_tiles_are_batched_and_boxes_return_to_full_frame_once(self):
        detector, image, calls, expected = self.make_fake_detector()
        result = detector.detect(image)
        self.assertEqual(calls, [5])
        self.assertEqual(detector.last_input_count, 5)
        self.assertEqual(result.shape, (1, 5))
        np.testing.assert_allclose(result[0], expected)

    def test_budget_exhaustion_reduces_batch_without_losing_tiles(self):
        detector, image, calls, expected = self.make_fake_detector(simulate_oom=True)
        with patch('thermal_detector.torch.cuda.empty_cache'):
            result = detector.detect(image)
        self.assertEqual(calls, [5, 2, 2, 1])
        self.assertEqual(detector.batch_size, 2)
        np.testing.assert_allclose(result[0], expected)

    def test_native_resolution_cannot_be_silently_reduced(self):
        detector, image, _, _ = self.make_fake_detector()
        detector.image_size = 320
        with self.assertRaisesRegex(ValueError, 'shrink'):
            detector.detect(image)

    def test_separate_people_survive_duplicate_suppression(self):
        boxes = np.array([[20, 20, 80, 180, 0.9], [21, 21, 81, 181, 0.8],
                          [200, 20, 260, 180, 0.85]], dtype=np.float32)
        self.assertEqual(len(merge_detections([boxes])), 2)

    def test_general_photo_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'verified thermal-trained'):
            ThermalDetector(ROOT / 'models/yolo11l.pt')


if __name__ == '__main__':
    unittest.main()
