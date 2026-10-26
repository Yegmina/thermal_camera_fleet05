"""False-positive filtering and the official DETR coordinate decoder."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import cv2

from thermal_detector import PersonConfirmation, corroborate_detections, prepare_detr_inputs


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.person = np.array([[100, 80, 160, 260, .9]], dtype=np.float32)
        self.empty = np.empty((0, 5), dtype=np.float32)

    def test_independent_detector_rejects_unmatched_false_positive(self):
        distractor = np.array([[300, 10, 308, 25, .85]], dtype=np.float32)
        primary = np.concatenate([self.person, distractor])
        verifier = np.array([[98, 82, 163, 258, .7]], dtype=np.float32)
        np.testing.assert_allclose(corroborate_detections(primary, verifier), self.person)
        self.assertEqual(len(corroborate_detections(primary, self.empty)), 0)

    def test_three_consecutive_observations_then_immediate_removal(self):
        confirmation = PersonConfirmation(3)
        for expected in (0, 0, 1):
            self.assertEqual(len(confirmation.update(self.person)), expected)
        self.assertEqual(len(confirmation.update(self.empty)), 0)
        self.assertEqual(len(confirmation.update(self.person)), 0)

    def test_intermittent_false_positive_never_confirms(self):
        confirmation = PersonConfirmation(3)
        for boxes in [self.person, self.person, self.empty] * 3:
            self.assertEqual(len(confirmation.update(boxes)), 0)

    def test_previous_box_cannot_confirm_two_people(self):
        confirmation = PersonConfirmation(2)
        confirmation.update(self.person)
        nearby = self.person.copy()
        nearby[:, [0, 2]] += 5
        nearby[:, 4] = .8
        result = confirmation.update(np.concatenate([self.person, nearby]))
        self.assertEqual(len(result), 1)

    def test_official_detr_decoder_preserves_native_non_square_coordinates(self):
        from transformers import DetrImageProcessor
        processor = DetrImageProcessor(do_resize=False)
        prediction = SimpleNamespace(
            logits=torch.tensor([[[-10., -10., 10., -10.]]]),
            pred_boxes=torch.tensor([[[.5, .5, .25, .5]]]))
        result = processor.post_process_object_detection(
            prediction, threshold=.65, target_sizes=torch.tensor([[640, 512]]))[0]
        self.assertEqual(result['labels'].tolist(), [2])
        np.testing.assert_allclose(result['boxes'].numpy(), [[192, 160, 320, 480]])

    def test_normalization_and_padding_match_official_detr_processor(self):
        from transformers import DetrImageProcessor
        rng = np.random.default_rng(42)
        images = [rng.integers(0, 256, (20, 32, 3), dtype=np.uint8),
                  rng.integers(0, 256, (32, 24, 3), dtype=np.uint8)]
        expected = DetrImageProcessor(do_resize=False)(
            images=[cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images], return_tensors='pt')
        actual = prepare_detr_inputs(images, 32, 'cpu', False)
        torch.testing.assert_close(actual['pixel_values'], expected['pixel_values'], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual['pixel_mask'], expected['pixel_mask'])


if __name__ == '__main__':
    unittest.main()
