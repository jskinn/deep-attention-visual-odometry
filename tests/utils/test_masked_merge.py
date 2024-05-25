import unittest
import torch
from deep_attention_visual_odometry.utils import masked_merge_tensors


class TestMaskedMergeTensors(unittest.TestCase):
    def test_returns_none_if_everything_is_none(self):
        update_mask = torch.tensor([True])
        values, mask = masked_merge_tensors(None, None, None, None, update_mask)
        self.assertIsNone(values)
        self.assertIsNone(mask)

    def test_returns_values_from_mask_2_if_update_mask_is_true(self):
        values_2 = torch.tensor([1, 2, 3])
        update_mask = torch.tensor([True, False, True])
        out_values, mask = masked_merge_tensors(None, None, values_2, None, update_mask)
        self.assertTrue(torch.equal(mask, update_mask))
        self.assertTrue(torch.equal(out_values, values_2))

    def test_combines_mask_2_with_update_mask(self):
        values_2 = torch.tensor([1, 2, 3])
        mask_2 = torch.tensor([True, True, False])
        update_mask = torch.tensor([True, False, True])
        out_values, mask = masked_merge_tensors(
            None, None, values_2, mask_2, update_mask
        )
        self.assertTrue(torch.equal(mask, torch.tensor([True, False, False])))
        self.assertTrue(torch.equal(out_values, values_2))

    def test_returns_values_from_mask_1_if_update_mask_is_false(self):
        values_1 = torch.tensor([1, 2, 3])
        update_mask = torch.tensor([False, False, True])
        out_values, mask = masked_merge_tensors(values_1, None, None, None, update_mask)
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(out_values, values_1))

    def test_combines_mask_1_with_update_mask(self):
        values_1 = torch.tensor([1, 2, 3])
        mask_1 = torch.tensor([True, True, False])
        update_mask = torch.tensor([False, True, False])
        out_values, mask = masked_merge_tensors(
            values_1, mask_1, None, None, update_mask
        )
        self.assertTrue(torch.equal(mask, torch.tensor([True, False, False])))
        self.assertTrue(torch.equal(out_values, values_1))

    def test_combines_values_1_and_2_with_update_mask(self):
        values_1 = torch.tensor([1, 2, 3])
        values_2 = torch.tensor([0.1, 0.2, 0.3])
        update_mask = torch.tensor([False, True, False])
        out_values, mask = masked_merge_tensors(
            values_1, None, values_2, None, update_mask
        )
        self.assertIsNone(mask)
        self.assertTrue(torch.equal(out_values, torch.tensor([1.0, 0.2, 3.0])))

    def test_combines_masks_1_and_2_with_update_masks(self):
        values_1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        mask_1 = torch.tensor([True, False, True, False, True, False, True, False])
        values_2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        mask_2 = torch.tensor([True, True, False, False, True, True, False, False])
        update_mask = torch.tensor([True, True, True, True, False, False, False, False])
        out_values, mask = masked_merge_tensors(
            values_1, mask_1, values_2, mask_2, update_mask
        )
        self.assertTrue(
            torch.equal(
                out_values, torch.tensor([0.1, 0.2, 0.3, 0.4, 5.0, 6.0, 7.0, 8.0])
            )
        )
        self.assertTrue(
            torch.equal(
                mask, torch.tensor([True, True, False, False, True, False, True, False])
            )
        )

    def test_treats_none_mask_1_as_all_true_if_values_1_is_defined(self):
        values_1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        values_2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        mask_2 = torch.tensor([True, True, False, False, True, True, False, False])
        update_mask = torch.tensor([True, True, True, True, False, False, False, False])
        out_values, mask = masked_merge_tensors(
            values_1, None, values_2, mask_2, update_mask
        )
        self.assertTrue(
            torch.equal(
                out_values, torch.tensor([0.1, 0.2, 0.3, 0.4, 5.0, 6.0, 7.0, 8.0])
            )
        )
        self.assertTrue(
            torch.equal(
                mask, torch.tensor([True, True, False, False, True, True, True, True])
            )
        )

    def test_treats_none_mask_2_as_all_true_if_values_2_is_defined(self):
        values_1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        mask_1 = torch.tensor([True, False, True, False, True, False, True, False])
        values_2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        update_mask = torch.tensor([True, True, True, True, False, False, False, False])
        out_values, mask = masked_merge_tensors(
            values_1, mask_1, values_2, None, update_mask
        )
        self.assertTrue(
            torch.equal(
                out_values, torch.tensor([0.1, 0.2, 0.3, 0.4, 5.0, 6.0, 7.0, 8.0])
            )
        )
        self.assertTrue(
            torch.equal(
                mask, torch.tensor([True, True, True, True, True, False, True, False])
            )
        )

    def test_handles_multidimensional_masks(self):
        values_1 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        mask_1 = torch.tensor(
            [[True, False], [True, False], [True, False], [True, False]]
        )
        values_2 = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        mask_2 = torch.tensor(
            [[True, True], [False, False], [True, True], [False, False]]
        )
        update_mask = torch.tensor(
            [[True, True], [True, True], [False, False], [False, False]]
        )
        out_values, mask = masked_merge_tensors(
            values_1, mask_1, values_2, mask_2, update_mask
        )
        self.assertTrue(
            torch.equal(
                out_values,
                torch.tensor([[0.1, 0.2], [0.3, 0.4], [5.0, 6.0], [7.0, 8.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                mask,
                torch.tensor(
                    [[True, True], [False, False], [True, False], [True, False]]
                ),
            )
        )

    def test_handles_more_value_dimensions_than_mask(self):
        values_1 = torch.tensor(list(range(4 * 2 * 3 * 7))).reshape(4, 2, 3, 7)
        mask_1 = torch.tensor(
            [[True, False], [True, False], [True, False], [True, False]]
        )
        values_2 = values_1 / 10
        mask_2 = torch.tensor(
            [[True, True], [False, False], [True, True], [False, False]]
        )
        update_mask = torch.tensor(
            [[True, True], [True, True], [False, False], [False, False]]
        )
        expected_values = torch.stack(
            [
                torch.stack([values_2[0, 0, :, :], values_2[0, 1, :, :]]),
                torch.stack([values_2[1, 0, :, :], values_2[1, 1, :, :]]),
                torch.stack([values_1[2, 0, :, :], values_1[2, 1, :, :]]),
                torch.stack([values_1[3, 0, :, :], values_1[3, 1, :, :]]),
            ]
        )

        out_values, mask = masked_merge_tensors(
            values_1, mask_1, values_2, mask_2, update_mask
        )

        self.assertTrue(torch.equal(out_values, expected_values))
        self.assertTrue(
            torch.equal(
                mask,
                torch.tensor(
                    [[True, True], [False, False], [True, False], [True, False]]
                ),
            )
        )
