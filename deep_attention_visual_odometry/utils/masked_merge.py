import torch


def masked_merge_tensors(
    values_1: torch.Tensor | None,
    mask_1: torch.Tensor | None,
    values_2: torch.Tensor | None,
    mask_2: torch.Tensor | None,
    update_mask: torch.Tensor,
) -> tuple[torch.Tensor, None] | tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    Given two tensors of values, and two masks where those values are valid,
    merge those two tensors into a new single tensor of values and potentially a mask
    where the values are valid.

    Chooses values from values 1 where update_mask is false and 2 where it is true.
    All the masks must be the same shape.

    :param values_1:
    :param mask_1:
    :param values_2:
    :param mask_2:
    :param update_mask:
    :return:
    """
    if values_1 is None and values_2 is None:
        return None, None
    elif values_1 is not None and values_2 is not None:
        values_update_mask = update_mask
        if update_mask.ndim < values_1.ndim:
            values_update_mask = update_mask.reshape(
                *update_mask.shape,
                *(1 for _ in range(values_1.ndim - update_mask.ndim))
            ).tile(
                *(1 for _ in range(update_mask.ndim)),
                *values_1.shape[update_mask.ndim :]
            )
        merged_values = torch.where(values_update_mask, values_2, values_1)
        if mask_1 is None and mask_2 is None:
            return merged_values, None
        elif mask_1 is not None and mask_2 is not None:
            return merged_values, torch.where(update_mask, mask_2, mask_1)
        elif mask_1 is not None and mask_2 is None:
            # All the values from 2 are valid, but only those from 1 where the mask is true are valid
            return merged_values, torch.logical_or(mask_1, update_mask)
        else:
            # All the values from 1 are valid, those from 2 are valid if the mask is true.
            return merged_values, torch.logical_or(
                mask_2, torch.logical_not(update_mask)
            )
    elif values_1 is not None and values_2 is None:
        if mask_1 is not None:
            # None of the 2 values are valid, where the mask selects 1 and 1 is valid
            return values_1, torch.logical_and(mask_1, torch.logical_not(update_mask))
        return values_1, torch.logical_not(update_mask)
    else:
        if mask_2 is not None:
            # None of the 1 values are valid, 2 are only valid where the mask is true
            return values_2, torch.logical_and(mask_2, update_mask)
        return values_2, update_mask
