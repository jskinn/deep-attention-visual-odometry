import torch
from torch.autograd import gradcheck
from deep_attention_visual_odometry.utils import (
    sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth,
)


def test_output_is_same_shape_as_input():
    inputs = torch.randn(5, 1, 2, 3)
    results = sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth(inputs)
    assert results.shape == (5, 1, 2, 3)


def test_is_minus_one_twelfth_at_zero():
    num_steps = 101
    zero_idx = (num_steps - 1) // 2
    inputs = torch.linspace(-0.5, 0.5, num_steps)
    results = sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth(inputs)
    assert results[zero_idx] == -1.0 / 12.0
    assert torch.all(
        results[[idx for idx in range(num_steps) if idx != zero_idx]] > -1.0 / 12.0
    )


def test_computes_gradients():
    inputs = (torch.pi * torch.randn(10, dtype=torch.double)).requires_grad_()
    results = sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth(inputs)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))
    assert torch.all(torch.greater(torch.abs(inputs.grad), 0))


def test_computes_gradients_when_input_is_zero():
    inputs = torch.linspace(-0.01, 0.01, 5, requires_grad=True)
    results = sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth(inputs)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))
    assert torch.all(torch.greater_equal(torch.abs(inputs.grad), 0))


def test_gradcheck_large():
    inputs = torch.pi * torch.randn(100, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth,
        inputs,
        eps=1e-6,
        atol=1e-4,
    )


def test_gradcheck_small():
    inputs = 0.005 * torch.randn(100, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth,
        inputs,
        eps=1e-6,
        atol=1e-4,
    )


def test_gradcheck_near_transitions():
    inputs = 0.005 * torch.randn(
        100, dtype=torch.double, requires_grad=True
    ) + torch.cat([0.25 * torch.ones(50), -0.25 * torch.ones(50)])
    assert gradcheck(
        sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth,
        inputs,
        eps=1e-6,
        atol=1e-4,
    )


def test_can_be_compiled():
    sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth_c = torch.compile(
        sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth
    )
    inputs = torch.pi * torch.randn(10, dtype=torch.double)
    results = sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth_c(inputs)
    assert results.shape == (10,)
    assert torch.all(results.abs() < 1.0)
