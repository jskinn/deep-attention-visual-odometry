import torch
from torch.autograd import gradcheck
from deep_attention_visual_odometry.utils import one_minus_cos_x_on_x_squared


def test_output_is_same_shape_as_input():
    inputs = torch.randn(5, 1, 2, 3)
    results = one_minus_cos_x_on_x_squared(inputs)
    assert results.shape == (5, 1, 2, 3)


def test_is_half_at_zero():
    num_steps = 101
    zero_idx = (num_steps - 1) // 2
    inputs = torch.linspace(-0.1, 0.1, num_steps)
    results = one_minus_cos_x_on_x_squared(inputs)
    assert results[zero_idx] == 0.5
    assert torch.all(results[[idx for idx in range(num_steps) if idx != zero_idx]] < 0.5)


def test_computes_gradients():
    inputs = (torch.pi * torch.randn(10, dtype=torch.double)).requires_grad_()
    results = one_minus_cos_x_on_x_squared(inputs)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))
    assert torch.all(torch.greater(torch.abs(inputs.grad), 0))


def test_computes_gradients_when_input_is_zero():
    inputs = torch.linspace(-0.01, 0.01, 5, requires_grad=True)
    results = one_minus_cos_x_on_x_squared(inputs)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))
    assert torch.all(torch.greater_equal(torch.abs(inputs.grad), 0))


def test_gradcheck_large():
    inputs = torch.pi * torch.randn(100, dtype=torch.double, requires_grad=True)
    assert gradcheck(one_minus_cos_x_on_x_squared, inputs, eps=1e-6, atol=1e-4)


def test_gradcheck_small():
    inputs = 0.005 * torch.randn(100, dtype=torch.double, requires_grad=True)
    assert gradcheck(one_minus_cos_x_on_x_squared, inputs, eps=1e-6, atol=1e-4)


def test_gradcheck_near_transitions():
    inputs = 0.005 * torch.randn(
        100, dtype=torch.double, requires_grad=True
    ) + torch.cat([0.01 * torch.ones(50), -0.01 * torch.ones(50)])
    assert gradcheck(one_minus_cos_x_on_x_squared, inputs, eps=1e-6, atol=1e-4)


def test_can_be_compiled():
    one_minus_cos_x_on_x_squared_c = torch.compile(one_minus_cos_x_on_x_squared)
    inputs = torch.pi * torch.randn(10, dtype=torch.double)
    results = one_minus_cos_x_on_x_squared_c(inputs)
    assert results.shape == (10,)
    assert torch.all(results.abs() < 1.0)
