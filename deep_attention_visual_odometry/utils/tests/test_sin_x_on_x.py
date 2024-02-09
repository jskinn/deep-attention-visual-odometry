import torch
from torch.autograd import gradcheck
from deep_attention_visual_odometry.utils import sin_x_on_x


def test_output_is_same_shape_as_input():
    inputs = torch.randn(5, 1, 2, 3)
    results = sin_x_on_x(inputs)
    assert results.shape == (5, 1, 2, 3)


def test_is_one_at_zero():
    inputs = torch.linspace(-0.01, 0.01, 5)
    results = sin_x_on_x(inputs)
    assert results[2] == 1.0
    assert torch.all(results[[0, 1, 3, 4]] < 1.0)


def test_computes_gradients():
    inputs = (torch.pi * torch.randn(10, dtype=torch.double)).requires_grad_()
    results = sin_x_on_x(inputs)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))
    assert torch.all(torch.greater(torch.abs(inputs.grad), 0))


def test_computes_gradients_when_input_is_zero():
    inputs = torch.linspace(-0.01, 0.01, 5, requires_grad=True)
    results = sin_x_on_x(inputs)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))
    assert torch.all(torch.greater_equal(torch.abs(inputs.grad), 0))


def test_gradcheck():
    inputs = torch.pi * torch.randn(100, dtype=torch.double, requires_grad=True)
    assert gradcheck(sin_x_on_x, inputs, eps=1e-6, atol=1e-4)


def test_can_be_compiled():
    sin_x_on_x_c = torch.compile(sin_x_on_x)
    inputs = torch.pi * torch.randn(10, dtype=torch.double)
    results = sin_x_on_x_c(inputs)
    assert results.shape == (10,)
    assert torch.all(results.abs() < 1.0)
