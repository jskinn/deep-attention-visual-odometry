import torch
from torch.autograd import gradcheck
from deep_attention_visual_odometry.utils import inverse_curvature


def test_output_reduces_final_dimension_to_one():
    step = torch.randn(5, 1, 2, 7)
    delta_gradient = torch.randn(5, 1, 2, 7)
    results = inverse_curvature(step, delta_gradient)
    assert results.shape == (5, 1, 2, 1)


def test_output_is_zero_or_positive():
    step = torch.randn(15, 7)
    delta_gradient = torch.randn(15, 7)
    results = inverse_curvature(step, delta_gradient)
    assert torch.all(torch.greater_equal(results, 0.0))


def test_result_is_one_on_dot_product_when_dot_product_is_positive():
    step = torch.randn(10, 5)
    delta_gradient = torch.randn(10, 5)
    delta_gradient = delta_gradient + 1.1 * torch.linalg.norm(delta_gradient, dim=-1, keepdim=True) * step
    dot_product = (delta_gradient * step).sum(dim=-1, keepdims=True)
    assert torch.all(dot_product > 0)
    results = inverse_curvature(step, delta_gradient)
    assert results.shape == (10, 1)
    assert torch.equal(results, 1.0 / dot_product)


def test_result_is_zero_when_dot_product_is_zero():
    step = torch.tensor([2., 1., 0])
    delta_gradient = torch.tensor([1., -2., 0])
    results = inverse_curvature(step, delta_gradient)
    assert torch.equal(results, torch.zeros(1))


def test_result_is_zero_when_dot_product_is_negative():
    step = torch.randn(10, 5)
    step_direction = step / torch.linalg.norm(step, dim=-1, keepdim=True)
    delta_gradient = torch.randn(10, 5)
    delta_gradient = delta_gradient - (step_direction * delta_gradient).sum(dim=-1, keepdims=True) * step_direction
    delta_gradient = delta_gradient - (0.1 + torch.rand(10, 1)) * step
    dot_product = (delta_gradient * step).sum(dim=-1, keepdims=True)
    assert torch.all(dot_product < 0)
    results = inverse_curvature(step, delta_gradient)
    assert torch.equal(results, torch.zeros(10, 1))


def test_gradcheck_random():
    step = torch.randn(100, 7, dtype=torch.double, requires_grad=True)
    delta_gradient = torch.randn(100, 7, dtype=torch.double, requires_grad=True)
    assert gradcheck(inverse_curvature, (step, delta_gradient), eps=1e-6, atol=1e-4)


def test_can_be_compiled():
    inverse_curvature_c = torch.compile(inverse_curvature)
    step = torch.randn(21, 6)
    delta_gradient = torch.randn(21, 6)
    results = inverse_curvature_c(step, delta_gradient)
    assert results.shape == (21, 1)
    assert torch.all(torch.greater_equal(results, 0.0))
