import torch
from torch.autograd import gradcheck
from deep_attention_visual_odometry.utils import interpolate_alpha


def test_output_is_same_shape_as_input():
    alpha1 = torch.randn(5, 1, 2, 3)
    alpha2 = torch.randn(5, 1, 2, 3)
    value1 = torch.randn(5, 1, 2, 3)
    value2 = torch.randn(5, 1, 2, 3)
    results = interpolate_alpha(alpha1, alpha2, value1, value2)
    assert results.shape == (5, 1, 2, 3)


def test_result_is_between_alphas():
    alpha1 = torch.randn(10, 4)
    alpha2 = torch.randn(10, 4)
    value1 = torch.randn(10, 4)
    value2 = torch.randn(10, 4)
    min_alpha = torch.minimum(alpha1, alpha2)
    max_alpha = torch.maximum(alpha1, alpha2)
    results = interpolate_alpha(alpha1, alpha2, value1, value2)
    assert results.shape == (10, 4)
    assert torch.all(torch.greater_equal(results, min_alpha))
    assert torch.all(torch.less_equal(results, max_alpha))


def test_linearly_interpolates():
    alpha1 = torch.tensor(-1.0)
    alpha2 = torch.tensor(2.0)
    value1 = torch.tensor(0.8)
    value2 = torch.tensor(-0.4)
    result = interpolate_alpha(alpha1, alpha2, value1, value2)
    assert result == 1.0


def test_bisects_when_interpolated_value_is_too_small():
    alpha1 = torch.tensor(-1.0)
    alpha2 = torch.tensor(2.0)
    value1 = torch.tensor(0.2)
    value2 = torch.tensor(0.4)
    result = interpolate_alpha(alpha1, alpha2, value1, value2)
    assert result == 0.5


def test_bisects_when_interpolated_value_is_too_large():
    alpha1 = torch.tensor(-3.0)
    alpha2 = torch.tensor(-1.0)
    value1 = torch.tensor(6.0)
    value2 = torch.tensor(1.0)
    result = interpolate_alpha(alpha1, alpha2, value1, value2)
    assert result == -2.0


def test_bisects_when_values_are_the_same():
    alpha1 = torch.tensor(16.0)
    alpha2 = torch.tensor(2.0)
    value1 = torch.tensor(4.0)
    value2 = torch.tensor(4.0)
    result = interpolate_alpha(alpha1, alpha2, value1, value2)
    assert result == 9.0


def test_gradcheck_interpolate():
    alpha1 = torch.randn(100, dtype=torch.double)
    alpha2 = torch.randn(100, dtype=torch.double)
    values1 = torch.randn(100, dtype=torch.double)
    true_alphas = (0.1 + 0.8 * torch.rand(100)) * (alpha2 - alpha1) + alpha1
    values2 = (alpha2 - true_alphas) * values1 / (alpha1 - true_alphas)
    alpha1.requires_grad_()
    alpha2.requires_grad_()
    values1.requires_grad_()
    values2.requires_grad_()
    assert gradcheck(
        interpolate_alpha, (alpha1, alpha2, values1, values2), eps=1e-6, atol=1e-4
    )


def test_gradcheck_bisect():
    alpha1 = torch.randn(100, dtype=torch.double)
    alpha2 = torch.randn(100, dtype=torch.double)
    values1 = torch.randn(100, dtype=torch.double)
    values2 = torch.randn(100, dtype=torch.double).abs() * values1
    alpha1.requires_grad_()
    alpha2.requires_grad_()
    values1.requires_grad_()
    values2.requires_grad_()
    assert gradcheck(
        interpolate_alpha, (alpha1, alpha2, values1, values2), eps=1e-6, atol=1e-4
    )


def test_gradcheck_random():
    alpha1 = torch.randn(100, dtype=torch.double, requires_grad=True)
    alpha2 = torch.randn(100, dtype=torch.double, requires_grad=True)
    values1 = torch.randn(100, dtype=torch.double, requires_grad=True)
    values2 = torch.randn(100, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        interpolate_alpha, (alpha1, alpha2, values1, values2), eps=1e-6, atol=1e-4
    )


def test_can_be_compiled():
    interpolate_alpha_c = torch.compile(interpolate_alpha)
    alpha1 = torch.randn(10)
    alpha2 = torch.randn(10)
    values1 = torch.randn(10)
    values2 = torch.randn(10)
    min_alpha = torch.minimum(alpha1, alpha2)
    max_alpha = torch.maximum(alpha1, alpha2)
    results = interpolate_alpha_c(alpha1, alpha2, values1, values2)
    assert results.shape == (10,)
    assert torch.all(torch.greater_equal(results, min_alpha))
    assert torch.all(torch.less_equal(results, max_alpha))
