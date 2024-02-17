import torch
from torch.autograd import gradcheck
from deep_attention_visual_odometry.utils import project_vector_onto_axis


def test_output_is_same_shape_as_input():
    vector = torch.randn(5, 1, 2, 3)
    axis = torch.randn(5, 1, 2, 3)
    results = project_vector_onto_axis(vector, axis)
    assert results.shape == (5, 1, 2, 3)


def test_is_zero_if_axis_is_zero():
    vector = torch.tensor([1.0, -2.3, 4.0])
    axis = torch.zeros(3)
    results = project_vector_onto_axis(vector, axis)
    assert torch.equal(results, torch.zeros(3))


def test_result_is_vector_if_vector_and_axis_are_parallel():
    vector = torch.randn(5, 3)
    axis = torch.randn(5, 1) * vector
    results = project_vector_onto_axis(vector, axis)
    assert torch.all(torch.isclose(results, vector))


def test_result_is_vector_if_vector_and_axis_are_parallel_in_high_dimensions():
    vector = torch.randn(5, 18)
    axis = torch.randn(5, 1) * vector
    results = project_vector_onto_axis(vector, axis)
    assert torch.all(torch.isclose(results, vector))


def test_is_zero_if_vector_and_axis_are_purpendicular():
    vector = torch.randn(5, 3)
    other_vector = torch.randn(5, 3)
    axis = torch.cross(vector, other_vector)
    results = project_vector_onto_axis(vector, axis)
    assert torch.all(results.abs() < 1e-7)


def test_result_is_parallel_to_axis():
    vectors = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    axes = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    results = project_vector_onto_axis(vectors, axes)
    assert results.shape == (7, 3)
    results_norm = results / torch.linalg.norm(results, dim=-1, keepdim=True)
    axes_norm = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    dot_product = (results_norm * axes_norm).sum(dim=-1)
    assert torch.all(torch.isclose(dot_product.abs(), torch.ones_like(dot_product)))


def test_result_is_independent_of_axis_length():
    vector = torch.randn(5, 3)
    axis1 = torch.randn(5, 3)
    axis2 = torch.randn(5, 1) * axis1
    results1 = project_vector_onto_axis(vector, axis1)
    results2 = project_vector_onto_axis(vector, axis2)
    assert torch.all(torch.isclose(results1, results2))


def test_result_is_independent_of_axis_length_for_many_dimensions():
    vector = torch.randn(5, 17)
    axis1 = torch.randn(5, 17)
    axis2 = torch.randn(5, 1) * axis1
    results1 = project_vector_onto_axis(vector, axis1)
    results2 = project_vector_onto_axis(vector, axis2)
    assert torch.all(torch.isclose(results1, results2))


def test_passing_in_the_square_axis_length_doesnt_change_results():
    vector = torch.randn(5, 3)
    axis = torch.randn(5, 3)
    square_length = axis.square().sum(dim=-1, keepdims=True)
    results1 = project_vector_onto_axis(vector, axis)
    results2 = project_vector_onto_axis(vector, axis, square_length)
    assert torch.equal(results1, results2)


def test_computes_gradients_for_vector():
    vector = torch.randn(5, 3).requires_grad_()
    axis = torch.randn(5, 3)
    results = project_vector_onto_axis(vector, axis)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert vector.grad is not None
    assert torch.all(torch.isfinite(vector.grad))
    assert torch.all(torch.greater(torch.abs(vector.grad), 0))


def test_computes_gradients_for_axis():
    vector = torch.randn(5, 3)
    axis = torch.randn(5, 3).requires_grad_()
    results = project_vector_onto_axis(vector, axis)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert axis.grad is not None
    assert torch.all(torch.isfinite(axis.grad))
    assert torch.all(torch.greater(torch.abs(axis.grad), 0))


def test_computes_gradients_for_axis_and_vector():
    vector = torch.randn(5, 3).requires_grad_()
    axis = torch.randn(5, 3).requires_grad_()
    results = project_vector_onto_axis(vector, axis)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert vector.grad is not None
    assert torch.all(torch.isfinite(vector.grad))
    assert torch.all(torch.greater(torch.abs(vector.grad), 0))
    assert axis.grad is not None
    assert torch.all(torch.isfinite(axis.grad))
    assert torch.all(torch.greater(torch.abs(axis.grad), 0))


def test_computes_gradients_when_axis_is_zero():
    vector = torch.randn(5, 3).requires_grad_()
    axis = torch.zeros(5, 3).requires_grad_()
    results = project_vector_onto_axis(vector, axis)
    assert results.requires_grad is True
    assert results.grad_fn is not None
    loss = results.square().sum()
    loss.backward()
    assert vector.grad is not None
    assert torch.all(torch.isfinite(vector.grad))
    assert torch.all(torch.greater_equal(torch.abs(vector.grad), 0))
    assert axis.grad is not None
    assert torch.all(torch.isfinite(axis.grad))
    assert torch.all(torch.greater_equal(torch.abs(axis.grad), 0))


def test_gradcheck_both_zeros():
    vectors = torch.zeros(3, dtype=torch.double, requires_grad=True)
    axes = torch.zeros(3, dtype=torch.double, requires_grad=True)
    assert gradcheck(project_vector_onto_axis, (vectors, axes), eps=1e-6, atol=1e-4)


def test_gradcheck_axis_zero():
    vectors = torch.randn(3, dtype=torch.double, requires_grad=True)
    axes = torch.zeros(3, dtype=torch.double, requires_grad=True)
    assert gradcheck(project_vector_onto_axis, (vectors, axes), eps=1e-6, atol=1e-4)


def test_gradcheck_vector_zeros():
    vectors = torch.zeros(3, dtype=torch.double, requires_grad=True)
    axes = torch.randn(3, dtype=torch.double, requires_grad=True)
    assert gradcheck(project_vector_onto_axis, (vectors, axes), eps=1e-6, atol=1e-4)


def test_gradcheck_vector_and_axes():
    vectors = torch.randn(100, 3, dtype=torch.double, requires_grad=True)
    axes = torch.randn(100, 3, dtype=torch.double, requires_grad=True)
    assert gradcheck(project_vector_onto_axis, (vectors, axes), eps=1e-6, atol=1e-4)


def test_gradcheck_all_zeros():
    vectors = torch.zeros(3, dtype=torch.double, requires_grad=True)
    axes = torch.zeros(3, dtype=torch.double, requires_grad=True)
    length = torch.zeros(1, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        project_vector_onto_axis, (vectors, axes, length), eps=1e-6, atol=1e-4
    )


def test_gradcheck_only_length_nonzer0():
    vectors = torch.zeros(3, dtype=torch.double, requires_grad=True)
    axes = torch.zeros(3, dtype=torch.double, requires_grad=True)
    length = torch.randn(1, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        project_vector_onto_axis, (vectors, axes, length), eps=1e-6, atol=1e-4
    )


def test_gradcheck_axis_zero_length_nonzero():
    vectors = torch.randn(3, dtype=torch.double, requires_grad=True)
    axes = torch.zeros(3, dtype=torch.double, requires_grad=True)
    length = torch.randn(1, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        project_vector_onto_axis, (vectors, axes, length), eps=1e-6, atol=1e-4
    )


def test_gradcheck_vector_zeros_with_length():
    vectors = torch.zeros(3, dtype=torch.double, requires_grad=True)
    axes = torch.randn(3, dtype=torch.double, requires_grad=True)
    length = torch.randn(1, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        project_vector_onto_axis, (vectors, axes, length), eps=1e-6, atol=1e-4
    )


def test_gradcheck_vector_and_length_zeros():
    vectors = torch.zeros(3, dtype=torch.double, requires_grad=True)
    axes = torch.randn(3, dtype=torch.double, requires_grad=True)
    length = torch.zeros(1, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        project_vector_onto_axis, (vectors, axes, length), eps=1e-6, atol=1e-4
    )


def test_gradcheck_length_zero():
    vectors = torch.randn(3, dtype=torch.double, requires_grad=True)
    axes = torch.randn(3, dtype=torch.double, requires_grad=True)
    length = torch.zeros(1, dtype=torch.double, requires_grad=True)
    assert gradcheck(
        project_vector_onto_axis, (vectors, axes, length), eps=1e-6, atol=1e-4
    )


def test_can_be_compiled():
    project_vector_onto_axis_c = torch.compile(project_vector_onto_axis)
    vectors = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    axes = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    results = project_vector_onto_axis_c(vectors, axes)
    assert results.shape == (7, 3)
    results_norm = results / torch.linalg.norm(results, dim=-1, keepdim=True)
    axes_norm = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    dot_product = (results_norm * axes_norm).sum(dim=-1)
    assert torch.all(torch.isclose(dot_product.abs(), torch.ones_like(dot_product)))
