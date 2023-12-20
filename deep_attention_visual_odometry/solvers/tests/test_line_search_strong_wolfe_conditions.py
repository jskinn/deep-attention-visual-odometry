from typing import Self
import unittest
import torch
from deep_attention_visual_odometry.solvers.i_optimisable_function import (
    IOptimisableFunction,
)
from deep_attention_visual_odometry.solvers.line_search_strong_wolfe_conditions import (
    LineSearchStrongWolfeConditions,
)


class ExampleFunctionDistanceToPoint(IOptimisableFunction):
    def __init__(self, parameters: torch.Tensor, target_points: torch.Tensor):
        self.parameters = parameters
        self.target_points = target_points

    @property
    def batch_size(self) -> int:
        return self.parameters.size(0)

    @property
    def num_estimates(self) -> int:
        return self.parameters.size(1)

    @property
    def num_parameters(self) -> int:
        return self.parameters.size(2)

    @property
    def device(self) -> torch.device:
        return self.parameters.device

    def get_error(self) -> torch.Tensor:
        """Get the error of the function at the current point. BxE"""
        return (self.parameters - self.target_points).square().sum(dim=2)

    def get_gradient(self) -> torch.Tensor:
        """Get the gradient of the function w.r.t. each of the parameters. BxExP"""
        return 2.0 * (self.parameters - self.target_points)

    def add(self, parameters: torch.Tensor) -> Self:
        """Return a new instance of this function at a new set of parameters.
        Used for the optimisation step.
        Input tensor should be BxExN, the same shape as the output of 'get_gradient'
        """
        return ExampleFunctionDistanceToPoint(
            parameters=self.parameters + parameters, target_points=self.target_points
        )

    def masked_update(self, other: Self, mask: torch.Tensor) -> Self:
        """
        The operation we need to be able to do is to keep values from the current parameters where false,
        and where true set them to the values from another instance, plus a delta.

        This might be able to be expressed as two functions, add(..) and then update(...) to merge.

        We want to retain already-computed gradients and errors as much as possible.
        """
        mask = mask[:, :, None].tile(1, 1, self.num_parameters)
        new_parameters = torch.where(mask, other.parameters, self.parameters)
        return ExampleFunctionDistanceToPoint(
            parameters=new_parameters, target_points=self.target_points
        )


class TestLineSearchStrongWolfeConditions(unittest.TestCase):
    def test_returns_instance_of_function_to_optimise(self):
        target_points = torch.tensor([20.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([0.0, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)

        result = subject(function_to_optimise, search_direction)

        self.assertIsInstance(result, ExampleFunctionDistanceToPoint)
        self.assertNotEqual(result, function_to_optimise)

    def test_searches_in_desired_direction(self):
        target_points = torch.tensor([20.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([0.5, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)

        result = subject(function_to_optimise, search_direction)

        self.assertEqual(result.parameters.shape, (1, 1, 2))
        self.assertGreaterEqual(result.parameters[0, 0, 0], 0.5)
        self.assertGreaterEqual(result.parameters[0, 0, 1], 1.0)
        result_direction = result.parameters / search_direction
        self.assertEqual(result_direction[0, 0, 0], result_direction[0, 0, 1])

    def test_reduces_error(self):
        target_points = torch.tensor([20.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([1.0, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        initial_error = function_to_optimise.get_error()
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)

        result = subject(function_to_optimise, search_direction)

        self.assertTrue(torch.all(torch.less(result.get_error(), initial_error)))

    def test_chosen_point_satisfies_strong_wolffe_conditions(self):
        c1 = 0.1
        c2 = 0.6
        target_points = torch.tensor([20.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([1.0, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        initial_error = function_to_optimise.get_error()
        initial_gradient = function_to_optimise.get_gradient()
        gradient_in_direction = (
            (initial_gradient * search_direction).sum(dim=-1).squeeze()
        )
        subject = LineSearchStrongWolfeConditions(
            max_step_size=10.0, zoom_iterations=5, sufficient_decrease=c1, curvature=c2
        )

        result = subject(function_to_optimise, search_direction)

        result_alpha = (result.parameters / search_direction).mean()
        result_error = result.get_error().squeeze()
        sufficient_decrease = (
            initial_error + c1 * result_alpha * gradient_in_direction
        ).squeeze()
        self.assertLessEqual(result_error, sufficient_decrease)
        result_gradient = result.get_gradient()
        result_gradient_in_direction = (
            (result_gradient * search_direction).sum(dim=-1).squeeze()
        )
        curvature_condition = c2 * gradient_in_direction.abs()
        self.assertLessEqual(result_gradient_in_direction.abs(), curvature_condition)

    def test_scales_down_too_large_step_size(self):
        target_points = torch.tensor([1.0, 1.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([10.0, 0.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)

        result = subject(function_to_optimise, search_direction)

        self.assertEqual(result.parameters.shape, (1, 1, 2))
        self.assertLess(result.parameters[0, 0, 0], 10.0)
        self.assertEqual(result.parameters[0, 0, 1], 0.0)

    def test_widens_for_too_small_search_direction(self):
        target_points = torch.tensor([10.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([0.2, 0.1]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(
            max_step_size=128.0, zoom_iterations=5
        )

        result = subject(function_to_optimise, search_direction)

        self.assertEqual(result.parameters.shape, (1, 1, 2))
        self.assertGreater(result.parameters[0, 0, 0], 0.2)
        self.assertGreater(result.parameters[0, 0, 1], 0.1)

    def test_does_not_exceed_step_size_times_search_direction(self):
        target_points = torch.tensor([10.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([0.2, 0.1]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=2.0, zoom_iterations=5)

        result = subject(function_to_optimise, search_direction)

        self.assertEqual(result.parameters.shape, (1, 1, 2))
        self.assertEqual(result.parameters[0, 0, 0], 0.4)
        self.assertEqual(result.parameters[0, 0, 1], 0.2)

    def test_finds_easy_solution(self):
        target_points = torch.tensor([0.5, 0.5]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([1.0, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)

        result = subject(function_to_optimise, search_direction)

        self.assertEqual(result.parameters.shape, (1, 1, 2))
        self.assertEqual(result.parameters[0, 0, 0], 0.5)
        self.assertEqual(result.parameters[0, 0, 1], 0.5)

    def test_produces_small_change_when_search_direction_is_wrong(self):
        zoom_iterations = 5
        target_points = torch.tensor([-9.7, 2.2]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([1.0, 0.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(
            max_step_size=10.0, zoom_iterations=zoom_iterations
        )

        result = subject(function_to_optimise, search_direction)

        self.assertEqual(result.parameters.shape, (1, 1, 2))
        self.assertEqual(result.parameters[0, 0, 0], 1 / (2**zoom_iterations))
        self.assertEqual(result.parameters[0, 0, 1], 0.0)

    def test_handles_multiple_estimates_and_batch(self):
        c1 = 1e-4
        c2 = 0.9
        target_points = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[4.3, -10], [-9.7, 2.2], [2.8, -8.2]],
            ]
        )
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor(
            [
                [[0.1, 1.0], [10.0, -0.1], [0.1, 0.1]],
                [[1.0, -1.0], [-1.0, -0.1], [2.0, 0.1]],
            ]
        )
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        initial_error = function_to_optimise.get_error()
        initial_gradient = function_to_optimise.get_gradient()
        gradient_in_direction = (initial_gradient * search_direction).sum(dim=-1)
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)

        result = subject(function_to_optimise, search_direction)

        result_alpha = (result.parameters / search_direction).mean(dim=-1)
        result_error = result.get_error()
        sufficient_decrease = initial_error + c1 * result_alpha * gradient_in_direction
        self.assertTrue(torch.all(torch.less_equal(result_error, sufficient_decrease)))
        result_gradient = result.get_gradient()
        result_gradient_in_direction = (result_gradient * search_direction).sum(dim=-1)
        curvature_condition = c2 * gradient_in_direction.abs()
        self.assertTrue(
            torch.all(
                torch.less_equal(
                    result_gradient_in_direction.abs(), curvature_condition
                )
            )
        )
