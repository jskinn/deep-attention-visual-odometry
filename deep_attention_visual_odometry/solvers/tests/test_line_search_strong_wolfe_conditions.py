import unittest
import torch
from deep_attention_visual_odometry.solvers.line_search_strong_wolfe_conditions import (
    LineSearchStrongWolfeConditions,
)
from deep_attention_visual_odometry.solvers.tests.example_function_distance_to_point import (
    ExampleFunctionDistanceToPoint,
)


class TestLineSearchStrongWolfeConditions(unittest.TestCase):
    def test_returns_instance_of_function_to_optimise_and_step_distance(self):
        target_points = torch.tensor([20.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([0.0, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)

        result, step = subject(function_to_optimise, search_direction)

        self.assertIsInstance(result, ExampleFunctionDistanceToPoint)
        self.assertNotEqual(result, function_to_optimise)
        self.assertTrue(torch.equal(step, result.parameters - initial_guess))

    def test_searches_in_desired_direction(self):
        target_points = torch.tensor([20.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([0.5, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)

        result, _ = subject(function_to_optimise, search_direction)

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

        result, _ = subject(function_to_optimise, search_direction)

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

        result, _ = subject(function_to_optimise, search_direction)

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

        result, _ = subject(function_to_optimise, search_direction)

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

        result, _ = subject(function_to_optimise, search_direction)

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

        result, _ = subject(function_to_optimise, search_direction)

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

        result, _ = subject(function_to_optimise, search_direction)

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

        result, _ = subject(function_to_optimise, search_direction)

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

        result, _ = subject(function_to_optimise, search_direction)

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

    def test_can_be_compiled(self):
        target_points = torch.tensor([2.0, 1.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([1.0, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)
        compiled_subject = torch.compile(subject)

        result, _ = compiled_subject(function_to_optimise, search_direction)

        self.assertEqual(result.parameters.shape, (1, 1, 2))
        self.assertEqual(result.parameters[0, 0, 0], 1.0)
        self.assertEqual(result.parameters[0, 0, 1], 1.0)

    def test_works_with_autograd_for_the_initial_guess(self):
        target_points = torch.tensor([20.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points, requires_grad=True)
        search_direction = torch.tensor([1.0, 1.0]).reshape(1, 1, 2)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)
        result, _ = subject(function_to_optimise, search_direction)
        result_error = result.get_error()
        self.assertIsNone(initial_guess.grad)

        result_error.sum().backward()

        self.assertIsNotNone(initial_guess.grad)
        self.assertEqual(initial_guess.grad.shape, (1, 1, 2))
        self.assertLess(initial_guess.grad[0, 0, 0], 0.0)
        self.assertLess(initial_guess.grad[0, 0, 1], 0.0)

    def test_works_with_autograd_for_the_search_direction(self):
        target_points = torch.tensor([20.0, 10.0]).reshape(1, 1, 2)
        initial_guess = torch.zeros_like(target_points)
        search_direction = torch.tensor([1.0, 1.0], requires_grad=True)
        function_to_optimise = ExampleFunctionDistanceToPoint(
            initial_guess, target_points
        )
        subject = LineSearchStrongWolfeConditions(max_step_size=10.0, zoom_iterations=5)
        result, _ = subject(function_to_optimise, search_direction.reshape(1, 1, 2))
        result_error = result.get_error()
        self.assertIsNone(search_direction.grad)

        result_error.sum().backward()

        self.assertIsNotNone(search_direction.grad)
        self.assertEqual(search_direction.grad.shape, (2,))
        self.assertLess(search_direction.grad[0], 0.0)
        self.assertLess(search_direction.grad[1], 0.0)
