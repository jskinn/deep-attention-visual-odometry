import torch
import torch.nn as nn
from .i_optimisable_function import IOptimisableFunction


class TransformerSolver(nn.Module):
    """
    Based on the BFGS algorithm, optimise a solution in a fixed number of steps.
    Each step picks a search direction for each estimate, evaluates the functions in those directions,
    then chooses how far to step in each direction. Scaling the step allows a given step to be
    "turned off" by setting the scale to zero.
    - Steps and scales are chosen by MLPs
    - There is only one estimate at a time.
    """

    def __init__(
        self,
        num_steps: int,
        num_parameters: int,
        search_direction_hidden_dim: int,
        line_search_hidden_dim: int,
    ):
        super().__init__()
        self.solver_steps = nn.Sequential(
            *(
                MLPSolverStep(
                    num_parameters, search_direction_hidden_dim, line_search_hidden_dim
                )
                for _ in range(num_steps)
            )
        )

    def forward(
        self,
        function: IOptimisableFunction,
    ) -> IOptimisableFunction:
        result, history = self.solver_steps(function)
        return result


class RecurrentTransformerSolver(nn.Module):
    """
    A variant of the transformer solver using the same set of weights for each step.
    """

    def __init__(
        self,
        num_steps: int,
        num_parameters: int,
        search_direction_hidden_dim: int,
        line_search_hidden_dim: int,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.solver_step = MLPSolverStep(
            num_parameters,
            search_direction_hidden_dim,
            line_search_hidden_dim,
        )

    def forward(
        self,
        function: IOptimisableFunction,
    ) -> IOptimisableFunction:
        result = function
        history = None
        for step_idx in range(self.num_steps):
            result, history = self.solver_steps(function, history)
        return result


class MLPSolverStep(nn.Module):
    """
    A single iteration of a solver step in a module, based on an iteration of the BFGS algorithm.
    The way BFGS steps is by choosing a search direction based in the gradient,
    and then line-searching in that direction for a point that satisfies the wolffe conditions.
    This module mimics that, but replaces all the calculations with transformers:
    1. Choose a search direction based on the current parameters and the error
    2. Evaluate the function in the search direction
    3. Pick a scaling for the search direction based on the current error and the error in that direction
    4. Return a scaled step in the search direction.
    The scale to the step is fed through GELU, so that a given step can be turned off
    """

    def __init__(
        self, num_parameters: int, search_direction_hidden: int, line_search_hidden: int
    ):
        super().__init__()
        self.search_direction_network = nn.Sequential(
            nn.Linear(num_parameters + 2, search_direction_hidden),
            nn.GELU(),
            nn.BatchNorm1d(search_direction_hidden),
            nn.Linear(search_direction_hidden, num_parameters),
        )
        self.line_search_network = nn.Sequential(
            nn.Linear(2 * num_parameters + 4, line_search_hidden),
            nn.GELU(),
            nn.BatchNorm1d(line_search_hidden),
            nn.Linear(line_search_hidden, 1),
            nn.GELU(),
        )

    def forward(self, function: IOptimisableFunction) -> IOptimisableFunction:
        num_estimates = function.num_estimates
        function_parameters = function.as_parameters_vector()
        # Normalise the parameter values, these will be used to rescale the function later
        parameters_mean = function_parameters.mean(dim=1, keepdim=True)
        parameters_std = function_parameters.std(dim=1, keepdim=True).clamp(min=1e-8)
        function_parameters = (function_parameters - parameters_mean) / parameters_std
        # Build a block of vectors encompassing all the history vectors
        error = function.get_error()
        inv_error = 1.0 / error.clamp(min=1e-8)
        inputs = torch.cat([function_parameters, error, inv_error], dim=-1)
        # Transform the history to produce a search direction
        search_direction = self.search_direction_network(inputs)
        search_direction = search_direction[:, 0:num_estimates, :]
        search_direction = search_direction * parameters_std + parameters_mean
        # Evaluate the function in the search direction
        candidate = function.add(search_direction)
        function_parameters = candidate.as_parameters_vector()
        parameters_mean = function_parameters.mean(dim=1, keepdim=True)
        parameters_std = function_parameters.std(dim=1, keepdim=True).clamp(min=1e-8)
        function_parameters = (function_parameters - parameters_mean) / parameters_std
        error = function.get_error()
        inv_error = 1.0 / error.clamp(min=1e-8)
        candidate_inputs = torch.cat([function_parameters, error, inv_error], dim=-1)
        # Transform the current and candidate parameters and error to work out how far to go in the search direction
        step_sizes = self.line_search_network(
            torch.cat([inputs, candidate_inputs], dim=1)
        )
        step_sizes = step_sizes[:, 0:num_estimates, :]
        # Return the step direction scaled by the results of the second network
        search_direction = step_sizes * search_direction
        candidate = function.add(search_direction)
        return candidate
