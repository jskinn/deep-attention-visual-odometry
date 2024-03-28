import torch
import torch.nn as nn
from .i_optimisable_function import IOptimisableFunction


class TransformerSolver(nn.Module):
    """
    Based on the BFGS algorithm, optimise a solution in a fixed number of steps.
    Each step picks a search direction for each estimate, evaluates the functions in those directions,
    then chooses how far to step in each direction. Scaling the step allows a given step to be
    "turned off" by setting the scale to zero.
    - Each step has its own set of weights.
    - It is expected that multiple estimates are used, forming the sequence dimension of the transformer.
    - Over each step, we accumulate the history of different estimates, lengthening the sequence.
      For E estimates and S steps, the final step has a sequence lenth of SxE.
    """

    def __init__(
        self,
        num_steps: int,
        num_parameters: int,
        embed_dim: int,
        search_direction_layers: int,
        line_search_layers: int,
    ):
        super().__init__()
        self.solver_steps = nn.Sequential(
            *(
                SolverStep(
                    num_parameters,
                    embed_dim,
                    search_direction_layers,
                    line_search_layers,
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
        embed_dim: int,
        search_direction_layers: int,
        line_search_layers: int,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.solver_step = SolverStep(
            num_parameters,
            embed_dim,
            search_direction_layers,
            line_search_layers,
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


class SolverStep(nn.Module):
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
        self,
        num_parameters: int,
        embed_dim: int,
        search_direction_layers: int = 6,
        line_search_layers: int = 6,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True
        )
        self.search_direction_network = nn.Sequential(
            nn.Linear(num_parameters + 2, embed_dim),
            nn.TransformerEncoder(encoder_layer, num_layers=search_direction_layers),
            nn.Linear(embed_dim, num_parameters),
        )
        self.line_search_network = nn.Sequential(
            nn.Linear(num_parameters + 2, embed_dim),
            nn.TransformerEncoder(encoder_layer, num_layers=line_search_layers),
            nn.Linear(embed_dim, 1),
            nn.GELU(),
        )

    def forward(
        self,
        function: IOptimisableFunction
        | tuple[IOptimisableFunction, torch.Tensor | None],
        history: torch.Tensor = None,
    ) -> tuple[IOptimisableFunction, torch.Tensor]:
        if isinstance(function, tuple):
            function, history = function
        num_estimates = function.num_estimates
        function_parameters = function.as_parameters_vector()
        # Normalise the parameter values, these will be used to
        parameters_mean = function_parameters.mean(dim=1, keepdim=True)
        parameters_std = function_parameters.std(dim=1, keepdim=True).clamp(min=1e-8)
        function_parameters = (function_parameters - parameters_mean) / parameters_std
        # Build a block of vectors encompassing all the history vectors
        if history is None:
            history = torch.empty(
                (function.batch_size, 0, function.num_parameters + 2),
                device=function_parameters.device,
                dtype=function_parameters.dtype,
            )
        error = function.get_error()
        inv_error = 1.0 / error.clamp(min=1e-8)
        inputs = torch.cat([function_parameters, error, inv_error], dim=-1)
        history = torch.cat([history, inputs], dim=1)
        # Transform the history to produce a search direction
        search_direction = self.search_direction_network(history)
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
        return candidate, history
