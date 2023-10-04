import torch
import torch.nn as nn


class SearchDirectionHeuristic(nn.Module):
    """
    Given a search direction, output a new search direction, potentially masking off certain dimensions.

    At the moment this is a simple MLP, but could potentially be expanded into
    a more comprehensive function approximation approach.
    Also, since it's called inside the optimisation loop, some kind of RNN
    or state might be appropriate.
    """

    def __init__(self, num_parameters: int, hidden_size: int = -1):
        super().__init__()
        if hidden_size < 0:
            hidden_size = 3 * num_parameters
        self.network = nn.Sequential(
            nn.Linear(num_parameters + 1, hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size, affine=True),
            nn.Linear(hidden_size, num_parameters),
            nn.Sigmoid(inplace=True),
        )

    def forward(
        self, search_direction: torch.Tensor, step_number: float | torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(step_number, torch.Tensor):
            step_number = torch.tensor(
                [step_number for _ in range(search_direction.size(0))],
                dtype=search_direction.dtype,
                device=search_direction.device,
            )
        x = torch.cat([search_direction, step_number], dim=-1)
        x = self.network(x)
        search_direction = search_direction * x
        return search_direction
