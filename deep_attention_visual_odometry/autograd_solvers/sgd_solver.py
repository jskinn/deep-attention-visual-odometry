# Copyright (C) 2024  John Skinner
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA
from typing import Callable
import torch
import torch.autograd
from torch.nn import Module


class SGDSolver(Module):
    def __init__(self, learning_rate: float, iterations: int):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.iterations = int(iterations)

    def forward(
        self,
        parameters: torch.Tensor,
        error_function: Callable[[torch.Tensor], torch.Tensor],
    ):
        create_graph = parameters.requires_grad
        for idx in range(self.iterations):
            if not parameters.requires_grad:
                parameters.requires_grad_(True)
            with torch.enable_grad():
                error = error_function(parameters).sum()
            gradient = torch.autograd.grad(error, parameters, create_graph=create_graph)
            parameters = parameters - self.learning_rate * gradient[0]
        if not create_graph:
            parameters = parameters.detach()
        return parameters
