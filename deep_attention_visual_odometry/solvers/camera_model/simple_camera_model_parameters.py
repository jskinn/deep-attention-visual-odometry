from typing import Final
import torch


class SimpleCameraModelParameters:
    # Parameter indices in order
    CX: Final[int] = 0
    CY: Final[int] = 1
    F: Final[int] = 2
    A1: Final[int] = 3
    A2: Final[int] = 4
    A3: Final[int] = 5
    B1: Final[int] = 6
    B2: Final[int] = 7
    B3: Final[int] = 8
    TX: Final[int] = 9
    TY: Final[int] = 10
    TZ: Final[int] = 11

    def __init__(self, data: torch.Tensor):
        self.data = data

    @property
    def cx(self) -> torch.Tensor:
        return self.data[self.CX]


def make_camera_parameters(
    cx: float,
    cy: float,
    f: float,
    a1: float,
    a2: float,
    a3: float,
    b1: float,
    b2: float,
    b3: float,
    tx: float,
    ty: float,
    tz: float,
) -> torch.Tensor:
    return torch.tensor([cx, cy, f, a1, a2, a3, b1, b2, b3, tx, ty, tz])


def stack_camera_parameters(
    cx: torch.Tensor,
    cy: torch.Tensor,
    f: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    tz: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    return torch.stack([cx, cy, f, a1, a2, a3, b1, b2, b3, tx, ty, tz], dim=dim)
