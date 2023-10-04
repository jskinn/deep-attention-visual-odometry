from typing import Iterable, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as fn
from deep_attention_visual_odometry.types import PointsAndJacobian
from .camera_model import ICameraModel
from .least_squares_utils import find_residuals, find_error, find_error_gradient


class _StepResidualsGradients(NamedTuple):
    step: torch.Tensor
    error: torch.Tensor
    gradient: torch.Tensor


class BFGSCameraSolver(nn.Module):
    """
    Given a set of points across multiple views,
    jointly optimise for the 3D positions of the points, the camera intrinsics, and extrinsics.

    Based on
    "More accurate pinhole camera calibration with imperfect planar target" by
    Klaus H Strobl and Gerd Hirzinger in ICCV 2011

    Uses the Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS) to iteratively optimise the various parameters.
    There are two learned heuristics modifying the algorithm, one as a warp on the search direction,
    and the other when searching for the step size.
    Changing the search direction in particular lets the optimiser specialise to only optimising some of the parameters,
    and leaving the others to the prior.

    Why BFGS and not levenberg-marquard?
    Due to optimising the 3D positions of the points, the number of parameters and
    the size of the Jacobian matrix can get very large.
    The Levenberg-Marquardt algorithm requires either computing and inverting J^T J or solving the normal equations for
    Jp = r, both of which (according to "The Levenberg-Marquardt algorithm: implementation and theory" by Jorge More)
    are numerically unstable.
    BFGS allows us to avoid inverting the Jacobian, which is an O(n^3) operation.
    """

    def __init__(
        self,
        camera_model: ICameraModel,
        max_iterations: int,
        search_direction_heuristic: nn.Module | None = None,
        line_search_sufficient_decrease: float = 1e-4,
        line_search_curvature: float = 0.9,
        line_search_max_step_size: float = 16.0,
        line_search_zoom_iterations: int = 20,
    ):
        super().__init__()
        self.camera_model = camera_model
        self.search_direction_heuristic = search_direction_heuristic
        self.max_iterations = int(max_iterations)
        self.line_search_max_step_size = int(line_search_max_step_size)
        self.line_search_zoom_iterations = int(line_search_zoom_iterations)

        # line search parameters (c_1 and c_2).
        # since the parameters are bounded 0 < c_1 < c_2 < 1,
        self.line_search_sufficient_decrease = line_search_sufficient_decrease
        self.line_search_curvature = line_search_curvature

    def forward(
        self,
        feature_points: torch.Tensor,
        point_weights: torch.Tensor | None = None,
        spatial_priors: torch.Tensor | None = None,
        camera_priors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """

        :param feature_points: A (batch x )num_frames x num_points x 2 tensor of pixel coordinates.
                               The coordinates for each frame should all correspond to the same point.
        :param point_weights: Confidence values for each feature match, size (batch x )num_frames x num_points.
                              Weights the residuals during optimisation,
                              points that do not appear in a frame should have a near-zero weighting.
        :param spatial_priors: 3D locations priors for each point, potentially the output of a previous optimisation
        :param camera_priors: Priors on the camera parameters, potentially the output of a previous optimisation
        :return:
        """
        if point_weights is not None and point_weights.ndim < feature_points.ndim:
            point_weights = point_weights.unsqueeze(-1)
        if feature_points.ndim < 4:
            feature_points = feature_points.unsqueeze(0)
            if point_weights is not None:
                point_weights.unsqueeze(0)
        batch_size = feature_points.size(0)

        # Make the initial camera parameters
        # TODO: initialise this correctly
        parameters = None
        inverse_hessian = torch.zeros([])

        # Calculate the initial residuals, error, and gradient
        # Updates in further iterations will come from the line search
        points_and_jacobian = self.camera_model.forward_model_and_jacobian(
            parameters
        )
        residuals = find_residuals(points_and_jacobian.points, feature_points)
        error = find_error(residuals, point_weights)
        gradient = find_error_gradient(residuals, points_and_jacobian.jacobian, point_weights)

        updating = torch.ones(
            batch_size, dtype=torch.bool, device=feature_points.device
        )
        for step in range(self.max_iterations):
            # Compute a search direction as -H \delta f
            search_direction = -1 * torch.matmul(inverse_hessian, gradient)
            # Use a heuristic to adjust the search direction, to stabilise the search
            search_direction = self.search_direction_heuristic(search_direction, step)
            # Line search for an update step that satisfies the wolfe conditions
            step = self.line_search(residuals, search_direction)

    def line_search(
        self,
        parameters,
        search_direction: torch.Tensor,
        base_error: torch.Tensor,
        base_gradient: torch.Tensor,
        feature_points: torch.Tensor,
        point_weights: torch.Tensor | None = None
    ):
        """
        Line search for a step that satisfies the strong wolfe conditions.
        Implements algorithms 3.5 and 3.6 from Numerical Optimisation by Nocedal and Wright

        :param parameters:
        :param search_direction:
        :param initial_points_and_jacobian:
        :return:
        """
        batch_size = search_direction.size(0)
        lower_bounds = torch.zeros(batch_size, dtype=search_direction.dtype, device=search_direction.device)
        upper_bounds = torch.ones(batch_size, dtype=search_direction.dtype, device=search_direction.device)
        updating = torch.ones(batch_size, dtype=torch.bool, device=search_direction.device)
        zooming = torch.zeros(batch_size, dtype=torch.bool, device=search_direction.device)
        # Algorithm 3.5: Find a
        for _ in range(self.line_search_zoom_iterations)
            candidate_parameters = parameters + step_size * search_direction
            candidate_points_and_jacobian = self.camera_model.forward_model_and_jacobian(candidate_parameters)
            residuals = find_residuals()
        # Step 2: algorithm



