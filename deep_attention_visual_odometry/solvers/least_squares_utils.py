import torch


def find_residuals(
    estimated_points: torch.Tensor, true_points: torch.Tensor
) -> torch.Tensor:
    """
    Get the residuals for each point
    :param estimated_points:
    :param true_points:
    :return:
    """
    return estimated_points - true_points


def find_error(
    residuals: torch.Tensor, point_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Given some residuals, find the total error
    :param residuals: Residuals, shape BxFxNx2
    :param point_weights: Optional weights for each point, shape BxFxNx1
    :return: The total error, size B
    """
    error = residuals.square()
    if point_weights is not None:
        error = point_weights * error
    return torch.sum(error, tuple(range(1, error.ndim)))


def find_error_gradient(
    residuals: torch.Tensor,
    jacobian: torch.Tensor,
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given a set of residuals and a jacobian, find the gradient of the error w.r.t. each parameter.
    :param residuals: Residuals, shape BxFxNx2
    :param jacobian: Partial derivatives, shape BxFxNx2xP
    :param point_weights: Weights for each residual, shape BxFxNx1, or None
    :return: Gradients for each parameter, shape BxP
    """
    gradient = 2.0 * residuals
    if point_weights is not None:
        gradient = point_weights * gradient
    gradient = gradient.unsqueeze(-1)
    gradient = gradient * jacobian
    return torch.sum(gradient, tuple(range(1, gradient.ndim - 1)))
