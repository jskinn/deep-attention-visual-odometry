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
import torch


def square_error(x: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
    """
    The sphere function,
    Global minima is at 0.
    :param x:
    :param _:
    :return: sum x_{i}^2
    """
    return x.square().sum(dim=-1)


def log_square_error(x: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
    """
    Log of the square error, log(1.0 + sum(x^2)).
    This has vanishing gradients for large x, which can make gradient descent slow.
    Has global minima

    :param x:
    :param _:
    :return:
    """
    return (x.square().sum(dim=-1) + 1.0).log()


def rosenbrock_function(params: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
    """
    The rosenbrock function, with a = 1 and b = 100.
    The global minima is at [1, 1]
    See https://en.wikipedia.org/wiki/Rosenbrock_function
    :param params: (Bx..)x2 vector x and y
    :param _:
    :return: (a - x)^2 + b(y - x^2)^2
    """
    x = params[..., 0]
    y = params[..., 1]
    return (1.0 - x).square() + 100.0 * (y - x.square()).square()


def rastrigrin_function(x: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
    """
    The rastrigin function, with many local minima.
    Has global minima at 0.
    See https://en.wikipedia.org/wiki/Rastrigin_function
    :param x: A vector of dimension n
    :param _:
    :return: An + sum_i(x_i^2 - A cos(2 pi x_i)
    """
    a = 10.0
    n = x.size(-1)
    return a * n + (x.square() - a * (2 * torch.pi * x).cos()).sum(dim=-1)


def beale_function(params: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
    """
    The beale function, with unusual contours.
    Global minima is at 3, 0.5.
    See https://en.wikipedia.org/wiki/Test_functions_for_optimization

    :param params: A 2-vector x and y
    :param _:
    :return: (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    """
    x = params[..., 0]
    y = params[..., 1]
    xy = x * y
    xyy = xy * y
    xyyy = xyy * y
    return (
        (1.5 - x + xy).square()
        + (2.25 - x + xyy).square()
        + (2.625 - x + xyyy).square()
    )


def bukin_function_6(params: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
    """
    Burkin function number 6, a function of two variables
    Global minima at -10, 1
    See https://en.wikipedia.org/wiki/Test_functions_for_optimization

    :param params: A two-vector, x and y
    :param _:
    :return: 100 * sqrt(abs(y - 0.01 * x^2)) + 0.01 * abs(x + 10)
    """
    x = params[..., 0]
    y = params[..., 1]
    return 100.0 * (y - 0.01 * x.square()).abs().sqrt() + 0.01 * (x + 10.0).abs()


def easom_function(params: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
    """
    The easom function, of two variables. This function has vanishing gradients
    away from the minima, and extremely high gradients approaching the minima.
    Global minima at pi,pi
    See https://en.wikipedia.org/wiki/Test_functions_for_optimization

    :param params: A 2-vector x and y
    :param _:
    :return: - cos(x) cos(y) e^(-(x - pi)^2 - (y - pi)^2)
    """
    x = params[..., 0]
    y = params[..., 1]
    return (
        -1.0
        * x.cos()
        * y.cos()
        * (-(x - torch.pi).square() - (y - torch.pi).square()).exp()
    )
