from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of variables."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(float(v.data))  # changed this to cast to a float
            else:
                scalars.append(minitorch.scalar.Scalar(float(v)))
                raw_vals.append(float(v))  # changed this to cast to a float

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative is 1 for both inputs."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Log of a number."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of log."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiply two numbers."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Derivative is the other input."""
        a, b = ctx.saved_values
        # multiply derivative is a constant, since this is multivar, the constant(1) is multiplied by the other var
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Inverse of a number."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of inverse."""
        (a,) = ctx.saved_values
        # use the built in inv_back function we made to calculate the derivative
        return operators.inv_back(a, d_output)


# Need to basically return the derivates * d_output (this is the chain rule) in the backward func for every class(mathematical function) which would be the forward


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negation of a number."""
        # dont need to save context as we are not using it in the backward function
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of negation."""
        # derivative of -x, is -1
        return -1 * d_output


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Exponential of a number."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of exponential."""
        (a,) = ctx.saved_values
        # derivatives of e^x is e^x
        return operators.exp(a) * d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1.0 / (1.0 + e^{-x})$ if x >=0 else $e^x / (1.0 + e^{x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Sigmoid of a number."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of sigmoid."""
        (a,) = ctx.saved_values
        # derivatives of sigmoid is sigmoid * (1 - sigmoid)
        return (operators.sigmoid(a) * (1 - operators.sigmoid(a))) * d_output


class ReLU(ScalarFunction):
    """Rectified linear unit"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """ReLU of a number."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of ReLU."""
        (a,) = ctx.saved_values
        # Use the relu_back function we made to calculate the derivative
        return operators.relu_back(a, d_output)


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Equality of two numbers."""
        # no need to save context as we are not using it in the backward function
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Derivative of equality."""
        # derivative of constant is 0
        return 0.0, 0.0


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Less than of two numbers."""
        # no need to save context as we are not using it in the backward function
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Derivative of less than."""
        # derivative of constant is 0
        return 0.0, 0.0
