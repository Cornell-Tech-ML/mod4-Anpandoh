from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw

    tile_out = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tile_out = tile_out.permute(0, 1, 2, 4, 3, 5).contiguous()
    return (
        tile_out.view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply a 2D average pooling to the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width thats pooled

    """
    batch, channel, height, width = input.shape
    tile_out, new_height, new_width = tile(input, kernel)
    return tile_out.mean(4).view(batch, channel, new_height, new_width)


max_reduction = FastOps.reduce(operators.max, float("-inf"))


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute the max of a tensor along a dimension

        Args:
        ----
            ctx : Context
            input : input tensor
            dim : dimension to compute the max

        Returns:
        -------
            Tensor of size batch x channel x new_height x new_width thats pooled

        """
        dims = dim.to_numpy().astype(int).tolist()
        max_tensor = max_reduction(input, dims[0])
        ctx.save_for_backward(input, max_tensor)
        return max_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the max function"""
        input, maxTensor = ctx.saved_values
        return (input == maxTensor) * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction

    Args:
    ----
        input : input tensor
        dim : dimension to compute the max

    Returns:
    -------
        Apply max to the input tensor

    """
    return Max.apply(input, input._ensure_tensor(dim))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input : input tensor
        dim : dimension to compute the max

    Returns:
    -------
        Tensor with 1 at the max value and 0 elsewhere

    """
    maxReduce = FastOps.reduce(operators.max, float("-inf"))
    return maxReduce(input, dim) == input


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input : input tensor
        dim : dimension to compute the softmax

    Returns:
    -------
        Tensor with the softmax applied

    """
    exp = input.exp()
    return exp / exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input : input tensor
        dim : dimension to compute the log softmax

    Returns:
    -------
        Tensor with the log softmax applied

    """
    max_val = max(input, dim)
    shift = input - max_val
    exp = shift.exp()
    sum = exp.sum(dim)
    log = sum.log()
    return shift - log


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply a 2D max pooling to the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width thats pooled

    """
    batch, channel, height, width = input.shape
    tile_out, new_height, new_width = tile(input, kernel)
    return max(tile_out, 4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: input tensor
        p: probability of dropout
        ignore: ignore the dropout

    Returns:
    -------
        Tensor with the dropout applied

    """
    if ignore:
        return input
    return input * (rand(input.shape) > p)
