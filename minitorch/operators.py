"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """Multiply x and y"""
    return x * y


# - id
def id(x: float) -> float:
    """Identity function"""
    return x


# - add
def add(x: float, y: float) -> float:
    """Add x and y"""
    return x + y


# - neg
def neg(x: float) -> float:
    """Negate x"""
    return -x


# - lt
def lt(x: float, y: float) -> bool:
    """Check if x is less than y"""
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    """Check if x is equal to y"""
    return x == y


# - max
def max(x: float, y: float) -> float:
    """Return the maximum of x and y"""
    return x if x > y else y


# - is_close
def is_close(x: float, y: float) -> bool:
    """Check if x is close to y"""
    return abs(x - y) < 1e-2


# - sigmoid
def sigmoid(x: float) -> float:
    """Sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """Rectified linear unit"""
    return x if x > 0 else 0.0


# - log
def log(x: float) -> float:
    """Natural logarithm"""
    return math.log(x)


# - exp
def exp(x: float) -> float:
    """Exponential function"""
    return math.exp(x)


# - inv
def inv(x: float) -> float:
    """Inverse function"""
    return 1.0 / x


# - log_back
def log_back(x: float, y: float) -> float:
    """Natural logarithm backpropagation, derivative of log * y, which reduces to y / x"""
    return y / x


# - inv_back
def inv_back(x: float, y: float) -> float:
    """Inverse backpropagation, derivative of (1 / x) * y, which reduces to -y / x^2"""
    return -y / (x**2)


# - relu_back
def relu_back(x: float, y: float) -> float:
    """Rectified linear unit backpropagation, derivative of relu * y, which reduces to y if x > 0 else 0"""
    return y if x > 0 else 0.0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn: Callable[[float], float], a: Iterable[float]) -> Iterable[float]:
    """Map a function over an iterable"""
    return [fn(x) for x in a]


# - zipWith
def zipWith(
    fn: Callable[[float, float], float], a: Iterable[float], b: Iterable[float]
) -> Iterable[float]:
    """Map a function over two iterables"""
    return [fn(x, y) for x, y in zip(a, b)]


# - reduce
def reduce(
    fn: Callable[[float, float], float], a: Iterable[float], init: float
) -> float:
    """Reduce an iterable with a function"""
    acc = init
    for x in a:
        acc = fn(acc, x)
    return acc


#
# Use these to implement
# - negList : negate a list
def negList(a: Iterable[float]) -> Iterable[float]:
    """Negate a list"""
    return map(neg, a)


# - addLists : add two lists together
def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add two lists together"""
    return zipWith(add, a, b)


# - sum: sum lists
def sum(a: Iterable[float]) -> float:
    """Sum a list"""
    return reduce(add, a, 0.0)


def prod(a: Iterable[float]) -> float:
    """Multiply a list"""
    return reduce(mul, a, 1.0)
