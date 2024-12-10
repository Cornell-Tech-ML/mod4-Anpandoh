from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    x = list(vals)
    x[arg] += epsilon
    f_plus = f(*x)
    x[arg] -= 2 * epsilon
    f_minus = f(*x)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...  # noqa: D102

    """Add `val` to the the derivative accumulated on this variable."""

    @property
    def unique_id(self) -> int: ...  # noqa: D102

    """Unique identifier for the variable. """

    def is_leaf(self) -> bool: ...  # noqa: D102

    """True if this variable created by the user (no `last_fn`)"""

    def is_constant(self) -> bool: ...  # noqa: D102

    """True if this variable is a constant (no derivative)"""

    @property
    def parents(self) -> Iterable["Variable"]: ...  # noqa: D102

    """Returns the parents of this variable."""

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...  # noqa: D102

    """Returns the parents of this variable."""


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def visit(node: Variable) -> None:
        if node.unique_id not in visited and not node.is_constant():
            visited.add(node.unique_id)
            for parent in node.parents:
                visit(parent)
            order.append(node)

    visit(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:  # noqa: D417
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # sort
    sorted_variables = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    # go through the graph
    for var in sorted_variables:
        var_id = var.unique_id
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var_id])
        else:
            # print(derivatives[var_id])
            for parent, parent_deriv in var.chain_rule(derivatives[var_id]):
                parent_id = parent.unique_id
                # add var: deriv to the dictionary
                if parent_id in derivatives:
                    derivatives[parent_id] += parent_deriv
                else:
                    derivatives[parent_id] = parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values."""
        return self.saved_values
