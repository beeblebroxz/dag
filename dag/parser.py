"""
AST-based dependency detection for cell functions.

This module parses cell function source code to detect dependencies
at decoration time (parse time), rather than at runtime.

The parser looks for patterns like:
- self.MethodName() - direct cell function calls
- self.A().B().C() - chained cell function calls
- self.db['key'].Method() - indirection patterns

Per the spec, the parser greedily assumes that function invocations
return cell objects, which is a safe overapproximation.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Optional, Set, Tuple


@dataclass
class Dependency:
    """
    A dependency found in a cell function.

    Attributes:
        name: The method name being called (e.g., 'Strike', 'Price')
        chain: The full call chain (e.g., ['PairObject', 'Spot'] for self.PairObject().Spot())
        has_args: Whether the call had arguments (for parameterized cells)
        is_indirect: Whether this is via indirection (self.db['key'].Method())
    """
    name: str
    chain: Tuple[str, ...] = ()
    has_args: bool = False
    is_indirect: bool = False


class DependencyVisitor(ast.NodeVisitor):
    """
    AST visitor that extracts cell function dependencies.

    Looks for patterns:
    1. self.X() - simple call
    2. self.X().Y() - chained call (both X and Y are dependencies)
    3. self.X().Y().Z() - longer chains
    4. self.db['key'].X() - indirection
    5. self.X(*args) - calls with arguments (still a dependency)
    """

    def __init__(self, self_name: str = "self"):
        self.self_name = self_name
        self.dependencies: Set[str] = set()
        self.dependency_details: List[Dependency] = []

    def _is_self(self, node: ast.AST) -> bool:
        """Check if a node is the 'self' variable."""
        return isinstance(node, ast.Name) and node.id == self.self_name

    def _extract_call_chain(self, node: ast.AST) -> Optional[List[str]]:
        """
        Extract a chain of attribute accesses from a node.

        Returns a list of attribute names if this is a self.X.Y.Z chain,
        or None if it doesn't match the pattern.
        """
        chain = []

        while True:
            if isinstance(node, ast.Call):
                # Unwrap the call to get the function being called
                node = node.func
            elif isinstance(node, ast.Attribute):
                chain.append(node.attr)
                node = node.value
            elif isinstance(node, ast.Subscript):
                # self.db['key'] pattern - mark as indirect but continue
                node = node.value
            elif self._is_self(node):
                chain.reverse()
                return chain
            else:
                return None

    def visit_Call(self, node: ast.Call) -> None:
        """
        Visit a function call node.

        We're interested in calls of the form:
        - self.X()
        - self.X().Y()
        - etc.
        """
        chain = self._extract_call_chain(node)

        if chain:
            # We have a self.X... chain
            # The last element in the chain is the method being called
            for i, method_name in enumerate(chain):
                self.dependencies.add(method_name)
                self.dependency_details.append(Dependency(
                    name=method_name,
                    chain=tuple(chain[:i+1]),
                    has_args=bool(node.args or node.keywords) if i == len(chain) - 1 else False,
                ))

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Visit an attribute access node.

        We need to handle cases where attributes are accessed but not called,
        e.g., in `x = self.Strike` (property-style access).
        """
        # Only record if it's directly on self (not a call)
        if self._is_self(node.value):
            # This could be a property access - we'll track it but note it's not a call
            # For now, we only track actual calls, so we skip this
            pass

        self.generic_visit(node)


def parse_dependencies(func: Callable) -> FrozenSet[str]:
    """
    Parse a cell function to extract its dependencies.

    Args:
        func: The cell function to parse

    Returns:
        A frozenset of dependency method names
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        # Can't get source (e.g., built-in function)
        return frozenset()

    # Dedent the source in case it's a method defined inside a class
    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse the source
        return frozenset()

    # Find the function definition
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            func_def = node
            break

    if func_def is None:
        return frozenset()

    # Get the 'self' parameter name (usually 'self' but could be different)
    self_name = "self"
    if func_def.args.args:
        self_name = func_def.args.args[0].arg

    # Visit the AST
    visitor = DependencyVisitor(self_name)
    visitor.visit(func_def)

    return frozenset(visitor.dependencies)


def parse_dependencies_detailed(func: Callable) -> List[Dependency]:
    """
    Parse a cell function to extract detailed dependency information.

    Args:
        func: The cell function to parse

    Returns:
        A list of Dependency objects with full details
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return []

    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            func_def = node
            break

    if func_def is None:
        return []

    self_name = "self"
    if func_def.args.args:
        self_name = func_def.args.args[0].arg

    visitor = DependencyVisitor(self_name)
    visitor.visit(func_def)

    return visitor.dependency_details


def get_function_parameters(func: Callable) -> List[str]:
    """
    Get the parameter names of a function (excluding self).

    Used to determine if a cell function takes arguments,
    which affects how nodes are keyed.
    """
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        # Remove 'self' if present
        if params and params[0] in ('self', 'cls'):
            params = params[1:]
        return params
    except (ValueError, TypeError):
        return []
