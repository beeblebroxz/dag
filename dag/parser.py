"""AST-based dependency detection for computed functions.

The parser performs conservative, local data-flow analysis. It starts from
known roots (``self`` and function arguments), follows attribute access,
calls, subscripts, aliases, loops, and comprehensions, and records calls that
may target computed functions. Runtime tracking remains authoritative; this
module supplies the declarations used to catch unresolved same-object calls.

In Pyodide, dynamically executed functions may not have inspectable source.
Applications can register source strings in ``dag._source_registry`` as a
fallback.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union


@dataclass(frozen=True)
class Dependency:
    """A statically discovered computed-function call path.

    ``chain`` includes attributes needed to reach the call. For example,
    ``pair = self.PairObject(); pair.Spot()`` produces a ``Spot`` dependency
    rooted at ``self`` with the chain ``("PairObject", "Spot")``.
    """

    name: str
    chain: Tuple[str, ...] = ()
    has_args: bool = False
    is_indirect: bool = False
    root: str = "self"
    via_alias: bool = False
    lineno: Optional[int] = None
    col_offset: Optional[int] = None


# More descriptive public spelling while retaining the original class name.
DependencyPath = Dependency


class ParseStatus(str, Enum):
    """Outcome of dependency parsing."""

    SUCCESS = "success"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SYNTAX_ERROR = "syntax_error"
    FUNCTION_NOT_FOUND = "function_not_found"


@dataclass(frozen=True)
class DependencyParseResult:
    """Structured dependencies plus an explicit parse outcome."""

    status: ParseStatus
    dependencies: Tuple[Dependency, ...] = ()
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """Whether source retrieval and parsing completed successfully."""
        return self.status == ParseStatus.SUCCESS

    @property
    def names(self) -> FrozenSet[str]:
        """Dependency names used by the runtime same-object safety check."""
        return frozenset(dependency.name for dependency in self.dependencies)


@dataclass(frozen=True)
class _SymbolicPath:
    """A path known to originate from a function root."""

    root: str
    chain: Tuple[str, ...] = ()
    is_indirect: bool = False
    via_alias: bool = False


class DependencyVisitor(ast.NodeVisitor):
    """Extract dependency calls while propagating symbolic root paths."""

    _ACCESSOR_OPERATIONS = frozenset(
        {"set", "override", "watch", "clearValue", "clear_value"}
    )

    def __init__(
        self,
        self_name: str = "self",
        argument_names: Iterable[str] = (),
        line_offset: int = 0,
    ) -> None:
        roots = [self_name]
        roots.extend(name for name in argument_names if name != self_name)
        self.self_name = self_name
        self.dependencies: Set[str] = set()
        self.dependency_details: List[Dependency] = []
        self._detail_keys: Set[Tuple[object, ...]] = set()
        self._line_offset = line_offset
        self._aliases: Dict[str, Set[_SymbolicPath]] = {
            name: {_SymbolicPath(root=name)} for name in roots
        }

    def visit_statements(self, statements: Iterable[ast.stmt]) -> None:
        """Visit a statement sequence using the current alias environment."""
        for statement in statements:
            self.visit(statement)

    def _copy_aliases(self) -> Dict[str, Set[_SymbolicPath]]:
        return {name: set(paths) for name, paths in self._aliases.items()}

    @staticmethod
    def _merge_aliases(
        *states: Dict[str, Set[_SymbolicPath]],
    ) -> Dict[str, Set[_SymbolicPath]]:
        merged: Dict[str, Set[_SymbolicPath]] = {}
        for state in states:
            for name, paths in state.items():
                merged.setdefault(name, set()).update(paths)
        return merged

    def _run_block(
        self,
        statements: Iterable[ast.stmt],
        initial: Dict[str, Set[_SymbolicPath]],
    ) -> Dict[str, Set[_SymbolicPath]]:
        self._aliases = {name: set(paths) for name, paths in initial.items()}
        self.visit_statements(statements)
        return self._copy_aliases()

    @staticmethod
    def _mark_alias(paths: Set[_SymbolicPath]) -> Set[_SymbolicPath]:
        return {replace(path, via_alias=True) for path in paths}

    def _resolve_paths(self, node: ast.AST) -> Set[_SymbolicPath]:
        """Resolve an expression to every known root path it may represent."""
        if isinstance(node, ast.Name):
            return set(self._aliases.get(node.id, set()))

        if isinstance(node, ast.Attribute):
            return {
                replace(path, chain=path.chain + (node.attr,))
                for path in self._resolve_paths(node.value)
            }

        if isinstance(node, ast.Subscript):
            return {
                replace(path, is_indirect=True)
                for path in self._resolve_paths(node.value)
            }

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr in self._ACCESSOR_OPERATIONS
                    and isinstance(node.func.value, ast.Attribute)
                ):
                    target = node.func.value
                    return {
                        replace(path, chain=path.chain + (target.attr,))
                        for path in self._resolve_paths(target.value)
                    }
                return {
                    replace(path, chain=path.chain + (node.func.attr,))
                    for path in self._resolve_paths(node.func.value)
                }
            return self._resolve_paths(node.func)

        if isinstance(node, ast.IfExp):
            return self._resolve_paths(node.body) | self._resolve_paths(node.orelse)

        if isinstance(node, ast.BoolOp):
            paths: Set[_SymbolicPath] = set()
            for value in node.values:
                paths.update(self._resolve_paths(value))
            return paths

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            paths = set()
            for element in node.elts:
                paths.update(self._resolve_paths(element))
            return paths

        if isinstance(node, ast.Dict):
            paths = set()
            for value in node.values:
                paths.update(self._resolve_paths(value))
            return paths

        if isinstance(node, ast.NamedExpr):
            return self._resolve_paths(node.value)

        if isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom, ast.Starred)):
            wrapped_value: Optional[ast.AST] = getattr(node, "value", None)
            return self._resolve_paths(wrapped_value) if wrapped_value is not None else set()

        return set()

    def _record_paths(
        self,
        paths: Set[_SymbolicPath],
        node: ast.Call,
    ) -> None:
        for path in paths:
            if not path.chain:
                continue
            dependency = Dependency(
                name=path.chain[-1],
                chain=path.chain,
                has_args=bool(node.args or node.keywords),
                is_indirect=path.is_indirect,
                root=path.root,
                via_alias=path.via_alias,
                lineno=(
                    node.lineno + self._line_offset
                    if getattr(node, "lineno", None) is not None
                    else None
                ),
                col_offset=getattr(node, "col_offset", None),
            )
            key = (
                dependency.name,
                dependency.chain,
                dependency.root,
                dependency.via_alias,
                dependency.is_indirect,
                dependency.lineno,
                dependency.col_offset,
            )
            if key in self._detail_keys:
                continue
            self._detail_keys.add(key)
            self.dependencies.add(dependency.name)
            self.dependency_details.append(dependency)

    def visit_Call(self, node: ast.Call) -> None:
        """Record calls reached from a known root, then inspect arguments."""
        if isinstance(node.func, ast.Attribute):
            if (
                node.func.attr in self._ACCESSOR_OPERATIONS
                and isinstance(node.func.value, ast.Attribute)
            ):
                target = node.func.value
                self.visit(target.value)
                paths = {
                    replace(path, chain=path.chain + (target.attr,))
                    for path in self._resolve_paths(target.value)
                }
            else:
                self.visit(node.func.value)
                paths = {
                    replace(path, chain=path.chain + (node.func.attr,))
                    for path in self._resolve_paths(node.func.value)
                }
            self._record_paths(paths, node)
        elif isinstance(node.func, ast.Name):
            # Supports aliases to bound accessors: ``cell = self.A; cell()``.
            self._record_paths(self._resolve_paths(node.func), node)
        else:
            self.visit(node.func)

        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        paths = self._resolve_paths(node.value)
        for target in node.targets:
            self._bind_target(target, paths, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            self._bind_target(node.target, set())
            return
        self.visit(node.value)
        self._bind_target(node.target, self._resolve_paths(node.value), node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_target(node.target, self._resolve_paths(node.value), node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        self._bind_target(node.target, set())

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._bind_target(target, set())

    def _bind_target(
        self,
        target: ast.AST,
        paths: Set[_SymbolicPath],
        value: Optional[ast.AST] = None,
    ) -> None:
        if isinstance(target, ast.Name):
            if paths:
                self._aliases[target.id] = self._mark_alias(paths)
            else:
                self._aliases.pop(target.id, None)
            return

        if isinstance(target, ast.Starred):
            self._bind_target(target.value, paths, value)
            return

        if isinstance(target, (ast.Tuple, ast.List)):
            values = value.elts if isinstance(value, (ast.Tuple, ast.List)) else ()
            for index, element in enumerate(target.elts):
                element_paths = paths
                element_value: Optional[ast.AST] = value
                if index < len(values):
                    element_value = values[index]
                    element_paths = self._resolve_paths(element_value)
                self._bind_target(element, element_paths, element_value)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        entry = self._copy_aliases()
        body_state = self._run_block(node.body, entry)
        else_state = self._run_block(node.orelse, entry) if node.orelse else entry
        self._aliases = self._merge_aliases(body_state, else_state)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        entry = self._copy_aliases()
        body_state = self._run_block(node.body, entry)
        self._aliases = self._merge_aliases(entry, body_state)
        self.visit_statements(node.orelse)

    def _visit_for(self, node: Union[ast.For, ast.AsyncFor]) -> None:
        self.visit(node.iter)
        entry = self._copy_aliases()
        self._bind_target(node.target, self._resolve_paths(node.iter), node.iter)
        self.visit_statements(node.body)
        body_state = self._copy_aliases()
        self._aliases = self._merge_aliases(entry, body_state)
        self.visit_statements(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for(node)

    def _visit_with(self, node: Union[ast.With, ast.AsyncWith]) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(
                    item.optional_vars,
                    self._resolve_paths(item.context_expr),
                    item.context_expr,
                )
        self.visit_statements(node.body)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def _visit_comprehension(
        self,
        generators: List[ast.comprehension],
        expressions: Iterable[ast.AST],
    ) -> None:
        outer = self._copy_aliases()
        try:
            for generator in generators:
                self.visit(generator.iter)
                self._bind_target(
                    generator.target,
                    self._resolve_paths(generator.iter),
                    generator.iter,
                )
                for condition in generator.ifs:
                    self.visit(condition)
            for expression in expressions:
                self.visit(expression)
        finally:
            self._aliases = outer

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])

    def _visit_nested_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> None:
        outer = self._copy_aliases()
        args = node.args
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in list(args.defaults) + list(args.kw_defaults):
            if default is not None:
                self.visit(default)
        for argument in _argument_names(args):
            self._aliases.pop(argument, None)
        self.visit_statements(node.body)
        self._aliases = outer

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_nested_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_nested_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        outer = self._copy_aliases()
        for default in list(node.args.defaults) + list(node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for argument in _argument_names(node.args):
            self._aliases.pop(argument, None)
        self.visit(node.body)
        self._aliases = outer


def _argument_names(arguments: ast.arguments) -> List[str]:
    positional = list(getattr(arguments, "posonlyargs", [])) + list(arguments.args)
    names = [argument.arg for argument in positional]
    names.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return names


def _get_source_from_registry(func: Callable) -> Optional[str]:
    """Try to get source code from the Pyodide source registry."""
    try:
        import dag

        registry = getattr(dag, "_source_registry", {})
        for source in registry.values():
            if f"def {func.__name__}" in source:
                return source
    except ImportError:
        pass
    return None


def _find_function(
    tree: ast.AST,
    function_name: str,
) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    return None


def parse_dependency_result(func: Callable) -> DependencyParseResult:
    """Parse a function and return dependencies with explicit diagnostics."""
    source: Optional[str]
    line_offset = 0
    try:
        source_lines, first_line = inspect.getsourcelines(func)
        source = "".join(source_lines)
        line_offset = first_line - 1
    except (OSError, TypeError):
        source = _get_source_from_registry(func)
    if source is None:
        return DependencyParseResult(ParseStatus.SOURCE_UNAVAILABLE)

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as error:
        return DependencyParseResult(ParseStatus.SYNTAX_ERROR, error=str(error))

    function = _find_function(tree, func.__name__)
    if function is None:
        return DependencyParseResult(ParseStatus.FUNCTION_NOT_FOUND)

    argument_names = _argument_names(getattr(function, "args"))
    self_name = argument_names[0] if argument_names else "self"
    visitor = DependencyVisitor(self_name, argument_names[1:], line_offset)
    visitor.visit_statements(getattr(function, "body"))
    return DependencyParseResult(
        ParseStatus.SUCCESS,
        dependencies=tuple(visitor.dependency_details),
    )


def parse_dependencies(func: Callable) -> Optional[FrozenSet[str]]:
    """Return dependency names, or ``None`` when parsing was unsuccessful."""
    result = parse_dependency_result(func)
    return result.names if result.succeeded else None


def parse_dependencies_detailed(func: Callable) -> List[Dependency]:
    """Return structured dependency paths, or an empty list on parse failure."""
    result = parse_dependency_result(func)
    return list(result.dependencies) if result.succeeded else []


def get_function_parameters(func: Callable) -> List[str]:
    """Get function parameter names, excluding ``self`` or ``cls``."""
    try:
        params = list(inspect.signature(func).parameters.keys())
        if params and params[0] in ("self", "cls"):
            params = params[1:]
        return params
    except (ValueError, TypeError):
        return []
