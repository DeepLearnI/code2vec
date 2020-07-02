from __future__ import annotations

import ast
from ast import NodeVisitor, increment_lineno
from contextlib import contextmanager
import dataclasses as dc
import collections
import itertools as it
import re
from typing import (
    Iterable,
    List,
    Iterator,
    Union,
    Dict,
    List,
    Optional,
    TypeVar,
)

A = TypeVar("A")


class Extractor(NodeVisitor):
    def __init__(self, max_path_length, max_path_width):
        self._stack: List[JsonTree] = list()
        self._func_name: str = str()
        self.tree: List[JsonTree] = list()
        self._json_tree: List[dict] = list()
        self.paths: List[List[JsonTree]] = list()
        self.paths_map: Dict[str, List[List[str]]] = dict()
        self.MAX_DEPTH = max_path_length
        self.MAX_WIDTH = max_path_width
        self._build_json = False
        self.replace_pattern = re.compile("[0-9_,]")

    def add_path(self, path: List[JsonTree]):
        """
        The hyperparameters for filtering out some paths could be applied here.

        """

        def transform(tree: JsonTree) -> str:
            return tree.node_type if isinstance(tree, JsonNode) else tree.value

        for prev_path in self.paths:
            merged = self.merge(prev_path, path)
            if merged:
                # For some reason we must cast to list here or the last token goes missing
                self.paths_map[self._func_name].append(list(map(transform, merged)))

        self.paths.append(list(path))

    def merge(
        self, left_path: List[JsonTree], right_path: List[JsonTree]
    ) -> Optional[Iterator[JsonTree]]:
        """
                        V
                      /   \

        Once we have vertex, the index of left and right canot be more than W apart?

        """
        if self.fails_depth_check(left_path, right_path):
            return None

        lefts, rights = iter(left_path), iter(right_path)
        for left, right in zip(lefts, rights):
            if id(left) == id(right):
                vertex = left
            else:
                break

        if self.fails_width_check(vertex.children, left, right):
            return None

        merged = it.chain(reversed(list(lefts)), [left, vertex, right], rights)
        return merged

    def fails_depth_check(self, left_path: Iterable, right_path: Iterable) -> bool:
        length = len(set(map(id, left_path)) ^ set(map(id, right_path))) + 1
        return length > self.MAX_DEPTH

    def fails_width_check(self, children: List[A], left: A, right: A) -> bool:
        left_index = children.index(left)
        right_index = children.index(right)

        return right_index - left_index > self.MAX_WIDTH

    def _parse(self, fname: str) -> List[dict]:
        """I only care about function/method definitions for now"""

        with open(fname, "r", encoding="ISO-8859-1") as stream:
            tree = ast.parse(stream.read())

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._func_name = self.clean(node.name)
                self.paths_map[self._func_name] = []
                self.visit(node)
                self.paths = list()

        return self._json_tree

    def clean(self, token: str) -> str:
        token = self.to_snake_case(token)
        return self.replace_pattern.sub("|", token).strip("|")

    @staticmethod
    def to_snake_case(token: str) -> str:
        splitted = re.sub(
            "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", token)
        ).split()
        return "_".join(word.strip('_').lower() for word in splitted)

    def extract_paths(self, fname: str) -> Iterator[str]:
        # Can perhaps take the transformation function as a parameter for more flexibility
        def transform(path_contexts: Iterable[Iterable[str]]) -> str:
            def to_str_path(path_context: Iterable[str]) -> str:
                context = list(path_context)
                return f"{self.clean(context[0])},{'|'.join(context[1:-1])},{self.clean(context[-1])}"

            str_paths = map(to_str_path, path_contexts)
            return f"{' '.join(str_paths)}".encode("unicode_escape").decode(
                "ISO-8859-1"
            )

        self._parse(fname)

        paths = (
            f"{func_name} {transform(contexts)}\n"
            for func_name, contexts in self.paths_map.items()
        )
        self.paths_map = dict()
        return paths

    def to_json(self, fname: str) -> List[dict]:
        """Parse the syntax tree and output to json

        :param fname: The filename.
        :type fname: str
        :return: A json representation of the syntax tree.
        :rtype: dict
        """
        self._build_json = True
        json_tree = self._parse(fname)
        self._json_tree = list()
        self._build_json = False
        return json_tree

    def visit(self, node: ast.AST):
        if isinstance(
            node, (ast.boolop, ast.cmpop, ast.unaryop, ast.operator, ast.expr_context)
        ):
            return
        if isinstance(node, (ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.AugAssign)):
            self.visit_Op(node)
        elif isinstance(node, (ast.Break, ast.Continue)):
            self.visit_Break_Continue(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.visit_Function(node)
        else:
            super().visit(node)

    def generic_visit(self, node: ast.AST):
        """Called if no explicit visitor function exists for a node."""
        json_node = JsonNode(type(node).__name__)
        with self.push_to_stack(json_node):
            super().generic_visit(node)

    def visit_Function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        method_name = JsonLeaf("MethodName", node.name)
        if ast.get_docstring(node):
            node.body = node.body[1:]
        json_node = JsonNode(type(node).__name__)
        with self.push_to_stack(json_node):
            with self.push_to_stack(method_name):
                self.add_path(self._stack)
            super().generic_visit(node)
        self.tree.append(json_node)
        if self._build_json:
            self._json_tree.append(json_node.to_dict())

    def visit_arg(self, node: ast.arg):
        arg_name = JsonLeaf("ArgName", node.arg)
        json_node = JsonNode(type(node).__name__)
        with self.push_to_stack(json_node):
            with self.push_to_stack(arg_name):
                self.add_path(self._stack)
            super().generic_visit(node)

    def visit_Op(self, node: Union[ast.UnaryOp, ast.BoolOp, ast.BinOp, ast.AugAssign]):
        operator = type(node.op).__name__
        node_type = "_".join([type(node).__name__, operator])
        json_node = JsonNode(node_type)
        with self.push_to_stack(json_node):
            super().generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        json_node = JsonNode("Compare_" + type(node.ops[0]).__name__)
        with self.push_to_stack(json_node):
            super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        json_node = JsonLeaf(type(node).__name__, node.id)
        with self.push_to_stack(json_node):
            self.add_path(self._stack)

    def visit_Constant(self, node: ast.Constant):
        json_node = JsonLeaf(
            type(node).__name__,
            str(node.value).encode("unicode_escape").decode().replace(" ", ""),
        )
        with self.push_to_stack(json_node):
            self.add_path(self._stack)

    def visit_Break_Continue(self, node: Union[ast.Break, ast.Continue]):
        json_node = JsonLeaf(type(node).__name__, type(node).__name__)
        with self.push_to_stack(json_node):
            self.add_path(self._stack)

    def add_to_tree(self, json_tree: JsonTree):
        current_node = self._stack.pop()
        if isinstance(current_node, JsonNode):
            current_node.children.append(json_tree)
        else:
            raise RuntimeError("JsonLeaf node left on stack!")
        self._stack.append(current_node)

    @contextmanager
    def push_to_stack(self, json_node: JsonTree):
        if self._stack:
            self.add_to_tree(json_node)
        self._stack.append(json_node)
        try:
            yield
        finally:
            self._stack.pop()


class _JsonTree:
    """"""

    def to_dict(self):
        return dc.asdict(self)


@dc.dataclass(frozen=True)
class JsonNode(_JsonTree):
    node_type: str
    children: List[_JsonTree] = dc.field(default_factory=list)


@dc.dataclass(frozen=True)
class JsonLeaf(_JsonTree):
    node_type: str
    value: str


JsonTree = Union[JsonNode, JsonLeaf]
