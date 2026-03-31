# codegen/tool_validator.py

"""
tool_validator.py

Static AST-based validation of LLM-generated tool code, plus code extraction
from raw LLM responses.

Static validation uses Python's ast module to analyze the code structure
without executing it. Checks for forbidden imports, forbidden name references,
and validates the structure of the top-level function definition (docstring,
type annotations). Also extracts the function name and builds a JSON Schema
from the function's type annotations.

Fixes applied:
  N5    — SUPPORTED_TYPES dict defined with complete type mapping.
  N6    — extract_python_code() implemented with regex fence extraction.
  N13   — visit_AsyncFunctionDef alias handled; FIX-G save/restore prevents
           double-counting for nested async functions.
  FIX-G — _in_function flag save/restore pattern prevents corruption.
  FIX-K — Optional[X] and Union[X, None] fully unwrapped in annotation
           converter.
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)  # FIX-N: __name__ not bare name

# ---------------------------------------------------------------------------
# Security: Forbidden Imports and Names
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORTS: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "builtins", "ctypes", "gc", "inspect",
    "threading", "multiprocessing", "signal", "pty",
    "pickle", "shelve", "marshal", "code", "codeop",
    "ast", "dis", "tokenize", "token", "py_compile",
    "compileall", "zipimport", "zipfile", "tarfile",
    "tempfile", "io", "struct", "mmap",
})

FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "exec", "eval", "compile", "import", "open",
    "builtins", "loader", "spec", "file",
    "__subclasses__", "__bases__", "__mro__",
    "globals", "locals", "vars", "dir", "delattr",
    "__reduce_ex__", "__reduce__",
    "breakpoint", "input",
})

# ---------------------------------------------------------------------------
# P5: SUPPORTED_TYPES — Python type names → JSON Schema type strings
# ---------------------------------------------------------------------------

SUPPORTED_TYPES: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "List": "array",
    "dict": "object",
    "Dict": "object",
    "tuple": "array",
    "Tuple": "array",
    "set": "array",
    "Set": "array",
    "bytes": "string",
    "None": "null",
    "Any": "string",
    "Sequence": "array",
    "Mapping": "object",
}

# ---------------------------------------------------------------------------
# Validation Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    The result of validating a piece of LLM-generated tool source code.

    Attributes:
        is_valid:      True if the code passed all validation checks.
        errors:        List of human-readable error messages.
        function_name: The name of the validated function, or None.
        input_schema:  JSON Schema dict for the function's parameters.
        description:   The function's docstring, or None.
    """
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    function_name: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------

class ToolCodeVisitor(ast.NodeVisitor):
    """
    AST visitor that performs static security and structure analysis on
    LLM-generated tool code.

    Checks for forbidden imports, forbidden name references, and validates
    the structure of the top-level function definition (docstring,
    annotations). Extracts the function name and docstring for registration.

    Fixes applied:
        FIX-G — save/restore _in_function so nested functions do not corrupt
                 the outer function's flag after generic_visit returns.
        N13   — visit_AsyncFunctionDef alias preserved; save/restore pattern
                 prevents double-counting when both sync and async top-level
                 functions exist (which is itself an error caught by stage 4).
    """

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.function_name: str | None = None
        self.function_node: ast.FunctionDef | None = None
        self.docstring: str | None = None
        self._top_level_function_count: int = 0
        self._in_function: bool = False

    def visit_Import(self, node: ast.Import) -> None:
        """Reject any import of a forbidden module."""
        for alias in node.names:
            base = alias.name.split(".")[0]
            if base in FORBIDDEN_IMPORTS:
                self.errors.append(
                    f"Line {node.lineno}: Forbidden import '{alias.name}'. "
                    f"Use the pre-loaded modules in the namespace instead."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Reject any 'from X import Y' where X is a forbidden module."""
        if node.module:
            base = node.module.split(".")[0]
            if base in FORBIDDEN_IMPORTS:
                self.errors.append(
                    f"Line {node.lineno}: Forbidden import "
                    f"'from {node.module} import ...'. "
                    f"Use the pre-loaded modules in the namespace instead."
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Reject references to forbidden built-in names."""
        if node.id in FORBIDDEN_NAMES:
            self.errors.append(
                f"Line {node.lineno}: Forbidden name '{node.id}' is not "
                f"allowed in generated tool code."
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Reject access to forbidden attributes (e.g., __subclasses__)."""
        if node.attr in FORBIDDEN_NAMES:
            self.errors.append(
                f"Line {node.lineno}: Forbidden attribute access "
                f"'.{node.attr}' is not allowed in generated tool code."
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Validate the structure of the top-level function definition.
        Only the outermost function definition is treated as the tool.
        Nested functions are visited for security checks but not analyzed
        as tool definitions.

        FIX-G: save/restore _in_function so nested functions do not
        corrupt the outer function's flag after generic_visit returns.
        N13:   The alias below means async top-level functions are also
               counted; save/restore prevents double-count corruption.
        """
        was_in_function = self._in_function  # FIX-G: save state

        if not self._in_function:
            # This is a top-level function definition.
            self._top_level_function_count += 1
            self._in_function = True

            if self._top_level_function_count == 1:
                self.function_node = node
                self.function_name = node.name

                # Validate snake_case naming.
                if not re.match(r"^[a-z][a-z0-9_]*$", node.name):
                    self.errors.append(
                        f"Function name '{node.name}' must be lowercase "
                        f"snake_case (e.g., 'calculate_sum', 'fetch_data')."
                    )

                # Validate docstring presence and minimum length.
                docstring = ast.get_docstring(node)
                if not docstring:
                    self.errors.append(
                        f"Function '{node.name}' must have a docstring as "
                        f"its first statement."
                    )
                elif len(docstring.strip()) < 10:
                    self.errors.append(
                        f"Function '{node.name}' docstring is too short "
                        f"(minimum 10 characters)."
                    )
                else:
                    self.docstring = docstring.strip()

                # Validate return type annotation.
                if node.returns is None:
                    self.errors.append(
                        f"Function '{node.name}' must have a return type "
                        f"annotation (e.g., '-> str:')."
                    )

                # Validate that all parameters have type annotations.
                for arg in node.args.args:
                    if arg.arg in ("self", "cls"):
                        continue
                    if arg.annotation is None:
                        self.errors.append(
                            f"Parameter '{arg.arg}' of function '{node.name}' "
                            f"must have a type annotation."
                        )

        # Visit children (handles nested functions and security checks).
        self.generic_visit(node)
        self._in_function = was_in_function  # FIX-G: restore state

    # N13: Async function definitions follow the same rules.
    # The save/restore in visit_FunctionDef prevents double-counting.
    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Type Annotation → JSON Schema Type Converter
# ---------------------------------------------------------------------------

def _annotation_to_json_type(annotation: ast.expr | None) -> str:
    """
    Convert an AST annotation node to a JSON Schema type string.

    FIX-K: fully unwraps Optional[X] (i.e. Union[X, None]) to the inner
    type X. Also handles Python 3.10+ union syntax (X | Y).

    Args:
        annotation: An AST expression node representing a type annotation.

    Returns:
        A JSON Schema type string such as 'string', 'integer', 'number', etc.
        Falls back to 'string' for unrecognised annotations.
    """
    if annotation is None:
        return "string"

    # Handle Optional[X] → Union[X, None]: unwrap to X
    # AST form: Subscript(value=Name('Optional'), slice=X)
    if isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            outer = annotation.value.id

            # Optional[X] is sugar for Union[X, None]
            if outer == "Optional":
                return _annotation_to_json_type(annotation.slice)

            # Union[X, None] — take the first non-None argument
            if outer == "Union":
                slice_node = annotation.slice
                # Python 3.9+: slice is a Tuple node
                if isinstance(slice_node, ast.Tuple):
                    for elt in slice_node.elts:
                        if not (
                            isinstance(elt, ast.Constant) and elt.value is None
                        ):
                            return _annotation_to_json_type(elt)
                else:
                    # Single-element Union (unusual but handle gracefully)
                    return _annotation_to_json_type(slice_node)

            # List[X] → array, Dict[K, V] → object
            if outer in ("List", "list", "Sequence", "Tuple", "tuple",
                         "Set", "set", "FrozenSet", "frozenset"):
                return "array"
            if outer in ("Dict", "dict", "Mapping", "MutableMapping"):
                return "object"

    # Simple Name annotation: str, int, float, bool, list, dict, …
    if isinstance(annotation, ast.Name):
        return SUPPORTED_TYPES.get(annotation.id, "string")

    # Attribute annotation (e.g. typing.Optional) — best-effort
    if isinstance(annotation, ast.Attribute):
        return SUPPORTED_TYPES.get(annotation.attr, "string")

    # Python 3.10+ union syntax: X | Y  (ast.BinOp with ast.BitOr)
    if isinstance(annotation, ast.BinOp) and isinstance(
        annotation.op, ast.BitOr
    ):
        # Take the left side if it's not None
        left = annotation.left
        if not (isinstance(left, ast.Constant) and left.value is None):
            return _annotation_to_json_type(left)
        return _annotation_to_json_type(annotation.right)

    # Constant annotation (e.g. None as a type)
    if isinstance(annotation, ast.Constant):
        if annotation.value is None:
            return "null"

    return "string"

# ---------------------------------------------------------------------------
# JSON Schema Builder
# ---------------------------------------------------------------------------

def build_input_schema(func_node: ast.FunctionDef) -> dict[str, Any]:
    """
    Build a JSON Schema 'object' dict from a function's parameter annotations.

    Args:
        func_node: The AST FunctionDef (or AsyncFunctionDef) node for the
                   tool function.

    Returns:
        A JSON Schema dict describing the function's input parameters.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    args = func_node.args
    defaults = args.defaults  # defaults are right-aligned

    # Map each argument to its default (or sentinel if no default).
    num_args = len(args.args)
    num_defaults = len(defaults)
    # Pad with None sentinels on the left for args that have no default.
    padded_defaults: list[ast.expr | None] = (
        [None] * (num_args - num_defaults) + list(defaults)
    )

    for arg, default in zip(args.args, padded_defaults):
        if arg.arg in ("self", "cls"):
            continue

        json_type = _annotation_to_json_type(arg.annotation)  # FIX-K
        properties[arg.arg] = {"type": json_type}

        if default is None:
            # No default value → required parameter.
            required.append(arg.arg)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }

# ---------------------------------------------------------------------------
# P6: Code Extraction from Raw LLM Responses
# ---------------------------------------------------------------------------

def extract_python_code(raw_response: str) -> str:
    """
    Extract a Python code block from a raw LLM response string.

    Handles the following common LLM output patterns:
      1. Triple-backtick fenced code block with 'python' language tag.
      2. Triple-backtick fenced code block without a language tag.
      3. Raw code with no fencing (returned as-is after stripping).

    Args:
        raw_response: The raw string returned by the LLM.

    Returns:
        The extracted Python source code, stripped of leading/trailing
        whitespace. Returns the original string (stripped) if no fence
        is found.
    """
    if not raw_response:
        return ""

    # Pattern 1: ```python ... ``` (most common)
    match = re.search(
        r"```(?:python|py)\s*\n(.*?)```",
        raw_response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Pattern 2: ``` ... ``` (no language tag)
    match = re.search(
        r"```\s*\n(.*?)```",
        raw_response,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Pattern 3: No fencing — return stripped raw response.
    return raw_response.strip()

# ---------------------------------------------------------------------------
# Main Validation Entry Point
# ---------------------------------------------------------------------------

def validate_tool_code(source_code: str) -> ValidationResult:
    """
    Perform full static validation of LLM-generated tool source code.

    Runs AST-based security and structure checks. Does NOT execute the code.

    Args:
        source_code: The Python source code string to validate.

    Returns:
        A ValidationResult with all errors found and extracted metadata.
    """
    if not source_code or not source_code.strip():
        return ValidationResult(
            is_valid=False,
            errors=["Source code is empty."],
        )

    # Stage 1: Syntax check via compile().
    try:
        compile(source_code, "<validation>", "exec")
    except SyntaxError as exc:
        return ValidationResult(
            is_valid=False,
            errors=[f"SyntaxError at line {exc.lineno}: {exc.msg}"],
        )

    # Stage 2: Parse into an AST.
    try:
        tree = ast.parse(source_code)
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            errors=[f"Failed to parse AST: {exc}"],
        )

    # Stage 3: Run the AST visitor for security and structure checks.
    visitor = ToolCodeVisitor()
    visitor.visit(tree)
    errors = list(visitor.errors)

    # Stage 4: Ensure exactly one top-level function was found.
    if visitor._top_level_function_count == 0:
        errors.append(
            "No top-level function definition found. "
            "The generated code must contain exactly one function."
        )
    elif visitor._top_level_function_count > 1:
        errors.append(
            f"Found {visitor._top_level_function_count} top-level function "
            f"definitions. Exactly one is required."
        )

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    # Stage 5: Build the JSON Schema from the function's annotations.
    input_schema = build_input_schema(visitor.function_node)

    return ValidationResult(
        is_valid=True,
        errors=[],
        function_name=visitor.function_name,
        input_schema=input_schema,
        description=visitor.docstring,
    )
