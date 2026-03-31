# codegen/dynamic_loader.py

"""
dynamic_loader.py

Utilities for loading Python source code as proper module objects at runtime,
without writing anything to disk. This module is the foundation of the
self-extending capability of the AI system.

All functions in this module are safe to call from async contexts.
They do not perform any network I/O and do not block the event loop.

Fixes applied:
  FIX-F  — All stdlib imports moved to module level (were inside function body).
           requests import guarded with try/except ImportError.
  P8     — safe_builtins completed with all 20+ common exception types.
  N11    — NotImplementedError was truncated; now fully present.
"""

# FIX-F: all stdlib imports moved to module level
import base64
import collections
import datetime
import decimal
import fractions
import functools
import hashlib
import importlib.abc
import importlib.util
import itertools
import json
import logging
import math
import random
import re
import statistics
import string
import sys
import types
import urllib.parse
from typing import Any

# FIX-F: guard requests import — it is an optional third-party dependency.
# Generated tools that use `requests` will only work if it is installed.
try:
    import requests as _requests_module
except ImportError:
    _requests_module = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)  # FIX-N: __name__ not bare name

# ---------------------------------------------------------------------------
# Custom In-Memory Module Loader
# ---------------------------------------------------------------------------

class InMemoryLoader(importlib.abc.SourceLoader):
    """
    A custom importlib loader that reads Python source code from an
    in-memory string rather than from the filesystem.

    Inherits from SourceLoader, which provides exec_module() automatically
    by compiling the bytes returned by get_data() and executing them.
    """

    def __init__(self, module_name: str, source_code: str) -> None:
        self._module_name = module_name
        self._source_code = source_code

    def get_source(self, fullname: str) -> str:
        """Return the source code string. Required by SourceLoader."""
        return self._source_code

    def get_data(self, path: str) -> bytes:
        """Return the source code as UTF-8 bytes. Required by SourceLoader."""
        return self._source_code.encode("utf-8")

    def get_filename(self, fullname: str) -> str:
        """
        Return a descriptive pseudo-filename for use in tracebacks.
        Makes error messages much more readable than just '<string>'.
        """
        return f"<dynamic_module:{self._module_name}>"

# ---------------------------------------------------------------------------
# Safe Execution Namespace Builder
# ---------------------------------------------------------------------------

def build_safe_namespace(
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a restricted execution namespace for dynamically generated tool code.

    The namespace contains a curated whitelist of safe built-in functions and
    a set of pre-imported utility libraries. Generated code can only access
    what is explicitly provided here.

    Args:
        extra_globals: Additional names to inject into the namespace.

    Returns:
        A dict suitable for use as the globals argument to exec().

    Fixes applied:
        FIX-F — imports now at module level; no repeated imports on each call.
        P8    — completed with all common exception types (N11, N20).
    """
    # P8 / N11 / N20: complete safe_builtins with all common exception types.
    safe_builtins: dict[str, Any] = {
        # ── Types ────────────────────────────────────────────────────────────
        "bool": bool,
        "bytes": bytes,
        "bytearray": bytearray,
        "complex": complex,
        "dict": dict,
        "float": float,
        "frozenset": frozenset,
        "int": int,
        "list": list,
        "memoryview": memoryview,
        "object": object,
        "set": set,
        "slice": slice,
        "str": str,
        "tuple": tuple,
        # ── Math / numeric builtins ──────────────────────────────────────────
        "abs": abs,
        "divmod": divmod,
        "hash": hash,
        "hex": hex,
        "oct": oct,
        "bin": bin,
        "pow": pow,
        "round": round,
        "sum": sum,
        "max": max,
        "min": min,
        # ── Iteration helpers ────────────────────────────────────────────────
        "all": all,
        "any": any,
        "enumerate": enumerate,
        "filter": filter,
        "len": len,
        "map": map,
        "range": range,
        "reversed": reversed,
        "sorted": sorted,
        "zip": zip,
        "next": next,
        "iter": iter,
        # ── String helpers ───────────────────────────────────────────────────
        "ascii": ascii,
        "chr": chr,
        "format": format,
        "ord": ord,
        # ── I/O (safe subset) ────────────────────────────────────────────────
        "print": print,
        "repr": repr,
        # ── Introspection (safe subset) ──────────────────────────────────────
        "callable": callable,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "getattr": getattr,
        "hasattr": hasattr,
        "setattr": setattr,
        "id": id,
        "len": len,
        # ── Exceptions — P8: complete set of common exceptions ───────────────
        "ArithmeticError": ArithmeticError,
        "AssertionError": AssertionError,
        "AttributeError": AttributeError,
        "BufferError": BufferError,
        "EOFError": EOFError,
        "Exception": Exception,
        "FloatingPointError": FloatingPointError,
        "GeneratorExit": GeneratorExit,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "KeyboardInterrupt": KeyboardInterrupt,
        "LookupError": LookupError,
        "MemoryError": MemoryError,
        "NameError": NameError,
        "NotImplementedError": NotImplementedError,  # N11: was truncated
        "OSError": OSError,
        "IOError": IOError,
        "OverflowError": OverflowError,
        "RecursionError": RecursionError,
        "RuntimeError": RuntimeError,
        "StopAsyncIteration": StopAsyncIteration,
        "StopIteration": StopIteration,
        "SyntaxError": SyntaxError,
        "TimeoutError": TimeoutError,
        "TypeError": TypeError,
        "UnicodeDecodeError": UnicodeDecodeError,
        "UnicodeEncodeError": UnicodeEncodeError,
        "UnicodeError": UnicodeError,
        "UnicodeTranslateError": UnicodeTranslateError,
        "ValueError": ValueError,
        "ZeroDivisionError": ZeroDivisionError,
        "ConnectionError": ConnectionError,
        "FileNotFoundError": FileNotFoundError,
        "IsADirectoryError": IsADirectoryError,
        "NotADirectoryError": NotADirectoryError,
        "PermissionError": PermissionError,
        "ProcessLookupError": ProcessLookupError,
        # ── Warning base (useful for tools that issue warnings) ──────────────
        "Warning": Warning,
        "UserWarning": UserWarning,
        "DeprecationWarning": DeprecationWarning,
        "RuntimeWarning": RuntimeWarning,
        "True": True,
        "False": False,
        "None": None,
        "NotImplemented": NotImplemented,
        "Ellipsis": Ellipsis,
    }

    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "json": json,
        "re": re,
        "math": math,
        "statistics": statistics,
        "hashlib": hashlib,
        "base64": base64,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
        "string": string,
        "decimal": decimal,
        "fractions": fractions,
        "random": random,
        "datetime": datetime,
        "urllib_parse": urllib.parse,
    }

    # FIX-F: conditionally add requests only if it was importable.
    if _requests_module is not None:
        namespace["requests"] = _requests_module

    if extra_globals:
        namespace.update(extra_globals)

    return namespace

# ---------------------------------------------------------------------------
# Core Module Creation
# ---------------------------------------------------------------------------

def create_module_from_source(
    module_name: str, source_code: str
) -> types.ModuleType:
    """
    Create a fully initialized Python module object from a source code string.

    The resulting module is registered in sys.modules under the given name,
    making it importable by other parts of the application.

    Args:
        module_name: The fully qualified module name, e.g. 'dynamic_tools.calc'.
        source_code: The Python source code to compile and execute as the
                     module body.

    Returns:
        The fully initialized module object.

    Raises:
        SyntaxError: If the source code contains syntax errors.
        Exception:   If the module's top-level code raises during exec_module.
    """
    loader = InMemoryLoader(module_name, source_code)

    spec = importlib.util.spec_from_loader(
        module_name,
        loader,
        origin=f"<dynamic:{module_name}>",
    )

    module = importlib.util.module_from_spec(spec)

    # Register BEFORE exec_module to handle potential self-references.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Remove the broken module so it is not found in a broken state.
        sys.modules.pop(module_name, None)
        raise

    logger.debug("Created module '%s' from in-memory source.", module_name)
    return module

# ---------------------------------------------------------------------------
# Safe Module Reload
# ---------------------------------------------------------------------------

def safe_reload_module(
    module_name: str,
    new_source_code: str,
) -> tuple[bool, str, types.ModuleType | None]:
    """
    Attempt to replace an existing dynamic module with new source code.

    Provides atomic replacement semantics: if the new code fails to compile
    or initialize, the old module is preserved in sys.modules and the
    function returns a failure result.

    Args:
        module_name:     The sys.modules key of the module to replace.
        new_source_code: The new Python source code.

    Returns:
        A tuple of (success: bool, message: str, module: ModuleType | None).
        On success, module is the newly created module object.
        On failure, module is None and the old module (if any) is preserved.
    """
    old_module = sys.modules.get(module_name)

    # Pre-flight syntax check before doing any module infrastructure work.
    try:
        compile(new_source_code, f"<dynamic:{module_name}>", "exec")
    except SyntaxError as exc:
        if old_module is not None:
            sys.modules[module_name] = old_module
        error_msg = f"SyntaxError in '{module_name}': {exc}"
        logger.error(error_msg)
        return False, error_msg, None

    try:
        new_module = create_module_from_source(module_name, new_source_code)
        logger.info("Successfully reloaded module '%s'.", module_name)
        return True, f"Module '{module_name}' reloaded successfully.", new_module

    except Exception as exc:
        if old_module is not None:
            sys.modules[module_name] = old_module
        elif module_name in sys.modules:
            del sys.modules[module_name]
        error_msg = f"RuntimeError initializing '{module_name}': {exc}"
        logger.error(error_msg)
        return False, error_msg, None
