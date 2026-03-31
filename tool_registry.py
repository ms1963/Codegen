# codegen/tool_registry.py

"""
tool_registry.py

The Dynamic Tool Registry: the central data store for all dynamically
generated and registered MCP tools.

Manages the full lifecycle of tools from registration through invocation
and removal. All state-modifying operations are protected by an asyncio.Lock
for safe concurrent access from the async MCP server and agent loop.

Fixes applied:
  N1/N7/N19 — import_tools deadlock fixed: register_tool called outside lock.
  N2/N10    — Stale entry reference fixed: snapshot entry before releasing lock;
               call_count and last_error mutated on snapshotted object.
  N4        — get_stats() fully implemented and wired.
  N17       — last_error exported and restored in persistence round-trip.
  FIX-A     — get_event_loop() replaced with get_running_loop().
  FIX-L     — tags normalised to list[str] defensively.
"""

import asyncio
import inspect
import logging
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any

from dynamic_loader import create_module_from_source

logger = logging.getLogger(__name__)  # FIX-N: __name__ not bare name

# ---------------------------------------------------------------------------
# Tool Entry Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ToolEntry:
    """
    A complete record for a single tool in the Dynamic Tool Registry.

    Attributes:
        name:         Unique tool identifier. Must be a valid Python identifier.
        description:  Human-readable description used by the LLM to decide
                      when to call this tool.
        input_schema: JSON Schema object describing the tool's input parameters.
        callable:     The actual Python function to invoke.
        source_code:  The original Python source code string.
        module_name:  The sys.modules key for the tool's dynamic module.
        created_at:   Unix timestamp of when the tool was registered.
        call_count:   Number of times this tool has been successfully called.
        last_error:   The last error message from a failed call, or None.
        tags:         Optional list of tags for categorization.
    """
    name: str
    description: str
    input_schema: dict
    callable: Any
    source_code: str
    module_name: str
    created_at: float = field(default_factory=time.time)
    call_count: int = 0
    last_error: str | None = None
    tags: list[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Dynamic Tool Registry
# ---------------------------------------------------------------------------

class DynamicToolRegistry:
    """
    Thread-safe registry of dynamically generated and loaded MCP tools.

    The registry is the single source of truth for all tools in the system.
    The MCP server queries it on every 'tools/list' request, ensuring that
    newly registered tools are immediately visible to clients.
    """

    DEFAULT_CALL_TIMEOUT: float = 30.0

    def __init__(self, call_timeout: float = DEFAULT_CALL_TIMEOUT) -> None:
        """
        Initialize an empty registry.

        Args:
            call_timeout: Maximum seconds to wait for a tool call to complete.
        """
        self._tools: dict[str, ToolEntry] = {}
        self._lock = asyncio.Lock()
        self._call_timeout = call_timeout

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    async def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        source_code: str,
        tags: list[str] | None = None,
    ) -> ToolEntry:
        """
        Register a new tool from its Python source code string.

        The source code is compiled into a proper Python module using
        importlib, and the named function is extracted from the module.
        If a tool with the same name already exists, it is atomically
        replaced with the new version.

        Args:
            name:         The tool's unique name. Must match the function name
                          in the source code exactly.
            description:  Human-readable description for the LLM.
            input_schema: JSON Schema dict for the tool's input parameters.
            source_code:  Python source code defining a function named 'name'.
            tags:         Optional list of tags for categorization.

        Returns:
            The newly created ToolEntry.

        Raises:
            ValueError:  If the source code does not contain a callable named
                         'name'.
            SyntaxError: If the source code has syntax errors.
            Exception:   If the module raises an exception during initialization.
        """
        module_name = f"dynamic_tools.{name}"

        # Load the source code as a proper Python module.
        # NOTE: create_module_from_source is called OUTSIDE the lock because
        # it is CPU-bound (compile + exec) and does not touch self._tools.
        # This avoids holding the lock during potentially slow compilation.
        module = create_module_from_source(module_name, source_code)

        # Extract the function from the module.
        func = getattr(module, name, None)
        if func is None:
            sys.modules.pop(module_name, None)
            raise ValueError(
                f"The generated module does not contain a function named "
                f"'{name}'. Ensure the function name in the source code "
                f"matches the tool name exactly."
            )
        if not callable(func):
            sys.modules.pop(module_name, None)
            raise ValueError(
                f"'{name}' in the generated module is not callable "
                f"(got {type(func)})."
            )

        # FIX-L: Normalise tags to list[str] defensively.
        normalised_tags: list[str] = (
            [str(t) for t in tags if isinstance(t, (str, int, float))]
            if tags
            else []
        )

        entry = ToolEntry(
            name=name,
            description=description,
            input_schema=input_schema,
            callable=func,
            source_code=source_code,
            module_name=module_name,
            tags=normalised_tags,
        )

        async with self._lock:
            # If replacing an existing tool, unload its old module first.
            old_entry = self._tools.get(name)
            if old_entry is not None and old_entry.module_name != module_name:
                sys.modules.pop(old_entry.module_name, None)
            self._tools[name] = entry

        logger.info(
            "Registered tool '%s' (module: %s, tags: %s).",
            name, module_name, normalised_tags,
        )
        return entry

    # -----------------------------------------------------------------------
    # Lookup
    # -----------------------------------------------------------------------

    async def get_all_tools(self) -> list[ToolEntry]:
        """Return a snapshot list of all currently registered tools."""
        async with self._lock:
            return list(self._tools.values())

    async def get_tool(self, name: str) -> ToolEntry | None:
        """Look up a tool by name. Returns None if not found."""
        async with self._lock:
            return self._tools.get(name)

    async def tool_exists(self, name: str) -> bool:
        """Return True if a tool with the given name is registered."""
        async with self._lock:
            return name in self._tools

    async def get_tool_names(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        async with self._lock:
            return sorted(self._tools.keys())

    # -----------------------------------------------------------------------
    # Invocation
    # -----------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Invoke a registered tool by name with the given keyword arguments.

        Supports both synchronous and asynchronous tool functions.
        Synchronous functions are run in a thread pool executor to avoid
        blocking the asyncio event loop. All calls are subject to the
        registry's call_timeout.

        Args:
            name:      The name of the tool to invoke.
            arguments: A dict of keyword arguments to pass to the function.

        Returns:
            The return value of the tool function.

        Raises:
            KeyError:             If no tool with the given name is registered.
            asyncio.TimeoutError: If the tool exceeds the call_timeout.
            Exception:            Any exception raised by the tool function.

        Fixes applied:
            N2/N10  — Snapshot entry reference inside the lock. After releasing
                       the lock, we hold a direct reference to the entry object.
                       Even if the tool is replaced in the registry, we invoke
                       the version we snapshotted — which is correct for an
                       in-flight call. call_count and last_error are mutated on
                       the snapshotted object, not re-fetched.
            FIX-A   — get_running_loop() replaces deprecated get_event_loop().
        """
        # FIX-B / N2: Snapshot the entry reference inside the lock.
        async with self._lock:
            entry = self._tools.get(name)
            if entry is None:
                raise KeyError(
                    f"No tool named '{name}' is registered. "
                    f"Available: {sorted(self._tools.keys())}"
                )
            # Snapshot — holds a direct reference to the ToolEntry object.
            snapshotted_entry = entry
        # Lock released here. We now work exclusively with snapshotted_entry.

        try:
            result = await asyncio.wait_for(
                self._invoke(snapshotted_entry, arguments),
                timeout=self._call_timeout,
            )
            # N10: update call_count on the snapshotted entry object directly.
            snapshotted_entry.call_count += 1
            snapshotted_entry.last_error = None
            return result

        except asyncio.TimeoutError:
            error_msg = (
                f"Tool '{name}' timed out after {self._call_timeout} seconds."
            )
            snapshotted_entry.last_error = error_msg
            raise asyncio.TimeoutError(error_msg)

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            snapshotted_entry.last_error = error_msg
            raise

    async def _invoke(self, entry: ToolEntry, arguments: dict[str, Any]) -> Any:
        """
        Internal helper that actually invokes the tool function.
        Handles both sync and async callables.

        FIX-A: use get_running_loop() — get_event_loop() is deprecated
        since Python 3.10 and raises DeprecationWarning in 3.12.
        """
        if inspect.iscoroutinefunction(entry.callable):
            return await entry.callable(**arguments)
        else:
            # FIX-A: use get_running_loop()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: entry.callable(**arguments),
            )

    # -----------------------------------------------------------------------
    # Removal
    # -----------------------------------------------------------------------

    async def remove_tool(self, name: str) -> bool:
        """
        Remove a tool from the registry and unload its module from sys.modules.

        Args:
            name: The tool name to remove.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        async with self._lock:
            entry = self._tools.pop(name, None)

        if entry is None:
            logger.warning(
                "Attempted to remove non-existent tool '%s'.", name
            )
            return False

        sys.modules.pop(entry.module_name, None)
        logger.info(
            "Removed tool '%s' and unloaded module '%s'.",
            name, entry.module_name,
        )
        return True

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    async def export_tools(self) -> list[dict]:
        """
        Export all registered tools as a list of serializable dicts.

        N17: last_error is now included in the export so it is preserved
        across process restarts.

        Returns:
            A list of dicts, each representing one tool's registration data.
        """
        async with self._lock:
            return [
                {
                    "name": e.name,
                    "description": e.description,
                    "input_schema": e.input_schema,
                    "source_code": e.source_code,
                    "tags": e.tags,
                    "created_at": e.created_at,
                    "call_count": e.call_count,
                    "last_error": e.last_error,  # N17: persist last_error
                }
                for e in self._tools.values()
            ]

    async def import_tools(self, tool_defs: list[dict]) -> tuple[int, int]:
        """
        Import and register tools from a list of serialized tool definitions.

        This is the counterpart to export_tools(). Used to restore previously
        generated tools after a process restart.

        P1 / N1 / N7 / N19: register_tool is called OUTSIDE any lock held by
        this method. register_tool acquires self._lock internally. Calling it
        while holding the lock would deadlock since asyncio.Lock is not
        reentrant.

        N17: last_error and call_count are restored from the persisted data.

        Args:
            tool_defs: A list of dicts as produced by export_tools().

        Returns:
            A tuple of (success_count, failure_count).
        """
        success = 0
        failure = 0

        for tool_def in tool_defs:
            try:
                name = tool_def["name"]
                description = tool_def["description"]
                input_schema = tool_def["input_schema"]
                source_code = tool_def["source_code"]
                tags = tool_def.get("tags", [])

                # P1: Call register_tool WITHOUT holding self._lock.
                await self.register_tool(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    source_code=source_code,
                    tags=tags,
                )

                # N17: Restore persisted stats after registration.
                # We re-acquire the lock briefly to update the entry.
                async with self._lock:
                    if name in self._tools:
                        self._tools[name].call_count = tool_def.get(
                            "call_count", 0
                        )
                        self._tools[name].last_error = tool_def.get(
                            "last_error", None
                        )
                        self._tools[name].created_at = tool_def.get(
                            "created_at", self._tools[name].created_at
                        )

                success += 1
                logger.info("Imported tool '%s' successfully.", name)

            except Exception as exc:
                failure += 1
                logger.error(
                    "Failed to import tool '%s': %s",
                    tool_def.get("name", "<unknown>"),
                    exc,
                    exc_info=True,
                )

        return success, failure

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    async def get_stats(self) -> dict:
        """
        Return a statistics snapshot of the registry for monitoring.

        N4: Fully implemented and wired.

        Returns:
            A dict with total tool count and per-tool statistics.
        """
        async with self._lock:
            return {
                "total_tools": len(self._tools),
                "tools": [
                    {
                        "name": e.name,
                        "call_count": e.call_count,
                        "last_error": e.last_error,
                        "created_at": e.created_at,
                        "tags": e.tags,
                        "description": e.description,
                    }
                    for e in self._tools.values()
                ],
            }
