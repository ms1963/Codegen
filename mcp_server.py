# codegen/mcp_server.py

"""
mcp_server.py

The Dynamic MCP Server: exposes the Dynamic Tool Registry via the
Model Context Protocol using the low-level MCP Python SDK Server class.

Handles 'tools/list' and 'tools/call' requests dynamically, reading from
the live registry on every request. Also exposes built-in meta-tools for
self-extension (generate_and_register_tool), introspection
(list_registered_tools, get_tool_source), lifecycle management (remove_tool),
and monitoring (get_registry_stats).

The server can be run over anyio memory streams (in-process), stdio, or HTTP.

Fixes applied:
  N3    — entry_to_mcp_tool() implemented.
  N4    — get_stats() call now works (registry has the method).
  N12   — get_capabilities() called with NotificationOptions() instead of
           None to prevent AttributeError on tools_changed access.
  N16   — _handle_get_registry_stats takes no arguments (consistent).
  N21   — ToolEntry kept as runtime import (needed for entry_to_mcp_tool param).
  FIX-C — Dead imports removed (RequestContext, ToolsCapability, math,
           duplicate InitializationOptions).
  FIX-J — include_schemas argument is now actually honoured.
  FIX-L — tags normalised with isinstance checks.
"""

import asyncio
import json
import logging
import re
from typing import Any

import anyio
import mcp.types as mcp_types
from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
)
from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
from code_generator import CodeGenerationPipeline
from tool_registry import DynamicToolRegistry, ToolEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P3: entry_to_mcp_tool — Convert ToolEntry → MCP Tool descriptor
# ---------------------------------------------------------------------------

def entry_to_mcp_tool(entry: ToolEntry) -> mcp_types.Tool:
    """
    Convert a ToolEntry from the registry into an MCP Tool descriptor.

    P3: This function was missing — now fully implemented.

    Args:
        entry: The ToolEntry to convert.

    Returns:
        An mcp_types.Tool suitable for returning from a list_tools handler.
    """
    return mcp_types.Tool(
        name=entry.name,
        description=entry.description,
        inputSchema=entry.input_schema,
    )


# ---------------------------------------------------------------------------
# Meta-Tool Definitions
# ---------------------------------------------------------------------------

def make_meta_tools() -> list[mcp_types.Tool]:
    """
    Return the list of built-in meta-tool descriptors.

    These tools are always available regardless of what dynamic tools
    have been registered.
    """
    return [
        mcp_types.Tool(
            name="generate_and_register_tool",
            description=(
                "Generate a new Python tool function from a natural language "
                "description and register it immediately for use. "
                "Use this when you need a capability that does not exist yet."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "capability_description": {
                        "type": "string",
                        "description": (
                            "A detailed natural language description of what "
                            "the tool should do, including expected inputs "
                            "and outputs."
                        ),
                    },
                    "suggested_name": {
                        "type": "string",
                        "description": (
                            "A snake_case name for the tool function "
                            "(e.g., 'calculate_compound_interest')."
                        ),
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of tags for categorizing the tool."
                        ),
                    },
                },
                "required": ["capability_description", "suggested_name"],
            },
        ),
        mcp_types.Tool(
            name="list_registered_tools",
            description=(
                "List all currently registered dynamic tools with their "
                "descriptions and optional JSON schemas. "
                "Use this to check what tools are available before deciding "
                "whether to generate a new one."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "include_schemas": {
                        "type": "boolean",
                        "description": (
                            "If true, include the JSON Schema for each tool."
                        ),
                    },
                },
                "required": [],
            },
        ),
        mcp_types.Tool(
            name="get_tool_source",
            description=(
                "Retrieve the Python source code of a registered dynamic tool. "
                "Use this to inspect or debug a generated tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the tool to retrieve.",
                    },
                },
                "required": ["name"],
            },
        ),
        mcp_types.Tool(
            name="remove_tool",
            description=(
                "Remove a registered dynamic tool from the registry. "
                "Use this to clean up tools that are no longer needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the tool to remove.",
                    },
                },
                "required": ["name"],
            },
        ),
        mcp_types.Tool(
            name="get_registry_stats",
            description=(
                "Get usage statistics for all registered tools, including "
                "call counts and last error messages."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# DynamicMCPServer
# ---------------------------------------------------------------------------

class DynamicMCPServer:
    """
    An MCP server that exposes a dynamically growing set of tools.

    Wraps a DynamicToolRegistry and a CodeGenerationPipeline. On every
    'tools/list' request, reads the current state of the registry, ensuring
    that newly generated tools are immediately visible to clients.

    Also handles the meta-tools (generate_and_register_tool, etc.) directly.
    """

    def __init__(
        self,
        registry: DynamicToolRegistry,
        code_generator: CodeGenerationPipeline,
        server_name: str = "dynamic-tool-server",
        server_version: str = "1.0.0",
    ) -> None:
        self._registry = registry
        self._code_generator = code_generator
        self._server_name = server_name
        self._server_version = server_version
        self._meta_tool_names: frozenset[str] = frozenset(
            t.name for t in make_meta_tools()
        )

        # Create the low-level MCP Server instance.
        self._server = Server(server_name)

        # Register the MCP protocol handlers.
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all MCP protocol method handlers on the low-level server."""

        @self._server.list_tools()
        async def handle_list_tools() -> list[mcp_types.Tool]:
            """
            Handle 'tools/list' requests.
            Returns meta-tools plus all dynamically registered tools.
            Called on every list request so newly registered tools appear
            immediately without any server restart.
            """
            meta_tools = make_meta_tools()
            dynamic_entries = await self._registry.get_all_tools()
            dynamic_tools = [entry_to_mcp_tool(e) for e in dynamic_entries]
            all_tools = meta_tools + dynamic_tools
            logger.debug(
                "tools/list: returning %d meta + %d dynamic tools.",
                len(meta_tools), len(dynamic_tools),
            )
            return all_tools

        @self._server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict[str, Any],
        ) -> list[mcp_types.TextContent | mcp_types.ImageContent | mcp_types.EmbeddedResource]:
            """
            Handle 'tools/call' requests.
            Routes the call to either a meta-tool handler or the registry.
            Always returns a list of MCP content objects.
            """
            logger.info(
                "tools/call: name='%s', arguments=%s", name, arguments
            )
            # Ensure arguments is always a dict.
            if arguments is None:
                arguments = {}

            try:
                if name in self._meta_tool_names:
                    result_text = await self._handle_meta_tool(name, arguments)
                else:
                    raw_result = await self._registry.call_tool(name, arguments)
                    result_text = self._format_result(raw_result)

                return [
                    mcp_types.TextContent(type="text", text=result_text)
                ]

            except KeyError as exc:
                error_text = f"ERROR: Tool not found - {exc}"
                logger.error(error_text)
                return [mcp_types.TextContent(type="text", text=error_text)]

            except asyncio.TimeoutError as exc:
                error_text = f"ERROR: Tool timed out - {exc}"
                logger.error(error_text)
                return [mcp_types.TextContent(type="text", text=error_text)]

            except Exception as exc:
                error_text = (
                    f"ERROR: Tool '{name}' raised "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.error(error_text, exc_info=True)
                return [mcp_types.TextContent(type="text", text=error_text)]

    # -----------------------------------------------------------------------
    # Meta-Tool Dispatch
    # -----------------------------------------------------------------------

    async def _handle_meta_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        """Dispatch a meta-tool call to the appropriate handler."""
        if name == "generate_and_register_tool":
            return await self._handle_generate_tool(arguments)
        elif name == "list_registered_tools":
            return await self._handle_list_tools(arguments)
        elif name == "get_tool_source":
            return await self._handle_get_tool_source(arguments)
        elif name == "remove_tool":
            return await self._handle_remove_tool(arguments)
        elif name == "get_registry_stats":
            return await self._handle_get_registry_stats()
        else:
            return f"ERROR: Unknown meta-tool '{name}'."

    async def _handle_generate_tool(
        self, arguments: dict[str, Any]
    ) -> str:
        """Handle the 'generate_and_register_tool' meta-tool."""
        capability_description = arguments.get(
            "capability_description", ""
        ).strip()
        suggested_name = arguments.get("suggested_name", "").strip()

        if not capability_description:
            return "ERROR: 'capability_description' is required."
        if not suggested_name:
            return "ERROR: 'suggested_name' is required."

        # FIX-L: normalise tags defensively.
        raw_tags = arguments.get("tags", [])
        tags: list[str] = (
            [str(t) for t in raw_tags if isinstance(t, (str, int, float))]
            if isinstance(raw_tags, list)
            else []
        )

        try:
            entry = await self._code_generator.generate_and_register(
                registry=self._registry,
                capability_description=capability_description,
                function_name=suggested_name,
                tags=tags,
            )
            return (
                f"SUCCESS: Tool '{entry.name}' has been generated and "
                f"registered.\n"
                f"Description: {entry.description}\n"
                f"Input Schema:\n"
                f"{json.dumps(entry.input_schema, indent=2)}\n"
                f"You can now call this tool using its name '{entry.name}'."
            )
        except Exception as exc:
            logger.error(
                "Tool generation failed for '%s': %s",
                suggested_name, exc, exc_info=True,
            )
            return f"ERROR: Failed to generate tool '{suggested_name}': {exc}"

    async def _handle_list_tools(self, arguments: dict[str, Any]) -> str:
        """
        Handle the 'list_registered_tools' meta-tool.

        FIX-J: actually honour the include_schemas argument — when False
        (the default), omit the input_schema from each tool's output.
        """
        include_schemas: bool = bool(arguments.get("include_schemas", False))
        entries = await self._registry.get_all_tools()

        if not entries:
            return (
                "No dynamic tools are currently registered. "
                "Use 'generate_and_register_tool' to create one."
            )

        lines: list[str] = [
            f"Registered dynamic tools ({len(entries)} total):\n"
        ]
        for entry in entries:
            lines.append(f"  • {entry.name}")
            lines.append(f"    Description: {entry.description}")
            lines.append(f"    Tags: {entry.tags or '(none)'}")
            lines.append(f"    Call count: {entry.call_count}")
            if include_schemas:
                lines.append(
                    f"    Schema: "
                    f"{json.dumps(entry.input_schema, indent=6)}"
                )
            lines.append("")

        return "\n".join(lines)

    async def _handle_get_tool_source(
        self, arguments: dict[str, Any]
    ) -> str:
        """Handle the 'get_tool_source' meta-tool."""
        name = arguments.get("name", "").strip()
        if not name:
            return "ERROR: 'name' is required."

        entry = await self._registry.get_tool(name)
        if entry is None:
            available = await self._registry.get_tool_names()
            return (
                f"ERROR: No tool named '{name}' is registered. "
                f"Available tools: {available}"
            )

        return (
            f"Source code for tool '{name}':\n\n"
            f"```python\n{entry.source_code}\n```"
        )

    async def _handle_remove_tool(self, arguments: dict[str, Any]) -> str:
        """Handle the 'remove_tool' meta-tool."""
        name = arguments.get("name", "").strip()
        if not name:
            return "ERROR: 'name' is required."

        if name in self._meta_tool_names:
            return f"ERROR: Cannot remove built-in meta-tool '{name}'."

        removed = await self._registry.remove_tool(name)
        if removed:
            return f"Tool '{name}' has been removed successfully."
        else:
            return f"ERROR: No tool named '{name}' was found."

    async def _handle_get_registry_stats(self) -> str:
        """
        Handle the 'get_registry_stats' meta-tool.

        N16: takes no arguments — consistent with the meta-tool definition.
        N4:  calls self._registry.get_stats() which is now fully implemented.
        """
        stats = await self._registry.get_stats()
        return json.dumps(stats, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Formatting
    # -----------------------------------------------------------------------

    @staticmethod
    def _format_result(result: Any) -> str:
        """
        Format a tool's return value as a string for the MCP response.

        Args:
            result: The raw return value from the tool function.

        Returns:
            A string representation suitable for the LLM to read.
        """
        if result is None:
            return "Tool completed successfully with no return value."
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            return repr(result)

    # -----------------------------------------------------------------------
    # Server Lifecycle
    # -----------------------------------------------------------------------

    def _build_initialization_options(self) -> InitializationOptions:
        """
        Build the InitializationOptions for the MCP server handshake.

        N12: get_capabilities() requires a NotificationOptions object.
        Passing None causes AttributeError on .tools_changed access.
        We try multiple signatures for SDK version compatibility.
        """
        # Try with NotificationOptions first (required by MCP SDK >= 1.0)
        try:
            capabilities = self._server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            )
        except TypeError:
            # Older SDK versions may not accept these keyword arguments.
            try:
                capabilities = self._server.get_capabilities(
                    notification_options=NotificationOptions()
                )
            except TypeError:
                capabilities = self._server.get_capabilities()

        return InitializationOptions(
            server_name=self._server_name,
            server_version=self._server_version,
            capabilities=capabilities,
        )

    async def run_with_streams(
        self,
        read_stream: MemoryObjectReceiveStream,
        write_stream: MemoryObjectSendStream,
    ) -> None:
        """
        Run the MCP server using anyio memory streams.

        This is the primary entry point for in-process usage (see main.py).

        Args:
            read_stream:  The stream the server reads client messages from.
            write_stream: The stream the server writes responses to.
        """
        init_options = self._build_initialization_options()
        logger.info(
            "Starting Dynamic MCP Server '%s' v%s with memory streams.",
            self._server_name, self._server_version,
        )
        async with anyio.create_task_group():
            await self._server.run(
                read_stream,
                write_stream,
                init_options,
            )

    async def run_stdio(self) -> None:
        """
        Run the MCP server over stdio (for use as a subprocess MCP server).
        """
        import mcp.server.stdio as mcp_stdio
        logger.info("Starting Dynamic MCP Server in stdio mode.")
        async with mcp_stdio.stdio_server() as (read_stream, write_stream):
            await self.run_with_streams(read_stream, write_stream)

