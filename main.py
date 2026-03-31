# codegen/main.py

"""
main.py

Entry point for the Self-Extending AI System.

Wires together all components:

- The Dynamic Tool Registry
- The Code Generation Pipeline
- The Dynamic MCP Server
- The LLM Agent

Uses anyio memory streams for in-process MCP client-server communication,
avoiding subprocess overhead while maintaining full MCP protocol semantics.

Configuration via environment variables or command-line arguments:
  LLM_BASE_URL  -- Base URL for the LLM API (default: http://localhost:11434/v1)
  LLM_API_KEY   -- API key (default: 'ollama' for local Ollama)
  LLM_MODEL     -- Model name (default: qwen2.5:14b)
  TOOLS_FILE    -- JSON file for tool persistence (default: tools.json)
  LOG_LEVEL     -- Logging level (default: INFO)

Fixes applied:
  N14   — Stream type imports kept at top level (needed for anyio unpacking);
           guarded with TYPE_CHECKING for pure type-hint contexts.
  FIX-E — create_memory_object_stream uses max_buffer_size= keyword arg.
  FIX-N — logging.getLogger(__name__) everywhere.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
# N14: These imports are needed at runtime for the stream unpacking below.
# They are also used as type hints. Kept at top level — the anyio.streams.memory
# submodule is stable across anyio >= 3.0.
from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
)
from mcp.client.session import ClientSession
from openai import AsyncOpenAI

from agent import SelfExtendingAgent
from code_generator import CodeGenerationPipeline
from mcp_server import DynamicMCPServer
from tool_registry import DynamicToolRegistry

logger = logging.getLogger(__name__)  # FIX-N: __name__ not bare name

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# ---------------------------------------------------------------------------
# Tool Persistence Helpers
# ---------------------------------------------------------------------------

async def load_persisted_tools(
    registry: DynamicToolRegistry,
    tools_file: Path,
) -> None:
    """Load previously generated tools from a JSON file into the registry."""
    if not tools_file.exists():
        logger.info(
            "No persisted tools file found at '%s'. Starting fresh.",
            tools_file,
        )
        return

    try:
        with tools_file.open("r", encoding="utf-8") as f:
            tool_defs = json.load(f)

        if not isinstance(tool_defs, list):
            logger.warning(
                "Tools file '%s' has invalid format (expected a list). "
                "Skipping.",
                tools_file,
            )
            return

        success, failure = await registry.import_tools(tool_defs)
        logger.info(
            "Loaded %d tools from '%s' (%d failed).",
            success, tools_file, failure,
        )
    except Exception as exc:
        logger.error(
            "Failed to load persisted tools from '%s': %s", tools_file, exc,
            exc_info=True,
        )

async def save_persisted_tools(
    registry: DynamicToolRegistry,
    tools_file: Path,
) -> None:
    """Save all registered tools to a JSON file for persistence."""
    try:
        tool_defs = await registry.export_tools()
        with tools_file.open("w", encoding="utf-8") as f:
            json.dump(tool_defs, f, indent=2, default=str)
        logger.info(
            "Saved %d tools to '%s'.", len(tool_defs), tools_file
        )
    except Exception as exc:
        logger.error(
            "Failed to save tools to '%s': %s", tools_file, exc,
            exc_info=True,
        )

# ---------------------------------------------------------------------------
# Run Modes
# ---------------------------------------------------------------------------

async def run_single_query(
    agent: SelfExtendingAgent,
    registry: DynamicToolRegistry,
    tools_file: Path,
    query: str,
) -> None:
    """Process a single query and print the result."""
    print(f"Query:\n{query}\n")
    print("Processing...\n")

    try:
        response, _ = await agent.run(user_message=query)
        print(f"Response:\n{response}")
        await save_persisted_tools(registry, tools_file)
    except Exception as exc:
        print(f"Error: {exc}")
        logger.error("Agent error: %s", exc, exc_info=True)
        sys.exit(1)

async def run_interactive(
    agent: SelfExtendingAgent,
    registry: DynamicToolRegistry,
    tools_file: Path,
) -> None:
    """
    Run an interactive REPL where the user can chat with the agent.

    The conversation history is maintained across turns. Tools are saved
    to disk after each turn. The user can type 'exit', 'quit', 'clear',
    or 'stats' as special commands.
    """
    print("\n" + "=" * 70)
    print("  Self-Extending AI System -- Interactive Mode")
    print("=" * 70)
    print("  Type your message and press Enter. The agent can generate")
    print("  new tools on demand to fulfill your requests.")
    print("  Commands: 'exit'/'quit' to stop, 'clear' to reset history,")
    print("            'stats' to show registry statistics.")
    print("=" * 70 + "\n")

    conversation_history: list[dict[str, Any]] | None = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_history = None
            print("[Conversation history cleared.]\n")
            continue

        if user_input.lower() == "stats":
            stats = await registry.get_stats()
            print(json.dumps(stats, indent=2, default=str))
            print()
            continue

        print("\nAgent: ", end="", flush=True)

        try:
            response, conversation_history = await agent.run(
                user_message=user_input,
                conversation_history=conversation_history,
            )
            print(response)
            print()

            # Save tools after each successful turn.
            await save_persisted_tools(registry, tools_file)

        except Exception as exc:
            print(f"[Error: {exc}]")
            logger.error("Agent error: %s", exc, exc_info=True)
            print()

# ---------------------------------------------------------------------------
# System Wiring
# ---------------------------------------------------------------------------

async def run_system(config: dict[str, Any]) -> None:
    """
    Set up and run the full self-extending AI system.

    Creates the MCP server and client in the same process, connected via
    anyio memory streams. The server runs as a background task in a task
    group; the client session is used by the agent for all MCP communication.
    """
    logger.info("Starting Self-Extending AI System.")
    logger.info(
        "LLM: %s @ %s", config["llm_model"], config["llm_base_url"]
    )
    logger.info("Tools file: %s", config["tools_file"])

    # --- Create the LLM client ---
    llm_client = AsyncOpenAI(
        base_url=config["llm_base_url"],
        api_key=config["llm_api_key"],
    )

    # --- Create the tool registry ---
    registry = DynamicToolRegistry(call_timeout=60.0)

    # --- Load persisted tools ---
    await load_persisted_tools(registry, config["tools_file"])

    # --- Create the code generation pipeline ---
    code_generator = CodeGenerationPipeline(
        llm_client=llm_client,
        model=config["llm_model"],
        max_retries=3,
        temperature=0.1,
    )

    # --- Create the MCP server ---
    mcp_server = DynamicMCPServer(
        registry=registry,
        code_generator=code_generator,
        server_name="self-extending-ai",
        server_version="1.0.0",
    )

    # --- Create anyio memory streams for in-process communication ---
    # FIX-E: use explicit keyword argument max_buffer_size= (required in
    # anyio >= 4.0; positional call silently broke in some versions).
    (
        client_to_server_send,
        client_to_server_recv,
    ) = anyio.create_memory_object_stream(max_buffer_size=1)
    (
        server_to_client_send,
        server_to_client_recv,
    ) = anyio.create_memory_object_stream(max_buffer_size=1)

    async with anyio.create_task_group() as tg:
        # Start the MCP server as a background task.
        tg.start_soon(
            mcp_server.run_with_streams,
            client_to_server_recv,
            server_to_client_send,
        )

        # Create the MCP client session.
        async with ClientSession(
            server_to_client_recv,
            client_to_server_send,
        ) as session:
            # Perform the MCP initialization handshake.
            await session.initialize()
            logger.info("MCP client session initialized.")

            # Create the agent with the initialized session.
            agent = SelfExtendingAgent(
                llm_client=llm_client,
                model=config["llm_model"],
                mcp_session=session,
                temperature=0.7,
            )

            # Run in the selected mode.
            if config["query"]:
                await run_single_query(
                    agent, registry, config["tools_file"], config["query"]
                )
            else:
                await run_interactive(agent, registry, config["tools_file"])

        # After the client session context exits, cancel the server task.
        tg.cancel_scope.cancel()

    logger.info("Self-Extending AI System stopped.")

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the runtime configuration dict from CLI args and env vars."""
    return {
        "llm_base_url": (
            args.base_url
            or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        ),
        "llm_api_key": (
            args.api_key
            or os.environ.get("LLM_API_KEY", "ollama")
        ),
        "llm_model": (
            args.model
            or os.environ.get("LLM_MODEL", "qwen2.5:14b")
        ),
        "tools_file": Path(
            args.tools_file
            or os.environ.get("TOOLS_FILE", "tools.json")
        ),
        "query": args.query or None,
        "log_level": (
            args.log_level
            or os.environ.get("LOG_LEVEL", "INFO")
        ),
    }

def main() -> None:
    """Parse CLI arguments and launch the system."""
    parser = argparse.ArgumentParser(
        description="Self-Extending AI System — an LLM agent that generates "
                    "its own tools on demand via the MCP protocol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="LLM API base URL (default: $LLM_BASE_URL or "
             "http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help="LLM API key (default: $LLM_API_KEY or 'ollama')",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="LLM model name (default: $LLM_MODEL or qwen2.5:14b)",
    )
    parser.add_argument(
        "--tools-file",
        metavar="FILE",
        help="JSON file for tool persistence (default: $TOOLS_FILE or "
             "tools.json)",
    )
    parser.add_argument(
        "--query",
        "-q",
        metavar="QUERY",
        help="Run a single query and exit (omit for interactive mode)",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: $LOG_LEVEL or INFO)",
    )

    args = parser.parse_args()
    config = build_config(args)

    setup_logging(config["log_level"])
    logger.info("Self-Extending AI System starting up.")

    try:
        asyncio.run(run_system(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
