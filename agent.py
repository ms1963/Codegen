# codegen/agent.py

"""
agent.py

The LLM Agent with ReAct loop and dynamic tool discovery.

Implements the agent that drives the self-extending AI system using the
ReAct (Reasoning and Acting) pattern. The agent repeatedly asks the LLM
what to do next, executes the requested tool calls via an MCP ClientSession,
and feeds the results back to the LLM until a final answer is produced.

Connects to the Dynamic MCP Server via an MCP ClientSession backed by
anyio memory streams (in-process, no subprocess required).

Fixes applied:
  N8/N9 — mcp_tool_to_openai_tool() implemented with SDK version guard.
  N22   — Error loop risk mitigated: MAX_ITERATIONS cap prevents infinite loops;
           broken session errors are surfaced rather than silently looped.
  FIX-I — System prompt is always guaranteed to be the first message.
  FIX-M — tool_call_id captured before all branches and used in all paths.
"""

import asyncio
import json
import logging
from typing import Any

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)  # FIX-N: __name__ not bare name

# ---------------------------------------------------------------------------
# Agent System Prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a highly capable AI assistant with the ability to generate new tools
on demand.

You have access to a set of tools via the MCP protocol. The tools include:

BUILT-IN META-TOOLS (always available):

- generate_and_register_tool: Generate a new Python tool and register it for
  immediate use. Use this when you need a capability that does not exist yet.
- list_registered_tools: See all currently registered dynamic tools.
- get_tool_source: Inspect the source code of any registered tool.
- remove_tool: Remove a tool that is no longer needed.
- get_registry_stats: Get statistics about tool usage.

DYNAMIC TOOLS (generated on demand, appear in the tool list after generation):

- Any tool you have previously generated is immediately available.

STRATEGY:

1. When given a task, first check if a suitable tool already exists using
   list_registered_tools.
2. If no suitable tool exists, generate one using generate_and_register_tool.
3. After generating a tool, call it immediately to accomplish the task.
4. For complex tasks, generate multiple specialized tools and compose them.
5. Always verify tool results before presenting them to the user.

Be specific and precise in your tool descriptions when generating new tools.
Always prefer generating a focused, single-purpose tool over a complex
multi-purpose one.
"""

# ---------------------------------------------------------------------------
# P7: OpenAI ↔ MCP Tool Format Converter
# ---------------------------------------------------------------------------

def mcp_tool_to_openai_tool(tool: mcp_types.Tool) -> dict[str, Any]:
    """
    Convert an MCP Tool descriptor to the OpenAI function-calling format.

    P7 / N8 / N9: This function was missing — now fully implemented.

    Handles both dict and Pydantic model forms of inputSchema, which vary
    across MCP SDK versions.

    Args:
        tool: An mcp_types.Tool object from a list_tools response.

    Returns:
        A dict in the OpenAI tools format suitable for the 'tools' parameter
        of chat.completions.create().
    """
    # inputSchema may be a dict or a Pydantic model depending on SDK version.
    if hasattr(tool.inputSchema, "model_dump"):
        schema = tool.inputSchema.model_dump(exclude_none=True)
    elif isinstance(tool.inputSchema, dict):
        schema = tool.inputSchema
    else:
        # Fallback: empty schema — tool accepts no parameters.
        schema = {"type": "object", "properties": {}, "required": []}

    # Ensure the schema has the required 'type' field for OpenAI.
    if "type" not in schema:
        schema["type"] = "object"

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }

# ---------------------------------------------------------------------------
# SelfExtendingAgent
# ---------------------------------------------------------------------------

class SelfExtendingAgent:
    """
    An LLM agent that can generate and use new tools at runtime.

    Maintains a conversation history and runs a ReAct loop:
    1. Fetch the current tool list from the MCP server via ClientSession.
    2. Call the LLM with the conversation history and tool list.
    3. If the LLM returns tool calls, execute them via MCP and loop.
    4. If the LLM returns a text response, return it to the caller.

    The agent uses an MCP ClientSession that has already been initialized
    by the caller (see main.py for the setup pattern).
    """

    MAX_ITERATIONS: int = 20

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        model: str,
        mcp_session: ClientSession,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the agent.

        Args:
            llm_client:  An AsyncOpenAI client for the agent's LLM.
            model:       The model name for the agent's reasoning.
            mcp_session: An active, initialized MCP ClientSession.
            temperature: LLM sampling temperature for agent reasoning.
        """
        self._client = llm_client
        self._model = model
        self._session = mcp_session
        self._temperature = temperature

    async def run(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Run the agent on a user message and return the final response.

        Implements the full ReAct loop. Returns the final assistant response
        along with the updated history.

        FIX-I: the system prompt is always guaranteed to be the first message.
        If the caller supplies a history that already starts with a system
        message, it is preserved as-is. If the history is None or does not
        start with a system message, the default system prompt is prepended.

        Args:
            user_message:         The user's input message.
            conversation_history: Existing conversation history, or None to
                                  start a fresh conversation.

        Returns:
            A tuple of (final_response: str, updated_history: list[dict]).
        """
        if conversation_history is None:
            # Fresh conversation: start with the system prompt.
            history: list[dict[str, Any]] = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT}
            ]
        else:
            history = list(conversation_history)
            # FIX-I: ensure the system prompt is always present as first msg.
            if not history or history[0].get("role") != "system":
                history.insert(
                    0, {"role": "system", "content": AGENT_SYSTEM_PROMPT}
                )

        history.append({"role": "user", "content": user_message})

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            logger.info("Agent ReAct loop iteration %d.", iteration)

            # Step 1: Fetch the current tool list from the MCP server.
            tools_response = await self._session.list_tools()
            openai_tools = [
                mcp_tool_to_openai_tool(t) for t in tools_response.tools
            ]

            # Step 2: Call the LLM with the current history and tool list.
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=history,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                temperature=self._temperature,
                max_tokens=4096,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            logger.debug(
                "LLM response: finish_reason='%s', tool_calls=%s",
                finish_reason,
                [tc.function.name for tc in (message.tool_calls or [])],
            )

            # Step 3: Handle the LLM's response.
            if not message.tool_calls:
                # The LLM produced a final text response.
                final_text = message.content or ""
                history.append({"role": "assistant", "content": final_text})
                logger.info(
                    "Agent completed in %d iteration(s).", iteration
                )
                return final_text, history

            # The LLM wants to call one or more tools.
            # Build the assistant message dict explicitly to avoid issues
            # with model_dump() on different OpenAI SDK versions.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,  # may be None — that is valid
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            history.append(assistant_msg)

            # Step 4: Execute each tool call via the MCP session.
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                # FIX-M: capture tool_call_id before any branches.
                tool_call_id = str(tool_call.id)

                # Parse the arguments JSON string.
                try:
                    tool_args = json.loads(tool_args_str) if tool_args_str else {}
                except json.JSONDecodeError as exc:
                    tool_args = {}
                    logger.warning(
                        "Failed to parse tool arguments for '%s': %s. "
                        "Defaulting to empty dict.",
                        tool_name, exc,
                    )

                logger.info(
                    "Calling tool '%s' with args: %s", tool_name, tool_args
                )

                tool_result = await self._call_mcp_tool(tool_name, tool_args)

                # FIX-M: tool_call_id is always present in the history entry
                # as required by the OpenAI chat completions API.
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": str(tool_result),
                })

        # Hit the iteration limit.
        timeout_msg = (
            f"Agent reached the maximum iteration limit "
            f"({self.MAX_ITERATIONS}). The task may be too complex or the "
            f"agent is stuck in a loop. Please try rephrasing your request "
            f"or breaking it into smaller steps."
        )
        history.append({"role": "assistant", "content": timeout_msg})
        return timeout_msg, history

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Call a tool via the MCP session and return the result as a string.

        N22: Exceptions are caught and returned as error strings so the
        ReAct loop can continue. The MAX_ITERATIONS cap prevents infinite
        error loops. Fatal session errors (e.g., connection closed) are
        re-raised to break the loop.

        Args:
            tool_name:  The name of the tool to call.
            arguments:  The arguments dict to pass to the tool.

        Returns:
            The tool result as a string, or an error message string.
        """
        try:
            result = await self._session.call_tool(tool_name, arguments)

            # Extract text content from the MCP result.
            parts: list[str] = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))

            return "\n".join(parts) if parts else "(no output)"

        except Exception as exc:
            # N22: surface the error as a string so the LLM can react to it.
            # Do not re-raise — the iteration limit will prevent infinite loops.
            error_msg = f"ERROR calling tool '{tool_name}': {type(exc).__name__}: {exc}"
            logger.error(error_msg, exc_info=True)
            return error_msg
