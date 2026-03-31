# codegen/code_generator.py

"""
code_generator.py

The Code Generation Pipeline: uses an LLM to generate Python tool functions
from natural language descriptions, validates them, and registers them in
the Dynamic Tool Registry.

Implements a retry loop: if the generated code fails validation, the error
messages are fed back to the LLM with a request to fix the specific problems.

Fixes applied:
  N15   — validation.function_name guarded before use (resolved_name).
  FIX-H — last_errors seeded with a default message; max_retries enforced
           to be at least 1 to prevent zero-iteration loops.
"""

import logging
from typing import Any

from openai import AsyncOpenAI

from tool_registry import DynamicToolRegistry, ToolEntry
from tool_validator import validate_tool_code, extract_python_code

logger = logging.getLogger(__name__)  # FIX-N: __name__ not bare name

# ---------------------------------------------------------------------------
# Code Generation System Prompt
# ---------------------------------------------------------------------------

CODE_GENERATION_SYSTEM_PROMPT = """\
You are an expert Python programmer specializing in writing clean, correct,
well-documented utility functions.

Your task is to generate a SINGLE Python function that implements the
requested capability.

STRICT REQUIREMENTS — your output must satisfy ALL of these:

1. Output ONLY the function definition. No imports, no explanations,
   no markdown fences, no extra text.
2. The function name MUST match the requested name exactly.
3. Every parameter MUST have a type annotation.
4. The function MUST have a return type annotation.
5. The first statement of the function body MUST be a docstring of at
   least 10 characters.
6. Do NOT use: exec, eval, compile, open, import, os, sys, subprocess,
   socket, or any other system-access module.
7. You may use: json, re, math, statistics, hashlib, base64, collections,
   itertools, functools, string, decimal, fractions, random, datetime,
   urllib_parse, requests (for HTTP).

EXAMPLE of a correctly formatted function:

def calculate_bmi(weight_kg: float, height_m: float) -> float:
    \"\"\"Calculate Body Mass Index from weight in kg and height in meters.\"\"\"
    if height_m <= 0:
        raise ValueError("Height must be positive.")
    return weight_kg / (height_m ** 2)
"""

# ---------------------------------------------------------------------------
# Code Generation Pipeline
# ---------------------------------------------------------------------------

class CodeGenerationPipeline:
    """
    Orchestrates LLM-based code generation with validation and retry.

    Calls the LLM to generate a Python function, validates it with the
    static AST validator, and registers it in the tool registry. If
    validation fails, the errors are fed back to the LLM for correction.
    """

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        model: str,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            llm_client:  An AsyncOpenAI client for code generation.
            model:       The model name to use for code generation.
            max_retries: Maximum number of generation+validation attempts.
                         FIX-H: enforced to be at least 1.
            temperature: LLM sampling temperature (low = more deterministic).
        """
        self._client = llm_client
        self._model = model
        # FIX-H: ensure at least 1 attempt even if caller passes 0.
        self._max_retries = max(1, max_retries)
        self._temperature = temperature

    async def generate_and_register(
        self,
        registry: DynamicToolRegistry,
        capability_description: str,
        function_name: str,
        tags: list[str] | None = None,
    ) -> ToolEntry:
        """
        Generate a Python tool function and register it in the registry.

        Runs a retry loop: generates code, validates it, and if validation
        fails, feeds the errors back to the LLM for correction.

        Args:
            registry:                The tool registry to register into.
            capability_description:  Natural language description of the tool.
            function_name:           The exact snake_case function name.
            tags:                    Optional tags for the registered tool.

        Returns:
            The registered ToolEntry on success.

        Raises:
            RuntimeError: If all retry attempts fail validation.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Generate a Python function named '{function_name}' "
                    f"that does the following:\n\n{capability_description}"
                ),
            },
        ]

        # FIX-H: initialise last_errors with a sensible default so the
        # RuntimeError at the bottom always has meaningful content.
        last_errors: list[str] = ["No generation attempts were made."]

        for attempt in range(1, self._max_retries + 1):
            logger.info(
                "Code generation attempt %d/%d for '%s'.",
                attempt, self._max_retries, function_name,
            )

            raw_response = await self._call_llm(messages)
            logger.debug(
                "LLM raw response (attempt %d):\n%s", attempt, raw_response
            )

            source_code = extract_python_code(raw_response)
            logger.debug("Extracted source code:\n%s", source_code)

            validation = validate_tool_code(source_code)

            if validation.is_valid:
                logger.info(
                    "Validation passed on attempt %d for tool '%s'.",
                    attempt, function_name,
                )

                # N15: Guard against None function_name (defensive — should
                # not happen if is_valid is True, but be safe).
                resolved_name = validation.function_name or function_name
                description = (
                    validation.description
                    or capability_description[:200]
                )

                entry = await registry.register_tool(
                    name=resolved_name,
                    description=description,
                    input_schema=validation.input_schema,
                    source_code=source_code,
                    tags=tags,
                )
                logger.info(
                    "Tool '%s' registered successfully.", resolved_name
                )
                return entry

            # Validation failed — record errors and ask LLM to fix them.
            last_errors = validation.errors
            error_summary = "\n".join(
                f"  - {e}" for e in validation.errors
            )
            logger.warning(
                "Validation failed on attempt %d for '%s':\n%s",
                attempt, function_name, error_summary,
            )

            # Feed errors back to the LLM for the next attempt.
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": (
                    f"The generated code has the following validation "
                    f"errors. Please fix ALL of them and return ONLY the "
                    f"corrected function definition:\n\n{error_summary}"
                ),
            })

        # All attempts exhausted.
        error_detail = "\n".join(f"  - {e}" for e in last_errors)
        raise RuntimeError(
            f"Failed to generate valid code for '{function_name}' after "
            f"{self._max_retries} attempt(s). Last errors:\n{error_detail}"
        )

    async def _call_llm(self, messages: list[dict[str, Any]]) -> str:
        """
        Call the LLM and return the response content as a string.

        Args:
            messages: The conversation history to send.

        Returns:
            The LLM's response text.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        return content or ""
