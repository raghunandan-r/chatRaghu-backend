"""
Template rendering utilities for prompt generation.

This module handles loading and rendering prompt templates with variable injection
and few-shot example formatting.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import MessagesState, HumanMessage, AIMessage, ToolMessage

TEMPLATES_PATH = Path(__file__).parent / "prompt_templates.json"
TEMPLATES = json.load(open(TEMPLATES_PATH, "r", encoding="utf-8"))


def get_graph_version(graph_type: str) -> str:
    """Gets the graph version for a specific graph type."""
    return TEMPLATES.get(graph_type, {}).get("version", "unknown")


def build_conversation_history(
    state: MessagesState, max_history=24, include_tool_content=True
):
    recent_messages = (
        state.messages[-max_history:]
        if len(state.messages) > max_history
        else state.messages
    )
    openai_messages = []

    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})

        elif isinstance(msg, AIMessage):
            openai_messages.append({"role": "assistant", "content": msg.content})

        elif (
            include_tool_content
            and isinstance(msg, ToolMessage)
            and msg.tool_name == "retrieve"
        ):
            # Only include the retrieved content, not the tool call details
            if isinstance(msg.output, list) and all(
                hasattr(item, "content") and hasattr(item, "score")
                for item in msg.output
            ):
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": "\n\n".join(
                            f"Content: {item.content} (Score: {item.score:.2f})"
                            for item in msg.output
                        ),
                    }
                )
            elif isinstance(msg.output, str):
                openai_messages.append({"role": "assistant", "content": msg.output})
            else:
                openai_messages.append(
                    {"role": "assistant", "content": str(msg.output)}
                )

    # logger.info("DEBUG: openai_messages", extra={"openai_messages": openai_messages})
    return openai_messages


def render_system_prompt_(
    *,
    user_query: str,
    name: str,
    graph_type: str,
    decision: str = "default",
    language: str = "english",
) -> str:
    """
    Render the generate_answer prompt with base system + mode-specific partial.

    Args:
        user_query: The user's query
        name: The name of the adapter
        graph_type: The type of graph being used
        decision: The decision made by the router
        language: The language for the response
        sparse_search_context: Context from sparse search for router decisions

    Returns:
        Rendered system prompt
    """
    ga = TEMPLATES[graph_type][name]
    base = ga["base_system"]

    # Router template has no partials, just base system prompt
    if name == "router":
        partial = ""
    else:
        if decision not in ga["partials"]:
            raise KeyError(f"Decision {decision} not found in {name} partials")
        partial = ga["partials"][decision]

    current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content_vars = {
        "current_date_str": current_date_str,
        "user_query": user_query,
        "language": language,
    }

    # Simple brace-style substitution
    def fmt(s: str) -> str:
        return s.format(**content_vars)

    # Only append partial if it exists (for non-router templates)
    rendered = fmt(base)
    if partial:
        rendered += "\n\n" + fmt(partial)
    return rendered
