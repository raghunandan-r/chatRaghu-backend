"""
Template rendering utilities for prompt generation.

This module handles loading and rendering prompt templates with variable injection
and few-shot example formatting.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from utils.logger import logger
from .models import MessagesState, HumanMessage, AIMessage, ToolMessage

TEMPLATES_PATH = Path(__file__).parent / "prompt_templates.json"


def _load_templates() -> Dict[str, Any]:
    """Load prompt templates from JSON file."""
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_conversation_history(
    state: MessagesState, max_history=8, include_tool_content=True
):
    recent_messages = (
        state.messages[-max_history * 2 :]
        if len(state.messages) > max_history * 2
        else state.messages
    )
    openai_messages = []
    
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})
            
        elif isinstance(msg, AIMessage):
            openai_messages.append({"role": "assistant", "content": msg.content})
        
        elif (include_tool_content and isinstance(msg, ToolMessage) and msg.tool_name == "retrieve"):
            # Only include the retrieved content, not the tool call details            
            if isinstance(msg.output, list) and all(
                hasattr(item, "content") and hasattr(item, "score") for item in msg.output
            ):
                openai_messages.append({"role": "assistant", "content": "\n\n".join(
                    f"Content: {item.content} (Score: {item.score:.2f})" for item in msg.output
                )})
            elif isinstance(msg.output, str):
                openai_messages.append({"role": "assistant", "content": msg.output})
            else:
                openai_messages.append({"role": "assistant", "content": str(msg.output)})
    
    logger.info("DEBUG: openai_messages", extra={"openai_messages": openai_messages})
    return openai_messages


def render_prompt_generate_answer(
    mode: str, 
    *, 
    user_query: str, 
    deflection_category: str = ""
) -> str:
    """
    Render the generate_answer prompt with base system + mode-specific partial.
    
    Args:
        mode: One of "deflection", "rag", "history"
        user_query: The user's query
        deflection_category: Classification category (for deflection mode)
    
    Returns:
        Rendered system prompt
    """
    templates = _load_templates()
    ga = templates["generate_answer"]
    base = ga["base_system"]
    partial = ga["partials"][mode]
    
    current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content_vars = {
        "current_date_str": current_date_str,
        "user_query": user_query,
        "deflection_category": deflection_category,
    }
    
    # Simple brace-style substitution
    def fmt(s: str) -> str:
        return s.format(**content_vars)
    
    return fmt(base) + "\n\n" + fmt(partial)


def render_deflection_categorizer(user_query: str, examples: list[dict] | None = None) -> str:
    """
    Render the deflection categorizer prompt with few-shot examples.
    
    Args:
        user_query: The user's query to classify
    
    Returns:
        Rendered system prompt with examples
    """
    templates = _load_templates()
    tpl = templates["deflection_categorizer"]
    sys = tpl["system_message"].format(user_query=user_query)
    # Use dynamic examples if provided; otherwise fall back to static few_shots from templates
    few_shots = examples if examples is not None and len(examples) > 0 else tpl.get("few_shots", [])
    
    # Append few-shots in a simple labeled format
    examples = "\n".join([
        f"Q: {ex['user_query']}\nLabel: {ex['label']}" 
        for ex in few_shots
    ])
    
    return f"{sys}\n{examples}"


def render_simple_template(template_name: str, **variables) -> str:
    """
    Render a simple template (relevance_check, query_or_respond) with variables.
    
    Args:
        template_name: Name of the template to render
        **variables: Variables to inject into the template
    
    Returns:
        Rendered template
    """
    templates = _load_templates()
    template = templates[template_name]
    
    if "system_message" in template:
        return template["system_message"].format(**variables)
    else:
        raise KeyError(f"Template {template_name} does not have a system_message field")

# TODO: this isnt used anywhere. remove?
def validate_templates() -> None:
    """
    Validate that all required templates and fields exist.
    
    Raises:
        KeyError: If required templates or fields are missing
    """
    templates = _load_templates()
    required = [
        ("relevance_check", ["system_message"]),
        ("query_or_respond", ["system_message"]),
        ("deflection_categorizer", ["system_message", "few_shots"]),
        ("generate_answer", ["base_system", "partials"]),
    ]
    
    for key, fields in required:
        if key not in templates:
            raise KeyError(f"Missing template group: {key}")
        for field in fields:
            if field not in templates[key]:
                raise KeyError(f"Missing field {field} in template group {key}")
    
    # Validate generate_answer partials
    ga_partials = templates["generate_answer"]["partials"]
    required_partials = ["deflection", "rag", "history"]
    for partial in required_partials:
        if partial not in ga_partials:
            raise KeyError(f"Missing partial {partial} in generate_answer.partials")
