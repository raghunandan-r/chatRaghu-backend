from pathlib import Path
import json
from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI
from config import config
from utils.logger import logger

# --- Environment and Config Loading ---
project_root = Path(__file__).parent.parent  # evals-service directory
env_path = project_root / ".env"
load_dotenv(env_path)

# Define paths relative to this file's location
_PROMPTS_PATH = Path(__file__).parent / "prompts.json"
_TEMPLATES_PATH = Path(__file__).parent / "prompt_templates.json"


def _load_json_file(path: Path, description: str) -> dict:
    """Helper to load a JSON file with logging."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(
            f"Successfully loaded {description}",
            extra={"path": str(path)},
        )
        return data
    except FileNotFoundError:
        logger.warning(
            f"{description.capitalize()} file not found, using empty dictionary",
            extra={"path": str(path)},
        )
        return {}
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse {description} JSON",
            extra={"error": str(e), "path": str(path)},
        )
        return {}


# Load the prompt templates and cache them at module level
EVALUATOR_PROMPTS = _load_json_file(_PROMPTS_PATH, "evaluator prompts")
MAIN_GRAPH_PROMPTS = _load_json_file(_TEMPLATES_PATH, "main graph prompt templates")


def get_eval_prompt(evaluator_name: str, **kwargs) -> str:
    """
    Formats and returns the user prompt for a given evaluator.

    Args:
        evaluator_name: The key for the evaluator in prompts.json.
        **kwargs: The values to format into the prompt template.

    Returns:
        The formatted prompt string.
    """
    template = EVALUATOR_PROMPTS.get(evaluator_name, {}).get("user_prompt_template", "")
    if not template:
        logger.warning(f"No prompt template found for evaluator: {evaluator_name}")
        return ""
    return template.format(**kwargs)


def get_system_message(evaluator_name: str) -> str:
    """Returns the system message for a given evaluator."""
    return EVALUATOR_PROMPTS.get(evaluator_name, {}).get("system_message", "")


def get_main_graph_prompt(node_name: str) -> str:
    """Returns the original system prompt used for a node in the main graph."""
    return MAIN_GRAPH_PROMPTS.get(node_name, {}).get("system_message", "")


# --- Shared OpenAI Client ---
# Initialize and patch OpenAI client with instructor for all evaluators to use
client = instructor.from_openai(
    AsyncOpenAI(
        api_key=config.llm.openai_api_key, timeout=config.llm.openai_timeout_seconds
    )
)
