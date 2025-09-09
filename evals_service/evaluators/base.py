from pathlib import Path
from config import config
import json
from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI
from utils.logger import logger

# --- Environment and Config Loading ---
project_root = Path(__file__).parent.parent  # evals-service directory
env_path = project_root / ".env"
load_dotenv(env_path)

# Define paths relative to this file's location
_PROMPTS_PATH = Path(__file__).parent / "prompts.json"


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


def get_eval_prompt(evaluator_name: str, graph_version: str, **kwargs) -> str:
    """
    Formats and returns the user prompt for a given evaluator.

    Args:
        evaluator_name: The key for the evaluator in prompts.json.
        graph_version: The graph version to determine which prompt section to use.
        **kwargs: The values to format into the prompt template.

    Returns:
        The formatted prompt string.
    """
    # Determine which section to use based on graph_version
    if graph_version.startswith("i"):
        section = "immi"
    elif graph_version.startswith("v"):
        section = "resume"
    else:
        logger.warning(
            f"Unknown graph_version: {graph_version}, defaulting to 'resume'"
        )
        section = "resume"

    template = (
        EVALUATOR_PROMPTS.get(section, {})
        .get(evaluator_name, {})
        .get("user_prompt_template", "")
    )
    if not template:
        logger.warning(
            f"No prompt template found for evaluator: {evaluator_name} in section: {section}"
        )
        return ""
    return template.format(**kwargs)


def get_system_message(evaluator_name: str, graph_version: str) -> str:
    """
    Returns the system message for a given evaluator.

    Args:
        evaluator_name: The key for the evaluator in prompts.json.
        graph_version: The graph version to determine which prompt section to use.

    Returns:
        The system message string.
    """
    # Determine which section to use based on graph_version
    if graph_version.startswith("i"):
        section = "immi"
    elif graph_version.startswith("v"):
        section = "resume"
    else:
        logger.warning(
            f"Unknown graph_version: {graph_version}, defaulting to 'resume'"
        )
        section = "resume"

    return (
        EVALUATOR_PROMPTS.get(section, {})
        .get(evaluator_name, {})
        .get("system_message", "")
    )


# --- Shared OpenAI Client ---
# Initialize and patch OpenAI client with instructor for all evaluators to use
client = instructor.from_openai(
    AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.llm.openai_api_key,
        default_headers={
            "HTTP-Referer": "https://chatraghu-backend.ai",
            "X-Title": "chatraghu-backend",
        },
    )
)
