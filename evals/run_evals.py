import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from evals.evaluators import RaghuPersonaEvaluator, RelevanceEvaluator
import graph.graph as graph  # Import your existing graph

# Load environment variables from .env file
# Get the project root directory (assuming evals is in project root)
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)


async def run_graph()


async def run_evaluations():
    # Verify environment variables are loaded
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise ValueError("LANGCHAIN_API_KEY not found in environment variables")
    
    client = Client()
    
    # Debug prints
    print("Graph object type:", type(graph))
    print("Graph object:", graph)
    
    # Load test cases
    with open(project_root / "evals/datasets/test_cases.json") as f:
        test_cases = json.load(f)["test_cases"]
    
    evaluators = [RaghuPersonaEvaluator(), RelevanceEvaluator()]
    
    # Debug print before evaluate
    print("About to call evaluate with:", {
        "target": graph,
        "dataset": test_cases,
        "evaluators": evaluators
    })
    
    experiment_results = client.evaluate(
        target=graph,  # Your chat implementation
        dataset=test_cases,
        evaluators=evaluators,
        experiment_prefix="raghu-chat-evaluation",
        max_concurrency=2
    )
    
    return experiment_results

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_evaluations()) 