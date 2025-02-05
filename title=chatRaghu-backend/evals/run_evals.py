import json
import os
import asyncio
from pathlib import Path
from types import SimpleNamespace
from langsmith import Client
from langsmith.utils import LangSmithConflictError

# Define project_root (assuming __file__ is inside evals/)
project_root = Path(__file__).parent.parent

async def run_evaluations():
    # Verify environment variables
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise ValueError("LANGCHAIN_API_KEY not found in environment variables")
    
    client = Client()
   
    # Load test cases
    with open(project_root / "evals/datasets/test_cases.json") as f:
        test_cases = json.load(f).get("test_cases", [])
    inputs = []
    outputs = []
    for idx, case in enumerate(test_cases):
        question = case.get("question")
        answer = case.get("answer")
        inputs.append({"question": question})
        outputs.append({"answer": answer})

    # Create or retrieve the dataset
    dataset_name = "Raghu's Q&A"
    try:
        dataset = client.create_dataset(dataset_name=dataset_name)
    except LangSmithConflictError:
        # Retrieve existing dataset by name
        existing_datasets = client.get_datasets()
        dataset = next((ds for ds in existing_datasets if ds.name == dataset_name), None)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found despite conflict error.")

    # Create examples for the dataset
    client.create_examples(
        inputs=inputs, 
        outputs=outputs, 
        dataset_id=dataset.id,
    )

    # Patch client.get_examples so that examples are returned as objects (SimpleNamespace)
    original_get_examples = client.get_examples

    def patched_get_examples(dataset_id):
        examples = original_get_examples(dataset_id)
        return [SimpleNamespace(**ex) if isinstance(ex, dict) else ex for ex in examples]

    client.get_examples = patched_get_examples

    # Continue with evaluation (example: run target graph, then evaluate)
    # target_graph = await get_graph()
    # experiment_results = await client.aevaluate(
    #     target_graph,
    #     dataset=test_cases,
    #     evaluators=[RaghuPersonaEvaluator(), RelevanceEvaluator()],
    #     experiment_prefix="raghu-chat-evaluation",
    #     max_concurrency=2
    # )
    # return experiment_results

if __name__ == "__main__":
    asyncio.run(run_evaluations())