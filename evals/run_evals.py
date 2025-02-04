import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from evals.evaluators import correctness, groundedness, relevance, retrieval_relevance
from graph.graph import get_graph 
from utils.logger import logger
# Load environment variables from .env file
# Get the project root directory (assuming evals is in project root)
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)




async def run_evaluations():
    # Verify environment variables are loaded
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
    dataset = client.create_dataset(dataset_name="Raghu's Q&A")
    client.create_examples(
        inputs=inputs, 
        outputs=outputs, 
        dataset_id=dataset.id,
    )      
    
    async def run_graph(inputs: dict) -> dict:
        """Run graph and track the trajectory it takes along with the final response."""
        result = await get_graph().ainvoke({"messages": [
            { "role": "user", "content": inputs['question']},
        ]}, config={"env": "test"})
        return {"response": result["followup"]}

    experiment_results = client.evaluate(
        run_graph,
        data=test_cases,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix="rag-doc-relevance",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )
    # Explore results locally as a dataframe if you have pandas installed
    # experiment_results.to_pandas()
    return experiment_results

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_evaluations()) 