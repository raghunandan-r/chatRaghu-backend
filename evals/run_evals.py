import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from types import SimpleNamespace
from evals.evaluators import correctness, groundedness, relevance, retrieval_relevance
from langchain_core.messages import ToolMessage, AIMessage
from graph.graph import get_graph 
from utils.logger import logger
from langsmith.utils import LangSmithConflictError
import asyncio

# Load environment variables from .env file
# Get the project root directory (assuming evals is in project root)
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

async def run_evaluations():
    # Verify necessary environment variables are loaded
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise ValueError("LANGCHAIN_API_KEY not found in environment variables")
    
    client = Client()
    examples = [
    (
        "Tell me about yourself.",
        "Raghunandan is a results-driven analytics and AI professional based in New York City, holding a Master's degree in Data Science from RIT. With a robust focus on Explainable AI, he has consistently delivered substantial business outcomes across various positions. Recently, during his tenure as a RevOps intern at Micron Technology in Boise, ID, Raghunandan automated Salesforce data analyses, directly contributing to a remarkable $5 million increase in quarterly revenue. His experience extends to developing Go-to-Market strategies for enterprise SaaS at Freshworks Inc., conducting operations research in hyperlocal delivery at Lynk Logistics, and enhancing financial access at Bankbazaar.com. Proficient in Python, JavaScript, SQL, and AWS, Raghunandan is poised to leverage his expertise to drive data-driven impact in a dynamic organization."
    ),
    (
        "do you have any internship experience",
        "Raghunandan interned at Micron Technology as a Revenue & Demand Operations Analyst Intern from May 2024 to August 2024. During this impactful internship, Raghu automated analysis from Salesforce CRM data, leading to a remarkable $5 million quarterly increase in accurately allocated revenue. Additionally, he developed real-time Tableau dashboards that significantly reduced the end-of-quarter workload by 12 hours, showcasing his ability to drive efficiency and results."
    ),
    (
        "what is your experience building with LLMs?",
        "Raghunandan has successfully spearheaded several impactful projects that highlight his technical acumen and commitment to enhancing user experience:\n\n                  1. **A Better Bing: Interactive Search Engine** (Dec 2023)\n                     - Raghunandan developed an innovative interactive search engine leveraging active learning and natural language processing techniques.\n                     - He engineered a linearSVM classifier and implemented dot-product similarity algorithms from scratch using JavaScript.\n                     - The results were impressive, with a 3x improvement in click-through rates, demonstrating a significant boost in user engagement.\n                     - [GitHub Link](https://github.com/raghunandan-r/FastSeach_DSCI_Capstone)\n                      \n                  2. **chatRaghu - Chat with my AI Assistant** (Jan 2025)\n                     - Raghunandan built a sophisticated chat application utilizing next.js, which integrates a large language model (LLM) with retrieval augmented generation (RAG) to address user inquiries about his profile and projects.\n                     - He established a robust backend using FastAPI and enabled agentic behavior through Langchain, with data efficiently stored in Pinecone and ChromaDB vectorstore.\n                     - [GitHub Link](https://github.com/raghunandan-r/raghunandan-r.github.io)\n                      \n                  3. **LLMOps** (Nov 2024)\n                     - Raghunandan implemented a finely-tuned LLM pipeline, which encompassed batch preprocessing, distributed fine-tuning (LoRA), and batch evaluation processes.\n                     - This initiative led to a remarkable 50% reduction in inference time, paving the way for substantial operational cost savings.\n                     - [GitHub Link](https://github.com/raghunandan-r/Made-With-ML)\n                      \n                  These projects clearly illustrate Raghunandan's expertise in NLP, LLMs, and web application development, with a steadfast focus on enhancing performance and driving user engagement."
    ),
    (
        "where did you do your studies?",
        "Raghunandan pursued a master's degree in Data Science at the prestigious Rochester Institute of Technology. This academic foundation has equipped Raghu with the skills necessary to excel in the data-driven landscape."
    )
    ]

    # Load test cases
    dataset_name = "Raghu's Q&A"
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)        
    else:
        dataset = client.create_dataset(dataset_name=dataset_name)        
        client.create_examples(
            inputs=[{"question": q} for q, _ in examples], 
            outputs=[{"answer": a} for _, a in examples], 
            dataset_id=dataset.id,
        )

    async def run_graph(inputs: dict) -> dict:
        """
        Run graph and return the final response along with retrieved documents.
        
        Assumes the graph returns a dict containing:
          - "followup": the final response text.
          - "documents": a list of document objects (with at least a 'content' or 'page_content' attribute)
        """
        graph = await get_graph()  # get_graph may be async even if internal compile is sync
        result = await graph.ainvoke(
            {"messages": [{ "role": "user", "content": inputs['question']} ]}
            #config={"env": "test"}
        )
        
        # Retrieve documents from the state via ToolMessage instances
        retrieved_docs = [
            {"page_content": msg.content} 
            for msg in result.get("messages", []) 
            if isinstance(msg, ToolMessage)
        ]

        last_ai_msg = [msg for msg in result.get("messages", []) if isinstance(msg, AIMessage)][-1]
        return {"answer": last_ai_msg.content, "documents": retrieved_docs}

    experiment_results = await client.aevaluate(
        run_graph,
        data=dataset_name,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix="rag-doc-relevance",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )
    # Explore results locally as a dataframe if you have pandas installed
    # experiment_results.to_pandas()
    return experiment_results

if __name__ == "__main__":
    asyncio.run(run_evaluations()) 