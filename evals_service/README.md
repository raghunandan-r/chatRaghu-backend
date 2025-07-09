# ChatRaghu Evaluation Service

The ChatRaghu Evaluation Service is a FastAPI application designed for robust, asynchronous, and extensible evaluation of conversation graphs. It features a decoupled dual-queue system, modular storage backends, and a plug-in architecture for adding new evaluation metrics.

## Key Features

- **Dual-Queue System**: Separate queues for audit logging and evaluation processing ensure data integrity and resilience. Raw requests are logged before processing, and evaluation failures do not impact the audit trail.
- **Asynchronous Processing**: Evaluation requests are handled in the background, allowing the main service to respond quickly. A synchronous endpoint is also available for testing.
- **Modular Storage**: Supports multiple storage backends, including local filesystem and Google Cloud Storage (GCS), for both audit logs and evaluation results. Data is written in batches for efficiency.
- **Extensible Evaluators**: A plug-in architecture makes it easy to add new evaluators for different nodes in the conversation graph without changing the core service logic.
- **Comprehensive Health Monitoring**: Endpoints for health checks and metrics provide deep visibility into the status of all service components.

### evals-service structure for reference:

├── __init__.py
├── app.py
├── audit_data
│   └── audit_data
├── config.py
├── entrypoint.sh
├── eval_results
│   └── eval_results
├── evaluators
│   ├── __init__.py
│   ├── base.py
│   ├── generate_with_context.py
│   ├── generate_with_persona.py
│   ├── judgements.py
│   ├── models.py
│   ├── prompts.json
│   ├── prompts.py
│   └── relevance_check.py
├── evaluators.py
├── models.py
├── prompt_templates.json
├── queue_manager.py
├── railway.json
├── requirements.txt
├── run_evals.py
├── storage.py


## Data Flow

1.  **Request Reception**: The API receives an evaluation request from the main service.
2.  **Immediate Audit Logging**: The raw request is immediately placed into the audit queue.
3.  **Background Evaluation**: The request is simultaneously placed into the evaluation queue for processing.
4.  **API Response**: The service responds immediately with `202 Accepted`.
5.  **Audit Storage**: A worker processes the audit queue, writing raw requests to the configured storage backend (e.g., `audit_data/`).
6.  **Evaluation Processing**: An evaluation worker picks up the request, invokes the appropriate evaluator(s) based on the graph's node executions, and generates evaluation results.
7.  **Results Storage**: The final evaluation results are written by a storage worker to the configured backend (e.g., `eval_results/`).

```mermaid
graph TD
    A[Main Service] -->|HTTP Request| B(API Endpoint);
    B --> C{Audit Queue};
    B --> D{Evaluation Queue};
    C --> E[Storage Worker];
    E --> F[Audit Storage (Local/GCS)];
    D --> G[Evaluation Worker];
    G --> H[Evaluator Function(s)];
    H --> I{Results Queue};
    I --> J[Storage Worker];
    J --> K[Results Storage (Local/GCS)];
    B -->|202 Accepted| A;
```

## API Endpoints

### POST `/evaluate`
Submits a conversation for asynchronous evaluation.

**Request Body:**
```json
{
  "thread_id": "string",
  "turn_index": 0,
  "conversation_flow": {
    "thread_id": "string",
    "user_query": "string",
    "node_executions": [...]
  }
}
```

### POST `/evaluate/sync`
Submits a conversation for synchronous evaluation, returning the results immediately. Useful for testing and development.

### GET `/health`
Provides a detailed health check of the service and its components (evaluator, queues, storage).

### GET `/metrics`
Exposes detailed metrics for monitoring, including queue sizes, processing counts, and component-specific stats.

### GET `/config`
Returns the current service configuration, with sensitive values redacted.

## Adding a New Evaluator

The service is designed for easy extension. To add a new evaluator:

1.  **Create the Evaluator Module**:
    -   Inside the `evals-service/evaluators/` directory, create a new Python file (e.g., `my_new_evaluator.py`).
    -   In this file, define an `async` function that accepts `node_execution: EnrichedNodeExecutionLog` and `user_query: str` as arguments and returns a `NodeEvaluation` instance.

2.  **Define Models (if needed)**:
    -   If your evaluator produces a unique data structure, define a Pydantic model for it in `evals-service/evaluators/models.py`, inheriting from `NodeEvaluation`.
    -   If your evaluator uses a structured LLM call with `instructor`, define the expected response model (the "judgement") in `evals-service/evaluators/judgements.py`.

3.  **Add Prompts (if needed)**:
    -   If your evaluator calls an LLM, add its system message and user prompt template to `evals-service/evaluators/prompts.json`. Follow the existing structure.
    -   Use the `get_eval_prompt()` and `get_system_message()` helpers from `evals-service/evaluators/base.py` to load and format your prompts.

4.  **Register the Evaluator**:
    -   Open `evals-service/evaluators/__init__.py`.
    -   Import your new evaluator function.
    -   Add an entry to the `EVALUATOR_REGISTRY` dictionary. The key should be the exact name of the graph node you want this evaluator to run for, and the value should be a list containing your evaluator function.

    ```python
    # evals-service/evaluators/__init__.py

    # ... existing imports
    from .my_new_evaluator import evaluate_my_new_node # 1. Import it

    EVALUATOR_REGISTRY: Dict[str, List[Callable]] = {
        "relevance_check": [evaluate_relevance_check],
        # ... other evaluators
        "my_new_node_name": [evaluate_my_new_node], # 2. Register it
    }
    ```

## Environment Variables
The service is configured via environment variables, managed by Pydantic.

### Core Service
- `LLM_OPENAI_API_KEY`: **Required**. Your OpenAI API key.
- `SENTRY_DSN`: Optional DSN for Sentry error tracking.

### Storage Configuration
- `STORAGE_STORAGE_BACKEND`: `local` (default) or `gcs`.
- `STORAGE_BATCH_SIZE`: Number of records to batch before writing to storage (default: `100`).
- `STORAGE_WRITE_TIMEOUT_SECONDS`: Max seconds to wait before writing an incomplete batch (default: `5.0`).
- `STORAGE_AUDIT_DATA_PATH`: Local path for audit data (default: `./audit_data`).
- `STORAGE_EVAL_RESULTS_PATH`: Local path for evaluation results (default: `./eval_results`).
- `STORAGE_GCS_EVAL_RESULTS_BUCKET_NAME`: **Required if `STORAGE_STORAGE_BACKEND` is `gcs`**.
- `STORAGE_GCS_AUDIT_BUCKET_NAME`: **Required if `STORAGE_STORAGE_BACKEND` is `gcs`**.

**GCS Authentication**: The service uses Google Cloud's standard Application Default Credentials (ADC). Ensure the runtime environment is authenticated (e.g., by setting `GOOGLE_APPLICATION_CREDENTIALS` or running on a configured GCP resource).

## Running the Service

### Development
```bash
# from repository root
cd evals-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### Docker
The repository includes a `docker-compose.yml` for local development.
```bash
# from repository root
docker-compose up evaluation-service
```

To run with GCS, create a `docker-compose.override.yml` or an `.env` file to set the `STORAGE_*` and `GOOGLE_APPLICATION_CREDENTIALS` variables.

## Integration with Main Service

The main backend service communicates with this evaluation service via its `evaluation_client.py` module, which handles the HTTP requests, serialization, and error handling.
