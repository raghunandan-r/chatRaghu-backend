# ChatRaghu Evaluation Service

This is a separate FastAPI service that handles conversation evaluation for the ChatRaghu backend. It processes conversation flows and evaluates the quality of responses using various metrics.

## Features

- **Asynchronous Evaluation**: Processes evaluation requests in the background
- **Synchronous Evaluation**: Provides immediate evaluation results for testing
- **Configurable Storage**: Supports multiple storage backends for evaluation results, including local filesystem and Google Cloud Storage (GCS)
- **Multiple Evaluation Types**:
  - Relevance check evaluation
  - Query/respond decision evaluation
  - Few-shot selector evaluation
  - Context generation evaluation
  - Persona consistency evaluation
- **Queue Management**: Handles evaluation requests with retry logic and batch processing
- **Health Monitoring**: Provides health check and metrics endpoints

## API Endpoints

### POST `/evaluate`
Submit a conversation for asynchronous evaluation.

**Request Body:**
```json
{
  "thread_id": "string",
  "query": "string",
  "response": "string",
  "retrieved_docs": [{"page_content": "string", "metadata": {}}],
  "conversation_flow": {
    "thread_id": "string",
    "user_query": "string",
    "node_executions": [...]
  }
}
```

### POST `/evaluate/sync`
Submit a conversation for synchronous evaluation (returns results immediately).

### GET `/health`
Health check endpoint.

### GET `/metrics`
Get service metrics including queue size and worker status.

## Environment Variables

The service uses a hierarchical configuration system powered by Pydantic. Environment variables are prefixed to avoid collisions.

### Core Service
- `LLM_OPENAI_API_KEY`: OpenAI API key for evaluation models
- `SENTRY_DSN`: Sentry DSN for error tracking (optional)

### Storage Configuration
- `STORAGE_STORAGE_BACKEND`: The storage backend to use. Can be `local` (default) or `gcs`
- `STORAGE_EVAL_RESULTS_PATH`: Path for local evaluation results storage (default: `./eval_results`)
- `STORAGE_AUDIT_DATA_PATH`: Path for local audit data storage (default: `./audit_data`)
- `STORAGE_GCS_EVAL_RESULTS_BUCKET_NAME`: GCS bucket name for evaluation results. Required if `STORAGE_BACKEND` is `gcs`
- `STORAGE_GCS_AUDIT_BUCKET_NAME`: GCS bucket name for audit data. Required if `STORAGE_BACKEND` is `gcs`

**Note on GCS Authentication:** When using the `gcs` backend, the service authenticates using Google Cloud's standard Application Default Credentials (ADC) mechanism. Ensure the environment where the service is running is authenticated (e.g., by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable or running on a configured GCP resource).

## Running the Service

### Development
```bash
cd evals-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### Docker
To run with the default local storage:
```bash
docker build -t chatraghu-evaluation .
docker run -p 8001:8001 --env-file .env chatraghu-evaluation
```

### Docker Compose

The default `docker-compose.yml` is configured for local development and uses the local filesystem for storage.

```bash
docker-compose up evaluation-service
```

To run the service with Google Cloud Storage (GCS), you can use a Docker Compose override file.

1. Create a `gcs.env` file with your GCS configuration:
   ```env
   STORAGE_STORAGE_BACKEND=gcs
   STORAGE_GCS_EVAL_RESULTS_BUCKET_NAME=your-eval-results-bucket
   STORAGE_GCS_AUDIT_BUCKET_NAME=your-audit-data-bucket
   # Ensure GOOGLE_APPLICATION_CREDENTIALS is set in this file
   # or the environment it runs in.
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
   ```

2. Create a `docker-compose.gcs.yml` override file:
   ```yml
   version: '3.8'
   services:
     evaluation-service:
       env_file:
         - gcs.env
       volumes:
         # Mount service account key if using a local file
         - /path/to/your/service-account.json:/path/to/your/service-account.json:ro
   ```

3. Start the services using both files:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.gcs.yml up
   ```

## Integration with Main Service

The main ChatRaghu backend service communicates with this evaluation service using the `evaluation_client.py` module. The client handles:

- HTTP communication with the evaluation service
- Request/response serialization
- Error handling and retries
- Connection pooling

## Evaluation Results Data Model

Evaluation results are stored as Parquet files in the configured storage backend (local filesystem or GCS). Each record follows a detailed schema to capture a comprehensive view of the evaluation.

The primary fields in the `EvaluationResult` model include:

- **Identifiers**: `run_id`, `thread_id`, `turn_index` for tracing and ordering.
- **Timestamps & Latency**:
  - `timestamp_start`, `timestamp_end`: Start and end times of the evaluation pipeline.
  - `graph_latency_ms`, `time_to_first_token_ms`, `evaluation_latency_ms`: Detailed performance metrics.
- **Core Conversation Data**: `query`, `response`, and `retrieved_docs`.
- **Token Counts**:
  - `graph_total_prompt_tokens`, `graph_total_completion_tokens`: Token usage from the main conversation graph.
  - `eval_total_prompt_tokens`, `eval_total_completion_tokens`: Token usage from the evaluation LLM calls.
- **Evaluation Scores**: A dictionary named `evaluations` containing scores for different metrics (e.g., correctness, relevance).
- **Metadata**: An open-ended `metadata` field for additional context.

## Architecture

```
Main Service (Port 3000)     Evaluation Service (Port 8001)
     |                              |
     |-- HTTP Request ------------>|
     |                              |-- Process Evaluation
     |                              |-- Store Results
     |<-- Success Response --------|
```

The evaluation service operates independently and can be scaled separately from the main service.
