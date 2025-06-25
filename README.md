# ChatRaghu Backend

A microservices-based backend for raghu.fyi application, consisting of a main API service and a separate evaluation service.

## Architecture

The application is split into two main services:

1. **Main ChatRaghu Backend** (Port port1): Handles chat requests, document retrieval, and conversation flow
2. **Evaluation Service** (Port port2): Processes conversation evaluations and quality metrics

## Services

### Main Service (`chatraghu-backend`)
- FastAPI application handling chat requests
- Document retrieval and RAG functionality
- Conversation flow management
- Authentication and rate limiting

### Evaluation Service (`evaluation-service`)
- Separate FastAPI service for evaluation processing
- Asynchronous evaluation queue management
- Multiple evaluation types (relevance, persona, etc.)
- Results storage and metrics

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd chatRaghu-backend
   cp .env.example .env  # Create and configure your .env file
   ```

2. **Start both services**:
   ```bash
   docker-compose up --build
   ```

3. **Access services**:
   - Main API: http://localhost:port1
   - Evaluation Service: http://localhost:port2
   - API Documentation: http://localhost:port1/docs
   - Evaluation Service Docs: http://localhost:port2/docs

### Development Setup

1. **Main Service**:
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port port1 --reload
   ```

2. **Evaluation Service**:
   ```bash
   cd evals-service
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port port2 --reload
   ```

## Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
VALID_API_KEYS=your_api_key1,your_api_key2

# Service Configuration
EVALUATION_SERVICE_URL=http://localhost:port2
EVALUATION_SERVICE_TIMEOUT=30

# Optional
SENTRY_DSN=your_sentry_dsn
OPIK_API_KEY=your_opik_key
OPIK_WORKSPACE=your_workspace
OPIK_PROJECT_NAME=your_project
```

## API Usage

### Chat Endpoint
```bash
curl -X POST "http://localhost:port1/api/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Evaluation Service Health Check
```bash
curl "http://localhost:port2/health"
```

## Service Communication

The main service communicates with the evaluation service using HTTP requests:

- **Asynchronous Evaluation**: Main service sends evaluation requests to `/evaluate`
- **Health Monitoring**: Regular health checks via `/health`

## Development

### Project Structure
```
chatRaghu-backend/
├── app.py                 # Main FastAPI application
├── evaluation_client.py   # Client for evaluation service
├── evaluation_models.py   # Local evaluation models
├── evaluation_queue_manager.py  # Queue manager for evaluations
├── graph/                 # Conversation graph logic
├── utils/                 # Utility functions
├── evals-service/         # Evaluation service
│   ├── app.py            # Evaluation service FastAPI app
│   ├── models.py         # Evaluation models
│   ├── evaluators.py     # Evaluation logic
│   ├── run_evals.py      # Evaluation runner
│   └── queue_manager.py  # Evaluation queue manager
└── docker-compose.yml    # Multi-service orchestration
```

### Adding New Evaluation Types

1. Add evaluation logic in `evals-service/evaluators.py`
2. Update models in `evals-service/models.py`
3. Add API endpoints in `evals-service/app.py`

### Scaling

Each service can be scaled independently:

```bash
# Scale main service
docker-compose up --scale chatraghu-backend=3

# Scale evaluation service
docker-compose up --scale evaluation-service=2
```

## Monitoring

### Health Checks
- Main Service: `GET /health`
- Evaluation Service: `GET /health`

### Metrics
- Evaluation Service: `GET /metrics`

### Logs
```bash
# View all logs
docker-compose logs
