# ChatRaghu Backend

A microservices-based backend for raghu.fyi application, consisting of a main API service and a separate evaluation service.

## Architecture

The application is split into two main services:

1. **Main ChatRaghu Backend** (Port 3000): Handles chat requests, document retrieval, and conversation flow
2. **Evaluation Service** (Port 8001): Processes conversation evaluations and quality metrics

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
- **NEW**: Structured LLM validation using Instructor and Pydantic

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

3. **For testing with real evaluation client**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build
   ```

4. **Access services**:
   - Main API: http://localhost:3000
   - Evaluation Service: http://localhost:8001
   - API Documentation: http://localhost:3000/docs
   - Evaluation Service Docs: http://localhost:8001/docs

### Development Setup

1. **Main Service**:
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port 3000 --reload
   ```

2. **Evaluation Service**:
   ```bash
   cd evals-service
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port 8001 --reload
   ```

## Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
VALID_API_KEYS=your_api_key1,your_api_key2

# Service Configuration
EVALUATION_SERVICE_URL=http://localhost:8001
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
curl -X POST "http://localhost:3000/api/chat" \
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
curl "http://localhost:8001/health"
```

## Service Communication

The main service communicates with the evaluation service using HTTP requests:

- **Asynchronous Evaluation**: Main service sends evaluation requests to `/evaluate`
- **Health Monitoring**: Regular health checks via `/health`

## Recent Updates

### Evaluation Service Enhancements

**Structured LLM Validation with Instructor**
- **NEW**: Replaced manual JSON parsing with Instructor-patched OpenAI client
- **NEW**: Added `LLMRelevanceJudgement` Pydantic model for strict response validation
- **NEW**: Eliminated LLM hallucination through structured validation
- **NEW**: Enhanced error handling and logging for evaluation processes

**Updated Dependencies**
- Upgraded to `openai==1.70.0` for Instructor compatibility
- Added `instructor==1.9.0` for structured LLM responses
- Updated `pydantic==2.8.2` and `pydantic-settings>=2.3.0`
- Enhanced logging with `opik>=1.7.40`

**Code Quality Improvements**
- Replaced deprecated `.dict()` calls with `.model_dump()` across codebase
- Enhanced type safety and validation
- Improved error handling and retry mechanisms

### Graph Component Updates

**Pydantic Compatibility**
- Updated all Pydantic model serialization to use `.model_dump()`
- Enhanced type safety in conversation flow management
- Improved data validation across graph nodes

## Testing

### Test Suite Overview

The test suite is organized into three consolidated files for better maintainability:

#### 1. `tests/test_unit.py` - Unit Tests
**Purpose**: Tests individual components and functions in isolation.

**Contains**:
- **TestMainService**: Main backend service endpoint tests (health, chat, API key validation)
- **TestEvaluationService**: Evaluation service endpoint tests (health, metrics, evaluate endpoints)
- **TestEvaluationComponents**: Internal evaluation service component tests (queue manager, storage, evaluator)
- **TestConfiguration**: Configuration loading and validation tests

**Markers**: `@pytest.mark.unit`

#### 2. `tests/test_integration.py` - Integration Tests
**Purpose**: Tests service-to-service communication and full system flows.

**Contains**:
- **test_ruthless_end_to_end_flow**: Complete end-to-end integration test with file creation verification
- Service health and availability tests
- Diverse query testing with different node paths

**Markers**: `@pytest.mark.integration`, `@pytest.mark.storage_integration`

#### 3. `tests/test_graph.py` - Graph Tests
**Purpose**: Tests the graph structure, nodes, and graph-related functionality.

**Contains**:
- **TestGraphStructure**: Graph package structure and import tests
- **TestGraphErrors**: Graph error handling and edge cases
- **TestGraphAssembly**: Graph assembly and utility tests
- Node functionality tests (relevance check, query/respond)

**Markers**: `@pytest.mark.graph_structure`, `@pytest.mark.graph_errors`, `@pytest.mark.graph_assembly`

### Running Tests

#### Prerequisites
Ensure both services are running:
```bash
# Start services for testing
docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build
```

#### Test Execution

**Run all tests**:
```bash
pytest
```

**Run specific test types**:
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Graph tests only
pytest -m "graph_structure or graph_errors or graph_assembly"
```

**Run specific test files**:
```bash
# Unit tests
pytest tests/test_unit.py

# Integration tests
pytest tests/test_integration.py

# Graph tests
pytest tests/test_graph.py
```

**Run specific test classes or functions**:
```bash
# Specific test class
pytest tests/test_unit.py::TestMainService

# Specific test function
pytest tests/test_integration.py::test_ruthless_end_to_end_flow
```

**Exclude specific test types**:
```bash
# Skip slow tests
pytest -m "not slow"

# Skip rate limiting tests
pytest -m "not rate_limiting"
```

### Test Configuration

#### Environment Variables
Tests use the existing `.env` file for configuration. For testing storage flows:
```bash
export STORAGE_BATCH_SIZE=1  # For immediate file writing during tests
```

#### Test Data
- Sample requests and responses are defined in `tests/conftest.py`
- Test fixtures are available for common test scenarios
- Mock data is used where appropriate to avoid external dependencies

### Test Dependencies

#### Required Services
- **Main Service**: Must be running on `http://localhost:3000`
- **Evaluation Service**: Must be running on `http://localhost:8001`

#### External Dependencies
- `httpx`: For HTTP client testing
- `pytest-asyncio`: For async test support
- `pytest-httpx`: For HTTP mocking (optional)

### Test Maintenance

#### Adding New Tests
1. **Unit Tests**: Add to appropriate class in `tests/test_unit.py`
2. **Integration Tests**: Add to `tests/test_integration.py`
3. **Graph Tests**: Add to appropriate class in `tests/test_graph.py`

#### Test Organization
- Use descriptive test names
- Group related tests in classes
- Use appropriate markers for categorization
- Add docstrings explaining test purpose

#### Test Data Management
- Use fixtures for reusable test data
- Clean up test data after tests
- Use mock objects for external dependencies
- Avoid hardcoded test values

### Troubleshooting

#### Common Issues
1. **Service Not Available**: Ensure both services are running
2. **Permission Errors**: Check file permissions for storage directories
3. **Import Errors**: Verify Python path includes project root
4. **Timeout Errors**: Increase timeout values for slow tests

#### Debug Mode
Run tests with verbose output:
```bash
pytest -v -s
```

#### Test Isolation
Run tests in isolation to identify issues:
```bash
pytest tests/test_unit.py::TestMainService::test_health_endpoint -v
```

## Development

### Project Structure
```
chatRaghu-backend/
├── app.py                 # Main FastAPI application
├── evaluation_client.py   # Client for evaluation service
├── evaluation_models.py   # Local evaluation models
├── evaluation_queue_manager.py  # Queue manager for evaluations
├── graph/                 # Conversation graph logic
│   ├── nodes.py          # Graph node implementations
│   ├── infrastructure.py # Graph infrastructure
│   ├── models.py         # Graph data models
│   └── retrieval.py      # Document retrieval logic
├── utils/                 # Utility functions
├── tests/                 # Comprehensive test suite
│   ├── test_unit.py      # Unit tests for all components
│   ├── test_integration.py # Integration tests
│   ├── test_graph.py     # Graph-specific tests
│   ├── conftest.py       # Shared test fixtures
│   └── README.md         # Test documentation
├── evals-service/         # Evaluation service
│   ├── app.py            # Evaluation service FastAPI app
│   ├── models.py         # Evaluation models
│   ├── evaluators.py     # Evaluation logic with Instructor validation
│   ├── run_evals.py      # Evaluation runner
│   ├── queue_manager.py  # Evaluation queue manager
│   ├── storage.py        # Results storage
│   └── config.py         # Service configuration
├── docker-compose.yml    # Multi-service orchestration
└── docker-compose.test.yml # Test-specific configuration
```

### Adding New Evaluation Types

1. **Create evaluation logic** in `evals-service/evaluators.py`:
   ```python
   # Use Instructor for structured validation
   class YourEvalModel(BaseModel):
       # Define your validation schema
       pass

   async def evaluate_your_type(node_execution: EnrichedNodeExecutionLog) -> YourEvalModel:
       # Use instructor-patched client for validation
       result = await client.chat.completions.create(
           model=config.llm.openai_model,
           response_model=YourEvalModel,
           messages=[...]
       )
   ```

2. **Update models** in `evals-service/models.py`
3. **Add API endpoints** in `evals-service/app.py`
4. **Add corresponding tests** in `tests/test_unit.py::TestEvaluationComponents`

### Key Technical Changes

**Instructor Integration**
- OpenAI client patched with `instructor.patch()` for structured responses
- Pydantic models used for LLM response validation
- Eliminated manual JSON parsing and potential hallucination

**Pydantic v2 Compatibility**
- All `.dict()` calls replaced with `.model_dump()`
- Enhanced type safety and validation
- Improved serialization performance

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

# View specific service logs
docker-compose logs evaluation-service
```

## Testing

### Test Configuration
Use `docker-compose.test.yml` for testing with real evaluation client:
- `MOCK_EVAL_CLIENT=false` - Uses real evaluation service
- `STORAGE_BATCH_SIZE=1` - Immediate file writes for tests
- `STORAGE_WRITE_TIMEOUT_SECONDS=5.0` - Faster timeouts

```bash
docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build
```
