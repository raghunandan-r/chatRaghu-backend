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
