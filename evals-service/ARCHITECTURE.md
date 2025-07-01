# Evaluation Service Architecture

## Overview

The ChatRaghu Evaluation Service has been redesigned with a decoupled dual-queue system that provides high availability, resilience, and comprehensive monitoring capabilities.

## Architecture Components

### 1. Dual Queue System

The service implements two separate queues for different purposes:

- **Audit Queue**: Immediately captures raw evaluation requests for audit logging
- **Evaluation Queue**: Processes evaluation requests in the background

This separation ensures that:
- Raw requests are always logged before any processing begins
- Evaluation failures don't impact audit logging
- Each queue can be scaled independently

### 2. Storage Management

The service uses a modular storage system with:

- **StorageManager**: Generic storage manager that handles batching and writing
- **StorageBackend Protocol**: Abstract interface for different storage backends
- **LocalStorageBackend**: Local file system storage (current implementation)
- **S3StorageBackend**: S3 storage (placeholder for future implementation)

### 3. Configuration Management

All configuration is managed through Pydantic settings with environment variable support:

```python
# Service Configuration
EVAL_SERVICE_NAME=chatraghu-evals
EVAL_ENVIRONMENT=development
EVAL_MAX_RETRY_ATTEMPTS=3

# Storage Configuration
STORAGE_AUDIT_DATA_PATH=./audit_data
STORAGE_EVAL_RESULTS_PATH=./eval_results
STORAGE_BATCH_SIZE=100

# LLM Configuration
LLM_OPENAI_API_KEY=your-api-key
LLM_OPENAI_MODEL=gpt-4o
LLM_OPENAI_MAX_RETRIES=3
```

## Data Flow

### Request Processing Flow

1. **Request Reception**: API endpoint receives evaluation request
2. **Immediate Audit Logging**: Request is immediately queued for audit logging
3. **Background Evaluation**: Request is scheduled for background evaluation
4. **Response**: API returns 202 Accepted immediately
5. **Audit Storage**: Raw request is written to audit parquet files
6. **Evaluation Processing**: Evaluation runs asynchronously
7. **Results Storage**: Evaluation results are written to results parquet files

### Storage Flow

```
Request → Audit Queue → StorageManager → audit_data/*.parquet
Request → Eval Queue → Evaluator → Results Queue → StorageManager → eval_results/*.parquet
```

## Health Monitoring

### Health Check Endpoint (`/health`)

Provides comprehensive health status for all components:

```json
{
  "status": "healthy",
  "service": "chatraghu-evals",
  "components": {
    "evaluator": {
      "evaluator_healthy": true,
      "evaluation_count": 150,
      "error_count": 2
    },
    "queue_manager": {
      "queue_manager_healthy": true,
      "audit_queue_size": 0,
      "eval_queue_size": 5
    },
    "audit_storage": {
      "storage_manager_healthy": true,
      "processed_count": 200
    },
    "results_storage": {
      "storage_manager_healthy": true,
      "processed_count": 150
    }
  }
}
```

### Metrics Endpoint (`/metrics`)

Provides detailed metrics for monitoring and alerting:

```json
{
  "service": {
    "name": "chatraghu-evals",
    "version": "1.0.0"
  },
  "components": {
    "evaluator": {
      "evaluation_count": 150,
      "success_rate": 0.987
    },
    "queue_manager": {
      "audit_processed": 200,
      "eval_processed": 150
    }
  }
}
```

## Resilience Features

### 1. Retry Logic

- **Exponential Backoff**: All LLM calls use exponential backoff
- **Configurable Retries**: Retry attempts are configurable per component
- **Graceful Degradation**: Service continues operating even if some components fail

### 2. Queue Management

- **Bounded Queues**: Queues have maximum sizes to prevent memory exhaustion
- **Backpressure**: Full queues trigger appropriate error responses
- **Worker Isolation**: Each queue has dedicated workers

### 3. Error Handling

- **Comprehensive Logging**: All errors are logged with context
- **Error Metrics**: Error counts are tracked and exposed via metrics
- **Graceful Shutdown**: Proper cleanup on service shutdown

## Configuration

### Environment Variables

The service uses a hierarchical configuration system:

```bash
# Service Configuration
EVAL_SERVICE_NAME=chatraghu-evals
EVAL_ENVIRONMENT=development
EVAL_MAX_RETRY_ATTEMPTS=3
EVAL_MAX_QUEUE_SIZE=10000

# Storage Configuration
STORAGE_AUDIT_DATA_PATH=./audit_data
STORAGE_EVAL_RESULTS_PATH=./eval_results
STORAGE_BATCH_SIZE=100
STORAGE_WRITE_TIMEOUT_SECONDS=5.0

# LLM Configuration
LLM_OPENAI_API_KEY=your-api-key
LLM_OPENAI_MODEL=gpt-4o
LLM_OPENAI_MAX_RETRIES=3
LLM_OPENAI_TIMEOUT_SECONDS=30

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=1
```

### Configuration Validation

All configuration is validated at startup using Pydantic, ensuring:
- Required fields are present
- Values are within acceptable ranges
- Type safety is enforced

## Extensibility

### Adding New Storage Backends

To add a new storage backend:

1. Implement the `StorageBackend` protocol
2. Add configuration options
3. Update the `create_storage_backend()` factory function

### Adding New Evaluation Models

To add new evaluation models:

1. Create new evaluator functions in `evaluators.py`
2. Add corresponding evaluation result models
3. Update the evaluation logic in `run_evals.py`

### Adding New Monitoring

To add new monitoring capabilities:

1. Extend the health check and metrics methods
2. Add new endpoints as needed
3. Update the configuration schema

## Performance Considerations

### Batch Processing

- **Configurable Batch Sizes**: Batch sizes are configurable for optimal performance
- **Timeout-based Batching**: Batches are written after timeout even if not full
- **Compression**: Parquet files use snappy compression for efficiency

### Queue Management

- **Bounded Queues**: Prevent memory exhaustion
- **Worker Isolation**: Independent workers for different queues
- **Graceful Backpressure**: Proper handling of queue overflow

### Storage Optimization

- **Columnar Format**: Parquet provides efficient storage and querying
- **Compression**: Reduces storage requirements
- **Batch Writes**: Minimizes I/O operations

## Deployment Considerations

### Containerization

The service is designed for containerized deployment with:
- Environment-based configuration
- Health check endpoints
- Graceful shutdown handling
- Proper signal handling

### Monitoring

For production deployment, consider:
- Prometheus metrics collection
- Distributed tracing (OpenTelemetry)
- Centralized logging (ELK stack)
- Alerting on health check failures

### Scaling

The service can be scaled by:
- Running multiple instances behind a load balancer
- Adjusting queue worker counts
- Using external storage backends (S3)
- Implementing horizontal scaling for evaluation workers
