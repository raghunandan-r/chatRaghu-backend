# ChatRaghu Test Suite

This directory contains the consolidated test suite for the ChatRaghu backend services.

## Test Structure

The test suite has been consolidated into three main files for better organization and maintainability:

### 1. `test_unit.py` - Unit Tests
**Purpose**: Tests individual components and functions in isolation.

**Contains**:
- **TestMainService**: Main backend service endpoint tests (health, chat, API key validation)
- **TestEvaluationService**: Evaluation service endpoint tests (health, metrics, evaluate endpoints)
- **TestEvaluationComponents**: Internal evaluation service component tests (queue manager, storage, evaluator)
- **TestConfiguration**: Configuration loading and validation tests

**Markers**: `@pytest.mark.unit`

### 2. `test_integration.py` - Integration Tests
**Purpose**: Tests service-to-service communication and full system flows.

**Contains**:
- **TestEvaluationServiceIntegration**: Evaluation service integration tests
- **test_full_integration_flow**: Complete end-to-end integration test (main service + evaluation service)
- **test_full_storage_integration_flow**: Storage flow with file verification
- **test_chat_endpoint_node_path_coverage**: Node path coverage tests

**Markers**: `@pytest.mark.integration`, `@pytest.mark.storage_integration`, `@pytest.mark.node_path`

### 3. `test_graph.py` - Graph Tests
**Purpose**: Tests the graph structure, nodes, and graph-related functionality.

**Contains**:
- **TestGraphStructure**: Graph package structure and import tests
- **TestGraphErrors**: Graph error handling and edge cases
- **TestGraphAssembly**: Graph assembly and utility tests
- Node functionality tests (relevance check, query/respond)

**Markers**: `@pytest.mark.graph_structure`, `@pytest.mark.graph_errors`, `@pytest.mark.graph_assembly`

## Test Configuration

### Environment Variables
Tests use the existing `.env` file for configuration. For testing storage flows, you can set:
```bash
export STORAGE_BATCH_SIZE=1  # For immediate file writing during tests
```

### Test Data
- Sample requests and responses are defined in `conftest.py`
- Test fixtures are available for common test scenarios
- Mock data is used where appropriate to avoid external dependencies

### conftest.py Content
The `conftest.py` file provides shared fixtures and configuration for all tests:

**Service URLs**:
- `MAIN_SERVICE_URL`: Main service endpoint (`http://localhost:3000`)
- `EVALUATION_SERVICE_URL`: Evaluation service endpoint (`http://localhost:8001`)

**HTTP Client Fixture**:
- `http_client`: Async HTTP client for making requests to services

**Sample Data Fixtures**:
- `valid_api_headers`: Valid API key headers for authentication
- `sample_chat_request`: Sample chat request for testing
- `sample_evaluation_request`: Sample evaluation request with conversation flow

**Error Classes**:
- `ServiceTestError`: Custom exception for test failures
- `ServiceUnavailableError`: Custom exception for service unavailability

**Logging**:
- `logger`: Configured logger for test output

## Test Markers

### Test Type Markers
- `unit`: Unit tests (individual components)
- `integration`: Integration tests (service-to-service)
- `storage_integration`: Storage integration tests (file creation verification)

### Service-Specific Markers
- `main_service`: Tests for main ChatRaghu backend service
- `evaluation_service`: Tests for evaluation service

### Component-Specific Markers
- `graph_structure`: Graph structure tests
- `graph_errors`: Graph error handling tests
- `graph_assembly`: Graph assembly tests

### Feature-Specific Markers
- `node_path`: Node path coverage tests
- `rate_limiting`: Rate limiting tests

### Test Execution Markers
- `asyncio`: Async tests
- `slow`: Slow-running tests

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test types
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Storage integration tests only
pytest -m storage_integration

# Graph tests only
pytest -m "graph_structure or graph_errors or graph_assembly"
```

### Run specific test files
```bash
# Unit tests
pytest tests/test_unit.py

# Integration tests
pytest tests/test_integration.py

# Graph tests
pytest tests/test_graph.py
```

### Run specific test classes or functions
```bash
# Specific test class
pytest tests/test_unit.py::TestMainService

# Specific test function
pytest tests/test_integration.py::test_full_integration_flow
```

### Exclude specific test types
```bash
# Skip slow tests
pytest -m "not slow"

# Skip rate limiting tests
pytest -m "not rate_limiting"
```

## Test Dependencies

### Required Services
- **Main Service**: Must be running on `http://localhost:3000`
- **Evaluation Service**: Must be running on `http://localhost:8001`

### External Dependencies
- `httpx`: For HTTP client testing
- `pytest-asyncio`: For async test support
- `pytest-httpx`: For HTTP mocking (optional)

## Test Maintenance

### Adding New Tests
1. **Unit Tests**: Add to appropriate class in `test_unit.py`
2. **Integration Tests**: Add to `test_integration.py`
3. **Graph Tests**: Add to appropriate class in `test_graph.py`

### Test Organization
- Use descriptive test names
- Group related tests in classes
- Use appropriate markers for categorization
- Add docstrings explaining test purpose

### Test Data Management
- Use fixtures for reusable test data
- Clean up test data after tests
- Use mock objects for external dependencies
- Avoid hardcoded test values

## Troubleshooting

### Common Issues
1. **Service Not Available**: Ensure both services are running
2. **Permission Errors**: Check file permissions for storage directories
3. **Import Errors**: Verify Python path includes project root
4. **Timeout Errors**: Increase timeout values for slow tests

### Debug Mode
Run tests with verbose output:
```bash
pytest -v -s
```

### Test Isolation
Run tests in isolation to identify issues:
```bash
pytest tests/test_unit.py::TestMainService::test_health_endpoint -v
```

## Migration Notes

### From Old Structure
The old test structure had multiple overlapping files:
- `test_main_service.py` → `test_unit.py::TestMainService`
- `test_evaluation_service.py` → `test_unit.py::TestEvaluationService`
- `test_evaluation_service_unit.py` → `test_unit.py::TestEvaluationComponents`
- `test_graph_structure.py` → `test_graph.py::TestGraphStructure`
- `test_graph_errors.py` → `test_graph.py::TestGraphErrors`
- `test_storage_flow.py` → `test_integration.py::test_full_storage_integration_flow`

### Legacy Markers
For backward compatibility, legacy markers are still supported:
- `eval_unit` → `unit`
- `eval_integration` → `integration`
