import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app import app

client = TestClient(app)

# Sample API Key for testing
VALID_API_KEY = "test_api_key"

@pytest.fixture
def api_key_header():
    return {"X-API-Key": VALID_API_KEY}

@pytest.fixture
def invalid_api_key_header():
    return {"X-API-Key": "invalid_key"}

# Mock dependencies
@pytest.fixture
def mock_llm():
    with patch("app.llm") as mock_llm:
        mock_llm.invoke = AsyncMock(return_value="Mocked AI Response")
        mock_llm.ainvoke = AsyncMock(return_value="Mocked AI Response")
        yield mock_llm

@pytest.fixture
def mock_vector_store():
    with patch("app.vector_store") as mock_store:
        mock_store.asimilarity_search = AsyncMock(return_value=[
            MockDoc(content="Mocked Document 1"),
            MockDoc(content="Mocked Document 2"),
            MockDoc(content="Mocked Document 3"),
        ])
        yield mock_store

class MockDoc:
    def __init__(self, content):
        self.content = content

def test_chat_success(api_key_header, mock_llm, mock_vector_store):
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, Raghu!",
                "thread_id": "thread1"
            }
        ]
    }
    
    with patch("app.graph.astream", return_value=AsyncMock()):
        response = client.post("/api/chat", json=payload, headers=api_key_header)
    
    assert response.status_code == 200
    assert "response" in response.json()

def test_chat_invalid_api_key(invalid_api_key_header):
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, Raghu!",
                "thread_id": "thread1"
            }
        ]
    }
    
    response = client.post("/api/chat", json=payload, headers=invalid_api_key_header)
    
    assert response.status_code == 401
    assert response.json()["error"] == "Invalid API key"

def test_chat_rate_limit_exceeded(api_key_header):
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, Raghu!",
                "thread_id": "thread1"
            }
        ]
    }
    
    # Simulate rate limit exceeded by populating the Usage with max requests
    with patch("app.Storage.api_key_usage", {VALID_API_KEY: [datetime.now()] * 100}):
        response = client.post("/api/chat", json=payload, headers=api_key_header)
    
    assert response.status_code == 429
    assert response.json()["error"] == "API rate limit exceeded"

@pytest.mark.asyncio
async def test_generate_with_retrieved_context(mock_llm, mock_vector_store):
    # Implement additional tests for asynchronous functions if needed
    pass

# Add more tests as needed