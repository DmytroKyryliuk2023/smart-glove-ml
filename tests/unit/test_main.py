from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, lifespan, local_models

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_local_models():
    """Clear local_models before each test"""
    local_models.clear()
    yield
    local_models.clear()


@pytest.mark.asyncio
async def test_lifespan():
    """Test lifespan context manager"""
    mock_rabbitmq = AsyncMock()
    
    with patch('app.main.rabbitmq', mock_rabbitmq):
        async with lifespan(app):
            mock_rabbitmq.connect.assert_called_once()
            mock_rabbitmq.declare_queue.assert_called()
            assert mock_rabbitmq.declare_queue.call_count == 2


def test_init_model_success():
    """Test successful model initialization"""
    with patch('app.main.training_service.storage.load_model', AsyncMock(return_value=Mock())):
        response = client.post("/models/test_model_123")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Model initialized successfully"}
        assert "test_model_123" in local_models


def test_init_model_already_exists():
    """Test initializing already existing model"""
    local_models["existing_model"] = Mock()
    
    response = client.post("/models/existing_model")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Model already initialized"}


def test_init_model_failure():
    """Test model initialization failure"""
    with patch('app.main.training_service.storage.load_model', AsyncMock(side_effect=Exception("Load error"))):
        response = client.post("/models/fail_model")
        
        assert response.status_code == 400
        assert "Failed to initialize model" in response.json()["detail"]


def test_delete_model_success():
    """Test successful model deletion"""
    local_models["test_model"] = Mock()
    
    response = client.delete("/models/test_model")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Model deleted successfully"}
    assert "test_model" not in local_models


def test_delete_model_not_found():
    """Test deleting non-existent model"""
    response = client.delete("/models/nonexistent")
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Model not found"


def test_predict_success():
    """Test successful prediction"""
    test_model = Mock()
    local_models["test_model"] = test_model
    
    with patch('app.main.prediction_service.predict', AsyncMock(return_value={"predictedLabel": "test", "confidence": 0.95})):
        request_data = {
            "modelId": "test_model",
            "rawData": [[1.0] * 18 for _ in range(50)]
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        assert "predictedLabel" in response.json()


def test_predict_model_not_initialized():
    """Test prediction with uninitialized model"""
    request_data = {
        "modelId": "nonexistent",
        "rawData": [[1.0] * 18 for _ in range(50)]
    }
    
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "No model initialized"


def test_predict_empty_data():
    """Test prediction with empty data"""
    test_model = Mock()
    local_models["test_model"] = test_model
    
    request_data = {
        "modelId": "test_model",
        "rawData": []
    }
    
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 400
    assert "Invalid format" in response.json()["detail"]


def test_predict_invalid_columns():
    """Test prediction with wrong column count"""
    test_model = Mock()
    local_models["test_model"] = test_model
    
    request_data = {
        "modelId": "test_model",
        "rawData": [[1, 2, 3]]  # Wrong number of columns
    }
    
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 400
    assert "Expected 18 columns" in response.json()["detail"]