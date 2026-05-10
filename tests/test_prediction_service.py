import pytest
import numpy as np
from unittest.mock import Mock
from app.prediction_service import PredictionService


@pytest.mark.asyncio
async def test_predict_returns_correct_format(sample_model, sample_gesture_data):
    """Test prediction returns expected format"""
    service = PredictionService()
    
    # Mock model prediction
    sample_model.model.predict = Mock(return_value=np.array([[0.1, 0.7, 0.2]]))
    
    result = await service.predict(sample_model, sample_gesture_data)
    
    assert "predictedLabel" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1


@pytest.mark.asyncio
async def test_predict_handles_different_lengths(sample_model):
    """Test prediction works with different input lengths"""
    service = PredictionService()
    sample_model.model.predict = Mock(return_value=np.array([[0.5, 0.5]]))
    
    # Test with shorter sequence
    short_data = [[1.0] * 18 for _ in range(30)]
    result = await service.predict(sample_model, short_data)
    assert result is not None
    
    # Test with longer sequence
    long_data = [[1.0] * 18 for _ in range(100)]
    result = await service.predict(sample_model, long_data)
    assert result is not None


@pytest.mark.asyncio
async def test_predict_sync_calls_model_predict(sample_model):
    """Test _predict_sync method properly calls model predict"""
    service = PredictionService()
    mock_prediction = np.array([[0.2, 0.5, 0.3]])
    sample_model.model.predict = Mock(return_value=mock_prediction)
    
    gesture_data = [[float(i)] * 18 for i in range(50)]
    
    result = service._predict_sync(sample_model, gesture_data)
    
    sample_model.model.predict.assert_called_once()
    assert "predictedLabel" in result
    assert "confidence" in result


def test_sequence_length_constant():
    """Test SEQUENCE_LENGTH is correctly set"""
    assert PredictionService.SEQUENCE_LENGTH == 50