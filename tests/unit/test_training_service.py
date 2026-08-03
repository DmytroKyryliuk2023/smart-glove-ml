from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.training_service import TrainingService


@pytest.mark.asyncio
async def test_fetch_training_data():
    """Test fetching training data from server"""
    service = TrainingService("localhost:9000", "key", "secret", "http://server")

    mock_response = Mock()
    mock_response.json = Mock(return_value={"gesture1": []})
    mock_response.raise_for_status = Mock()

    with patch("app.training_service.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await service.fetch_training_data("test_id")

        assert result == {"gesture1": []}


@pytest.mark.asyncio
async def test_train_model_success(training_data):
    """Test successful model training"""
    service = TrainingService("localhost:9000", "key", "secret", "http://server")

    with patch.object(
        service, "fetch_training_data", AsyncMock(return_value=training_data)
    ):
        with patch.object(service, "_actual_training", AsyncMock()) as mock_training:
            await service.train_model("test_id")

            mock_training.assert_called_once_with("test_id", training_data)


@pytest.mark.asyncio
async def test_actual_training_with_valid_data(training_data):
    """Test actual training with valid data"""
    service = TrainingService("localhost:9000", "key", "secret", "http://server")
    service.storage = Mock()
    service.storage.save_model = AsyncMock()

    await service._actual_training("test_id", training_data)

    service.storage.save_model.assert_called_once()


@pytest.mark.asyncio
async def test_actual_training_with_empty_data():
    """Test training with empty data raises error"""
    service = TrainingService("localhost:9000", "key", "secret", "http://server")

    with pytest.raises(Exception, match="Отримано порожні дані"):
        await service._actual_training("test_id", {})


@pytest.mark.asyncio
async def test_actual_training_with_invalid_column_count():
    """Test training with invalid column count"""
    invalid_data = {
        "gesture1": [
            [[1, 2, 3]]  # Wrong number of columns
        ]
    }

    service = TrainingService("localhost:9000", "key", "secret", "http://server")

    with pytest.raises(Exception, match="Немає валідних даних"):
        await service._actual_training("test_id", invalid_data)


def test_expected_columns_constant():
    """Test EXPECTED_COLUMNS constant"""
    assert TrainingService.EXPECTED_COLUMNS == 18


def test_sequence_length_constant():
    """Test SEQUENCE_LENGTH constant"""
    assert TrainingService.SEQUENCE_LENGTH == 50
