from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

env_path = Path(__file__).parent.parent / "start_docker" / ".env"
load_dotenv(dotenv_path=env_path)


def pytest_configure(config):
    """Configure pytest with mocked services"""
    # Patch Minio before any app imports
    patcher_minio = patch("minio.Minio")
    patcher_minio.start()

    # Patch aio_pika before app imports
    patcher_aio_pika = patch("aio_pika.connect")
    patcher_aio_pika.start()


@pytest.fixture
def sample_gesture_data():
    """Create sample gesture data for testing"""
    return [[float(i) for _ in range(18)] for i in range(50)]


@pytest.fixture
def sample_model():
    """Create a simple Keras model for testing"""
    model = Sequential([Dense(3, activation="softmax", input_shape=(50, 18))])
    scaler = MinMaxScaler()
    scaler.fit(np.random.rand(100, 18))

    from app.GestureService import Model

    return Model(
        model=model, scaler=scaler, classes=np.array(["class1", "class2", "class3"])
    )


@pytest.fixture
def mock_minio_client():
    """Create mock MinIO client"""
    mock = Mock()
    mock.bucket_exists = Mock(return_value=True)
    mock.fput_object = Mock()
    mock.fget_object = Mock()
    return mock


@pytest.fixture
def training_data():
    """Create sample training data"""
    return {
        "gesture1": [
            [[float(x) for x in range(18)] for _ in range(50)],
            [[float(x) for x in range(18)] for _ in range(60)],
            [[float(x) for x in range(18)] for _ in range(55)],
            [[float(x) for x in range(18)] for _ in range(50)],
        ],
        "gesture2": [
            [[float(x) for x in range(18)] for _ in range(55)],
            [[float(x) for x in range(18)] for _ in range(45)],
            [[float(x) for x in range(18)] for _ in range(50)],
            [[float(x) for x in range(18)] for _ in range(52)],
        ],
    }
