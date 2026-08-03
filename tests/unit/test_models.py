import numpy as np
import pandas as pd

from app import models


def test_resample_sequence_interpolation():
    """Test resampling when input is shorter than target"""
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    result = models.resample_sequence(df, 50)
    
    assert len(result) == 50
    assert result.shape[1] == 2
    assert result.iloc[0][0] == 1
    assert result.iloc[-1][0] == 5


def test_resample_sequence_downsample():
    """Test resampling when input is longer than target"""
    df = pd.DataFrame(np.random.rand(100, 5))
    result = models.resample_sequence(df, 50)
    
    assert len(result) == 50
    assert result.shape[1] == 5
    assert all(result.iloc[i] is not None for i in range(len(result)))


def test_resample_sequence_equal_length():
    """Test resampling when input equals target length"""
    df = pd.DataFrame(np.random.rand(50, 5))
    result = models.resample_sequence(df, 50)
    
    assert len(result) == 50
    pd.testing.assert_frame_equal(df.reset_index(drop=True), result)


def test_gesture_data_validation():
    """Test GestureData model validation"""
    data = models.GestureData(
        modelId="test_model",
        rawData=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    assert data.modelId == "test_model"
    assert len(data.rawData) == 2


def test_model_dataclass():
    """Test Model dataclass creation"""
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    
    mock_model = Sequential()
    mock_scaler = MinMaxScaler()
    mock_classes = np.array(['a', 'b'])
    
    model_obj = models.Model(
        model=mock_model,
        scaler=mock_scaler,
        classes=mock_classes
    )
    
    assert model_obj.model == mock_model
    assert model_obj.scaler == mock_scaler
    assert np.array_equal(model_obj.classes, mock_classes)