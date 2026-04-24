from dataclasses import dataclass
import http

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential


@dataclass
class Model():
    model: Sequential
    scaler: MinMaxScaler
    classes: np.ndarray
    

class InitModelRequest(BaseModel):
    modelId: str
    modelUrl: str
    scalerUrl: str
    labelsUrl: str


class GestureData(BaseModel):
    modelId: str
    rawData: list[list[float]]
    

def resample_sequence(df: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """
    Приводить кількість рядків у df до target_length.
    Якщо рядків менше — виконує інтерполяцію.
    Якщо більше — рівномірно вибирає точки.
    
    Args:
        df: pandas DataFrame із часовими даними (один жест)
        target_length: бажана кількість точок після вирівнювання
        
    Returns:
        pandas DataFrame тієї ж структури, але з target_length рядків
    """
    # Завжди скидаємо індекс для чистої роботи
    df = df.reset_index(drop=True)
    current_length = len(df)

    # якщо даних менше - інтерполюємо
    if current_length < target_length:
        # Створюємо новий дробовий індекс, що відповідає цільовій довжині
        new_index = np.linspace(0, current_length - 1, target_length)
        
        # Розширюємо DataFrame до нового індексу. 
        # Точки, що не існували, заповняться значеннями NaN.
        df_resampled = df.reindex(new_index)
        
        # 3. Інтерполюємо (заповнюємо) пропущені значення NaN лінійно
        df_resampled = df_resampled.interpolate(method='linear')
        
        return df_resampled.reset_index(drop=True)

    # якщо даних більше - рівномірно вибираємо точки (цей метод працював правильно)
    elif current_length > target_length:
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        df_resampled = df.iloc[indices].reset_index(drop=True)
        return df_resampled

    # якщо вже рівно — нічого не робимо
    else:
        return df
    