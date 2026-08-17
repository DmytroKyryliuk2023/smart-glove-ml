import asyncio
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential


class GestureService:
    @dataclass
    class Model:
        model: Sequential
        scaler: MinMaxScaler
        classes: np.ndarray

    def __init__(self, sequence_length: int, num_features: int):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.local_models: dict[str, GestureService.Model] = {}

    async def predict(self, model: Model, gesture_data: list):
        return await asyncio.to_thread(self._predict_sync, model, gesture_data)

    def resample_sequence(self, df: pd.DataFrame, target_length: int) -> pd.DataFrame:
        """
        Adjusts the number of rows in the DataFrame to target_length.
        If there are fewer rows, interpolation is performed.
        If there are more rows, points are selected uniformly.

        Args:
            df: Pandas DataFrame containing time-series data (one gesture).
            target_length: Desired number of points after resampling.

        Returns:
            Pandas DataFrame with the same structure and target_length rows.
        """
        df = df.reset_index(drop=True)
        current_length = len(df)

        if current_length < target_length:
            new_index = np.linspace(0, current_length - 1, target_length)
            df_resampled = df.reindex(new_index)
            df_resampled = df_resampled.interpolate(method="linear")
            return df_resampled.reset_index(drop=True)

        elif current_length > target_length:
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            df_resampled = df.iloc[indices].reset_index(drop=True)
            return df_resampled

        else:
            return df

    def _predict_sync(self, model: Model, gesture_data: list):
        df = pd.DataFrame(gesture_data)
        df_resampled = self.resample_sequence(df, self.sequence_length)

        input_data = df_resampled.values.astype(float)
        data_scaled = np.clip(model.scaler.transform(input_data), -1, 1)
        data_for_model = np.expand_dims(data_scaled, axis=0)

        prediction_probs = model.model.predict(data_for_model, verbose=0)
        label_index = np.argmax(prediction_probs)
        predicted_label = model.classes[label_index]
        confidence = np.max(prediction_probs)

        return {"predictedLabel": predicted_label, "confidence": float(confidence)}
