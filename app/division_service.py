import asyncio
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential


class DivisionService:
    @dataclass
    class Model:
        model: Sequential
        scaler: MinMaxScaler

    def __init__(self, confidence_threshold, window_size: int, num_features: int):
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.num_features = num_features
        self.local_models: dict[str, DivisionService.Model] = {}

    async def predict(self, model: Model, window_data: list, left: int):
        return await asyncio.to_thread(self._predict_sync, model, window_data, left)

    def _predict_sync(
        self, model: Model, window_data: list, left: int
    ) -> tuple[int, int]:
        # Прибрати reshape
        window_2d = np.asarray(window_data, dtype=np.float32).reshape(
            -1, self.num_features
        )
        window_scaled = model.scaler.transform(window_2d).reshape(
            1, self.window_size, self.num_features
        )
        window_scaled = np.clip(window_scaled, -1, 1)

        has_start_prob, start_norm, has_end_prob, end_norm = model.model.predict(
            window_scaled, verbose=0
        )

        has_start_prob = has_start_prob[0][0]
        has_end_prob = has_end_prob[0][0]

        start, end = None, None

        if has_start_prob > self.confidence_threshold:
            local_idx = int(start_norm[0][0] * (self.window_size - 1))
            absolute_idx = left + local_idx
            start = absolute_idx

        if has_end_prob > self.confidence_threshold:
            local_idx = int(end_norm[0][0] * (self.window_size - 1))
            absolute_idx = left + local_idx
            end = absolute_idx

        return start, end
