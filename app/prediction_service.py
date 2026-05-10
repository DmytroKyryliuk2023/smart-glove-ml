import asyncio
import numpy as np
import pandas as pd
from . import models


class PredictionService:
    SEQUENCE_LENGTH = 50

    async def predict(self, model: models.Model, gesture_data: list):
        return await asyncio.to_thread(self._predict_sync, model, gesture_data)

    def _predict_sync(self, model: models.Model, gesture_data: list):
        df = pd.DataFrame(gesture_data)
        df_resampled = models.resample_sequence(df, self.SEQUENCE_LENGTH)

        input_data = df_resampled.values.astype(float)
        data_scaled = np.clip(model.scaler.transform(input_data), -1, 1)
        data_for_model = np.expand_dims(data_scaled, axis=0)

        prediction_probs = model.model.predict(data_for_model, verbose=0)
        label_index = np.argmax(prediction_probs)
        predicted_label = model.classes[label_index]
        confidence = np.max(prediction_probs)

        return {"predictedLabel": predicted_label, "confidence": float(confidence)}