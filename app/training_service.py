import httpx
import numpy as np
import pandas as pd
from minio import Minio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

import models as Models
from storages import ModelMinIOStorage


class TrainingService:
    SEQUENCE_LENGTH = 50
    EXPECTED_COLUMNS = 18

    def __init__(self):
        self.storage = ModelMinIOStorage(
            Minio(
                "minio:9000",
                access_key="minioadmin",
                secret_key="minioadminpassword",
                secure=False,
            ),
            "gesture-models",
        )

    async def fetch_training_data(self, model_id: str):
        async with httpx.AsyncClient() as client:
            url = f"http://host.docker.internal:8080/api/v1/internal/models/{model_id}/training-data"
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def train_model(self, model_id: str) -> None:
        print(f"Отримано задачу на тренування: modelId={model_id}")
        training_data = await self.fetch_training_data(model_id)
        print(f"Отримано дані для моделі {model_id}")
        await self._actual_training(model_id, training_data)
        print(f"Модель {model_id} успішно натренована")

    async def _actual_training(self, model_id: str, gestures: dict):
        if not gestures:
            raise Exception("Отримано порожні дані для тренування")

        model = Models.Model(model=None, scaler=None, classes=None)
        samples, labels = [], []

        for label, sequences in gestures.items():
            for seq in sequences:
                df = pd.DataFrame(seq)
                if df.shape[1] != self.EXPECTED_COLUMNS:
                    print(f"Пропускаю {label} — неправильна кількість колонок {df.shape[1]}")
                    continue

                df_resampled = Models.resample_sequence(df, self.SEQUENCE_LENGTH)
                if df_resampled.shape != (self.SEQUENCE_LENGTH, self.EXPECTED_COLUMNS):
                    print(f"Пропускаю {label} після ресемплінгу — отримано {df_resampled.shape}")
                    continue

                samples.append(df_resampled.values.astype(float))
                labels.append(label)

        if len(samples) == 0:
            raise Exception("Немає валідних даних для тренування")

        samples, labels = np.array(samples), np.array(labels)

        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        model.classes = encoder.classes_

        _, counts = np.unique(y, return_counts=True)
        if np.any(counts < 2):
            raise Exception("Кожен клас повинен мати мінімум 2 приклади")

        X_train, X_test, y_train, y_test = train_test_split(
            samples, y, test_size=0.2, random_state=42, stratify=y,
        )

        model.scaler = MinMaxScaler(feature_range=(-1, 1))
        N_train, T, F = X_train.shape
        N_test = X_test.shape[0]

        X_train_2d = X_train.reshape(-1, F)
        X_test_2d = X_test.reshape(-1, F)

        model.scaler.fit(X_train_2d)
        X_train_scaled = model.scaler.transform(X_train_2d).reshape(N_train, T, F)
        X_test_scaled = np.clip(model.scaler.transform(X_test_2d).reshape(N_test, T, F), -1, 1)

        model.model = Sequential([
            Input(shape=(T, F)),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(len(np.unique(y)), activation="softmax"),
        ])

        model.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        model.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=30, batch_size=16, verbose=1,
        )

        _, test_accuracy = model.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Точність моделі на тестових даних: {test_accuracy * 100:.2f}%")

        await self.storage.save_model(model_id, model)