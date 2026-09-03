import asyncio
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class GestureService:
    @dataclass
    class Model:
        model: tf.keras.models.Sequential
        scaler: MinMaxScaler
        classes: np.ndarray

    def __init__(self, sequence_length: int, num_features: int):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.local_models: dict[str, GestureService.Model] = {}
        
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

    async def predict(self, model: Model, gesture_data: list):
        return await asyncio.to_thread(self._predict_sync, model, gesture_data)

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

    async def train(self, training_data: dict) -> Model:
        return await asyncio.to_thread(self._train_sync, training_data)

    def _train_sync(self, gestures: dict) -> Model:
        if not gestures:
            raise Exception("Отримано порожні дані для тренування")

        model = GestureService.Model(model=None, scaler=None, classes=None)
        samples, labels = [], []

        for label, sequences in gestures.items():
            for seq in sequences:
                df = pd.DataFrame(seq)
                if df.shape[1] != self.num_features:
                    print(
                        f"Пропускаю {label} — неправильна кількість колонок {df.shape[1]}"
                    )
                    continue

                df_resampled = GestureService.resample_sequence(
                    df, self.sequence_length
                )
                if df_resampled.shape != (self.sequence_length, self.num_features):
                    print(
                        f"Пропускаю {label} після ресемплінгу — отримано {df_resampled.shape}"
                    )
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
            samples,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model.scaler = MinMaxScaler(feature_range=(-1, 1))
        N_train, T, F = X_train.shape
        N_test = X_test.shape[0]

        X_train_2d = X_train.reshape(-1, F)
        X_test_2d = X_test.reshape(-1, F)

        model.scaler.fit(X_train_2d)
        X_train_scaled = model.scaler.transform(X_train_2d).reshape(N_train, T, F)
        X_test_scaled = np.clip(
            model.scaler.transform(X_test_2d).reshape(N_test, T, F), -1, 1
        )

        model.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(T, F)),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(np.unique(y)), activation="softmax"),
            ]
        )

        model.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=30,
            batch_size=16,
            verbose=1,
        )

        _, test_accuracy = model.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Точність моделі на тестових даних: {test_accuracy * 100:.2f}%")

        return model
