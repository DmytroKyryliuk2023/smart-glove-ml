import asyncio
import os
import tempfile

import joblib
import numpy as np
from minio import Minio
from tensorflow.keras.models import (
    load_model as tf_load_model,
)
from tensorflow.keras.models import (
    save_model as tf_save_model,
)

from . import models


class ModelMinIOStorage:
    def __init__(self, minio_client: Minio, bucket_name: str):
        self.client = minio_client
        self.bucket_name = bucket_name
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except Exception as e:
            print(f"Failed to create bucket: {e}")

    async def save_model(self, model_id: str, model: models.Model):
        """Зберігає модель в MinIO"""

        def _save_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                classes_path = os.path.join(tmpdir, "classes.npy")

                # Зберігаємо компоненти
                tf_save_model(model.model, model_path)
                joblib.dump(model.scaler, scaler_path)
                np.save(classes_path, model.classes)

                # Завантажуємо в MinIO
                self.client.fput_object(
                    self.bucket_name, f"model_{model_id}.keras", model_path
                )
                self.client.fput_object(
                    self.bucket_name, f"scaler_{model_id}.pkl", scaler_path
                )
                self.client.fput_object(
                    self.bucket_name, f"labels_{model_id}.npy", classes_path
                )

        # Виконуємо всі синхронні операції в одному потоці
        await asyncio.to_thread(_save_sync)
        print(f"Model {model_id} saved to MinIO")

    async def load_model(self, model_id: str) -> models.Model:
        """Завантажує модель з MinIO"""

        def _load_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                classes_path = os.path.join(tmpdir, "classes.npy")

                # Завантажуємо з MinIO
                self.client.fget_object(
                    self.bucket_name, f"model_{model_id}.keras", model_path
                )
                self.client.fget_object(
                    self.bucket_name, f"scaler_{model_id}.pkl", scaler_path
                )
                self.client.fget_object(
                    self.bucket_name, f"labels_{model_id}.npy", classes_path
                )

                # Завантажуємо компоненти
                keras_model = tf_load_model(model_path)
                scaler = joblib.load(scaler_path)
                classes = np.load(classes_path, allow_pickle=True)

                return models.Model(model=keras_model, scaler=scaler, classes=classes)

        return await asyncio.to_thread(_load_sync)
