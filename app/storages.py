import asyncio

import joblib
import models as Models
import numpy as np
from minio import Minio
from tensorflow.keras.models import load_model, save_model


class ModelMinIOStorage:
    def __init__(self, minio_client: Minio, bucket_name: str):
        self.client = minio_client
        self.bucket_name = bucket_name

    async def save_model(self, model_id: str, model: Models.Model):
        """Зберігає модель використовуючи тимчасові файли"""

        import os
        import tempfile

        # Створюємо тимчасову директорію для моделі
        with tempfile.TemporaryDirectory() as tmpdir:
            # Зберігаємо компоненти в тимчасові файли
            model_path = os.path.join(tmpdir, "model.keras")
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            classes_path = os.path.join(tmpdir, "classes.npy")

            # Синхронні операції з диском виконуємо в окремому потоці
            await asyncio.to_thread(save_model, model.model, model_path)
            await asyncio.to_thread(joblib.dump, model.scaler, scaler_path)
            await asyncio.to_thread(np.save, classes_path, model.classes)

            # Асинхронні операції з MinIO
            await asyncio.to_thread(
                self.client.fput_object,
                self.bucket_name,
                f"model_{model_id}.keras",
                model_path,
            )
            await asyncio.to_thread(
                self.client.fput_object,
                self.bucket_name,
                f"scaler_{model_id}.pkl",
                scaler_path,
            )
            await asyncio.to_thread(
                self.client.fput_object,
                self.bucket_name,
                f"labels_{model_id}.npy",
                classes_path,
            )

    async def load_model(self, model_id: str) -> Models.Model:
        """Завантажує модель з MinIO"""

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Завантажуємо файли з MinIO
            model_path = os.path.join(tmpdir, "model.keras")
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            classes_path = os.path.join(tmpdir, "classes.npy")

            # Асинхронне завантаження з MinIO
            await asyncio.to_thread(
                self.client.fget_object,
                self.bucket_name,
                f"model_{model_id}.keras",
                model_path
            )
            await asyncio.to_thread(
                self.client.fget_object,
                self.bucket_name,
                f"scaler_{model_id}.pkl",
                scaler_path
            )
            await asyncio.to_thread(
                self.client.fget_object,
                self.bucket_name,
                f"labels_{model_id}.npy",
                classes_path
            )

            # Синхронні операції з диском в окремому потоці
            keras_model = await asyncio.to_thread(load_model, model_path)
            scaler = await asyncio.to_thread(joblib.load, scaler_path)
            classes = await asyncio.to_thread(np.load, classes_path, allow_pickle=True)

            return Models.Model(model=keras_model, scaler=scaler, classes=classes)
