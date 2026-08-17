import asyncio
import os
import tempfile

import joblib
import numpy as np
from minio import Minio
from tensorflow.keras.models import load_model, save_model

from .division_service import DivisionService
from .gesture_service import GestureService


class MinioStorage:
    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        bucket_name: str,
    ):
        self.client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )
        self.bucket_name = bucket_name
        
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except Exception as e:
            print(f"Failed to create bucket: {e}")

    async def save_gesture_model(self, model_id: str, model: GestureService.Model):
        def _save_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                classes_path = os.path.join(tmpdir, "classes.npy")

                save_model(model.model, model_path)
                joblib.dump(model.scaler, scaler_path)
                np.save(classes_path, model.classes)

                self.client.fput_object(
                    self.bucket_name, f"gesture/model_{model_id}.keras", model_path
                )
                self.client.fput_object(
                    self.bucket_name, f"gesture/scaler_{model_id}.pkl", scaler_path
                )
                self.client.fput_object(
                    self.bucket_name, f"gesture/labels_{model_id}.npy", classes_path
                )

        await asyncio.to_thread(_save_sync)
        print(f"Model {model_id} saved to MinIO")

    async def load_gesture_model(self, model_id: str) -> GestureService.Model:
        def _load_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                classes_path = os.path.join(tmpdir, "classes.npy")

                self.client.fget_object(
                    self.bucket_name, f"gesture/model_{model_id}.keras", model_path
                )
                self.client.fget_object(
                    self.bucket_name, f"gesture/scaler_{model_id}.pkl", scaler_path
                )
                self.client.fget_object(
                    self.bucket_name, f"gesture/labels_{model_id}.npy", classes_path
                )

                keras_model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                classes = np.load(classes_path, allow_pickle=True)

                return GestureService.Model(
                    model=keras_model, scaler=scaler, classes=classes
                )

        return await asyncio.to_thread(_load_sync)

    async def save_division_model(self, model_id: str, model: DivisionService.Model):
        def _save_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")

                save_model(model.model, model_path)
                joblib.dump(model.scaler, scaler_path)

                self.client.fput_object(
                    self.bucket_name, f"division/model_{model_id}.keras", model_path
                )
                self.client.fput_object(
                    self.bucket_name, f"division/scaler_{model_id}.pkl", scaler_path
                )

        await asyncio.to_thread(_save_sync)
        print(f"Model {model_id} saved to MinIO")

    async def load_division_model(self, model_id: str) -> DivisionService.Model:
        def _load_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")

                self.client.fget_object(
                    self.bucket_name, f"division/model_{model_id}.keras", model_path
                )
                self.client.fget_object(
                    self.bucket_name, f"division/scaler_{model_id}.pkl", scaler_path
                )

                keras_model = load_model(model_path)
                scaler = joblib.load(scaler_path)

                return DivisionService.Model(
                    model=keras_model, scaler=scaler
                )

        return await asyncio.to_thread(_load_sync)
