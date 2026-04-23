import joblib

from minio import Minio
import numpy as np
from tensorflow.keras.models import save_model, load_model

import models as Models


class ModelMinIOStorage:
    def __init__(self, minio_client: Minio, bucket_name: str):
        self.client = minio_client
        self.bucket_name = bucket_name
    
    def save_model(self, model: Models.Model, model_id: str):
        """Зберігає модель використовуючи тимчасові файли"""
        
        import tempfile
        import os
        
        # Створюємо тимчасову директорію для моделі
        with tempfile.TemporaryDirectory() as tmpdir:
            # Зберігаємо компоненти в тимчасові файли
            model_path = os.path.join(tmpdir, "model.keras")
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            classes_path = os.path.join(tmpdir, "classes.npy")
            
            save_model(model.model, model_path)
            joblib.dump(model.scaler, scaler_path)
            np.save(classes_path, model.classes)
            
            # Завантажуємо файли в MinIO
            self.client.fput_object(
                self.bucket_name,
                f"model_{model_id}.keras",
                model_path
            )
            self.client.fput_object(
                self.bucket_name,
                f"scaler_{model_id}.pkl",
                scaler_path
            )
            self.client.fput_object(
                self.bucket_name,
                f"labels_{model_id}.npy",
                classes_path
            )
    
    def load_model(self, model_id: str) -> Models.Model:
        """Завантажує модель з MinIO"""
        
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Завантажуємо файли з MinIO
            model_path = os.path.join(tmpdir, "model.keras")
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            classes_path = os.path.join(tmpdir, "classes.npy")
            
            self.client.fget_object(
                self.bucket_name,
                f"model_{model_id}.keras",
                model_path
            )
            self.client.fget_object(
                self.bucket_name,
                f"scaler_{model_id}.pkl",
                scaler_path
            )
            self.client.fget_object(
                self.bucket_name,
                f"labels_{model_id}.npy",
                classes_path
            )
            
            # Завантажуємо компоненти
            keras_model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            classes = np.load(classes_path, allow_pickle=True)
            
            return Models.Model(
                model=keras_model,
                scaler=scaler,
                classes=classes
            )
        