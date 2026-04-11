import base64
import io
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel


class Models:
    class Model(BaseModel):
        sequence_length: int
        expected_columns: int
        model: Sequential
        scaler: MinMaxScaler
        classes: np.ndarray
        
    class EncodedModel(BaseModel):
        sequence_length: int
        expected_columns: int
        model: str
        scaler: str
        classes: str
    
    @staticmethod
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
        
    
class ModelSerializer:
    @staticmethod
    def encode(model: Models.Model) -> Models.EncodedModel:
        # Серіалізація моделі Keras
        model_buffer = io.BytesIO()
        save_model(model.model, model_buffer, save_format='h5')
        model_buffer.seek(0)
        model_encoded = base64.b64encode(model_buffer.read()).decode('utf-8')
        model_buffer.close()
        
        # Серіалізація scaler
        scaler_buffer = io.BytesIO()
        joblib.dump(model.scaler, scaler_buffer, save_format='pkl')
        scaler_buffer.seek(0)
        scaler_encoded = base64.b64encode(scaler_buffer.read()).decode('utf-8')
        scaler_buffer.close()
        
        # Серіалізація класів
        classes_buffer = io.BytesIO()
        np.save(classes_buffer, model.classes, save_format='npy')
        classes_buffer.seek(0)
        classes_encoded = base64.b64encode(classes_buffer.read()).decode('utf-8')
        classes_buffer.close()

        return Models.EncodedModel(
            sequence_length=model.sequence_length,
            expected_columns=model.expected_columns,
            model=model_encoded,
            scaler=scaler_encoded,
            classes=classes_encoded
        )
    
    @staticmethod
    def decode(model: Models.EncodedModel) -> Models.Model:
        # Десеріалізація моделі Keras
        model_buffer = io.BytesIO(base64.b64decode(model.model))
        model_buffer.seek(0)
        keras_model = load_model(model_buffer)
        model_buffer.close()
        
        # Десеріалізація scaler
        scaler_buffer = io.BytesIO(base64.b64decode(model.scaler))
        scaler_buffer.seek(0)
        scaler = joblib.load(scaler_buffer)
        scaler_buffer.close()
        
        # Десеріалізація класів
        classes_buffer = io.BytesIO(base64.b64decode(model.classes))
        classes_buffer.seek(0)
        classes = np.load(classes_buffer, allow_pickle=True)
        classes_buffer.close()

        return Models.Model(
            sequence_length=model.sequence_length,
            expected_columns=model.expected_columns,
            model=keras_model,
            scaler=scaler,
            classes=classes
        )
        
      
app = FastAPI()
current_model: Models.Model = None


@app.post("/train")
def train_model(data: dict):
    """
    Ендпоінт для тренування нової моделі.
    Отримує дані (наприклад, жести), тренує модель і повертає її.
    """
    
    model = Models.Model(
        sequence_length=100,
        expected_columns=10,
        model=Sequential(),
        scaler=MinMaxScaler(),
        classes=np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    )
    
    encoded_model = ModelSerializer.encode(model)
    
    return encoded_model


@app.post("/init")
def init_model(model: Models.EncodedModel):
    """
    Ендпоінт для ініціалізації моделі.
    Отримує модель (наприклад, збережену Keras-модель) і 
    присвоює її змінній current_model.
    """
    global current_model
    
    try:
        current_model = ModelSerializer.decode(model)
        return "Model initialized successfully"
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize model: {str(e)}"
        )


@app.post("/predict")
def predict_gesture(data: dict):
    if not current_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model initialized"
        )
        
    if not data or 'gesture_data' not in data or not data['gesture_data']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format or empty 'gesture_data' array"
        )
    
    sequence_data = data['gesture_data']

    # Перевірка, що кожен запис має правильну кількість ознак (18)
    if len(sequence_data[0]) != current_model.expected_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {current_model.expected_columns} \
            columns, but got {len(sequence_data[0])}"
        )

    df = pd.DataFrame(sequence_data)
    
    # --- Приведення даних до єдиної довжини ---
    df_resampled = Models.resample_sequence(df, current_model.sequence_length)
    
    # 1. Перетворюємо дані в numpy масив
    input_data = df_resampled.values.astype(float)
    
    # 2. Масштабуємо дані за допомогою завантаженого скейлера
    data_scaled = np.clip(current_model.scaler.transform(input_data), -1, 1)
    
    # 3. Додаємо "batch" вимір для моделі: (50, 18) -> (1, 50, 18)
    data_for_model = np.expand_dims(data_scaled, axis=0)
    
    # 4. Робимо передбачення
    prediction_probs = current_model.model.predict(data_for_model, verbose=0)
    
    # 5. Інтерпретуємо результат
    label_index = np.argmax(prediction_probs)
    predicted_label = current_model.classes[label_index]
    confidence = np.max(prediction_probs)
    
    return {
        "prediction": predicted_label,
        "confidence": float(confidence)
    }


