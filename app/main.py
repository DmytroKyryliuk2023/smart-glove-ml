import json

from fastapi import FastAPI, HTTPException, status
from faststream.rabbit.fastapi import RabbitRouter
import httpx
from minio import Minio
from minio.error import S3Error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

import models as Models
from storages import ModelMinIOStorage


SEQUENCE_LENGTH = 50
EXPECTED_COLUMNS = 18


app = FastAPI()
local_models: dict[str, Models.Model] = {}
storage = ModelMinIOStorage(
    Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadminpassword",
        secure=False
    ),
    "gesture-models"
)
rabbit_router = RabbitRouter()
app.include_router(rabbit_router)


@rabbit_router.subscriber("train_tasks_queue")
async def train_model(message: dict):
    """Обробник повідомлень з черги train_tasks_queue"""
    task_id = message.get("taskId")
    model_id = message.get("modelId")
    
    print(f"Отримано задачу: taskId={task_id}, modelId={model_id}")
    
    try:
        # Отримуємо дані з Java бекенду
        async with httpx.AsyncClient() as client:
            url = f"http://java-backend:8080/api/v1/internal/models/{model_id}/training-data"
            response = await client.get(url)
            response.raise_for_status()
            training_data = response.json()
        
        print(f"Отримано дані для моделі {model_id}")
        print(f"Дані: {json.dumps(training_data, indent=2)[:50]}...")  # Логуємо перші 50 символів
        
        # Тренування моделі
        train_model(model_id, training_data)
        
        print(f"Модель {model_id} успішно натренована")
        
    except Exception as e:
        if isinstance(e, httpx.HTTPError):
            print(f"Помилка при отриманні даних: {e}")
        else:
            print(f"Помилка тренування моделі {model_id}: {e}")
        
        # Звіт про помилку у чергу
        
        
    



def train_model(gestures: dict[str, list[list[list[float]]]]):
    """
    Ендпоінт для тренування нової моделі.
    Отримує дані (наприклад, жести), тренує модель і повертає її.
    """
    if not gestures:
        raise HTTPException(400, "Empty training data")
    
    model = Models.Model(
        sequence_length=50,
        expected_columns=18,
        model=None,
        scaler=None,
        classes=None
    )
    
    samples = []
    labels = []

    # -------------------------------
    # 1. Зчитування JSON
    # -------------------------------
    for label, sequences in gestures.items():
        for seq in sequences:
            df = pd.DataFrame(seq)

            # Перевірка кількості колонок
            if df.shape[1] != model.expected_columns:
                print(f"Пропускаю {label} — неправильна кількість колонок {df.shape[1]}")
                continue

            # Ресемплінг
            df_resampled = Models.resample_sequence(df, model.sequence_length)

            if df_resampled.shape != (model.sequence_length , model.expected_columns):
                print(f"Пропускаю {label} після ресемплінгу — отримано {df_resampled.shape}")
                continue

            samples.append(df_resampled.values.astype(float))
            labels.append(label)

    samples = np.array(samples)
    labels = np.array(labels)
    
    # -------------------------------
    # 2. Кодування міток
    # -------------------------------
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    model.classes = encoder.classes_

    # -------------------------------
    # 3. Stratified train/test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        samples, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # забезпечує правильне розділення по класах
    )

    # -------------------------------
    # 4. Масштабування
    # -------------------------------
    model.scaler = MinMaxScaler(feature_range=(-1, 1))
    N_train, T, F = X_train.shape
    N_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, F)
    X_test_2d = X_test.reshape(-1, F)

    model.scaler.fit(X_train_2d)
    X_train_scaled = model.scaler.transform(X_train_2d).reshape(N_train, T, F)
    X_test_scaled = np.clip(model.scaler.transform(X_test_2d).reshape(N_test, T, F), -1, 1)

    # -------------------------------
    # 5. Модель LSTM(32) з Input шаром
    # -------------------------------
    model.model = Sequential([
        Input(shape=(T, F)),       # Забирає попередження input_shape
        LSTM(32, return_sequences=False),  # 32 юніти
        Dropout(0.3),               # трохи більше регуляризації
        Dense(64, activation='relu'),  # Dense шар перед виходом
        Dropout(0.2),
        Dense(len(np.unique(y)), activation='softmax')  # вихідний шар
    ])

    model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # -------------------------------
    # 6. Навчання
    # -------------------------------
    model.model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=4,
        verbose=1
    )

    # -------------------------------
    # 7. Оцінка
    # -------------------------------
    test_loss, test_accuracy = model.model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Точність моделі на тестових даних: {test_accuracy * 100:.2f}%")
    
    # -------------------------------
    # 8. Кодування
    # -------------------------------
    encoded_model = ModelSerializer.encode(model)
    
    return encoded_model


@app.post("/init")
def init_model(model: Models.IdentifiedModel):
    """
    Ендпоінт для ініціалізації моделі.
    Отримує модель (наприклад, збережену Keras-модель) і 
    присвоює її змінній current_model.
    """
    
    try:
        model_id, model_instance = model.model_id, ModelSerializer.decode(model.model)
        models[model_id] = model_instance
        return {"message": "Model initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize model: {str(e)}"
        )


@app.post("/predict")
def predict_gesture(gesture: Models.GestureData):
    """
    Ендпоінт для передбачення жесту.
    Використовує поточну модель (current_model) для передбачення
    і повертає результат.
    """
    model_id, gesture_data = gesture.model_id, gesture.gesture_data

    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model initialized"
        )
        
    if not gesture_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format or empty 'gesture_data' array"
        )
    
    current_model = models[model_id]

    # Перевірка, що кожен запис має правильну кількість ознак (18)
    if len(gesture_data[0]) != current_model.expected_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {current_model.expected_columns} \
            columns, but got {len(gesture_data[0])}"
        )

    df = pd.DataFrame(gesture_data)
    
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
