import asyncio
from contextlib import asynccontextmanager

import aio_pika
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from minio import Minio
from minio.error import S3Error
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

import models as Models
from storages import ModelMinIOStorage


SEQUENCE_LENGTH = 50
EXPECTED_COLUMNS = 18


RABBIT_URL = "amqp://guest:guest@rabbitmq:5672/"

connection: aio_pika.RobustConnection = None
channel: aio_pika.abc.AbstractChannel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection, channel

    connection = await aio_pika.connect_robust(RABBIT_URL)
    channel = await connection.channel()

    await channel.declare_queue("train_tasks_queue", durable=True)
    await channel.declare_queue("train_results_queue", durable=True)

    asyncio.create_task(consume_train_tasks())

    yield

    await connection.close()


app = FastAPI(lifespan=lifespan)

async def consume_train_tasks():
    queue = await channel.declare_queue("train_tasks_queue", durable=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                try:
                    body = json.loads(message.body.decode())
                    await train_model(body)
                except Exception as e:
                    print(f"Consumer error: {e}")
    

local_models: dict[str, Models.Model] = {}
storage = ModelMinIOStorage(
    Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadminpassword",
        secure=False,
    ),
    "gesture-models",
)


async def train_model(message: dict) -> None:
    task_id = message.get("taskId")
    model_id = message.get("modelId")

    print(f"Отримано задачу: taskId={task_id}, modelId={model_id}")

    try:
        async with httpx.AsyncClient() as client:
            url = f"http://host.docker.internal:8080/api/v1/internal/models/{model_id}/training-data"

            response = await client.get(url)
            response.raise_for_status()
            training_data = response.json()

        print(f"Отримано дані для моделі {model_id}")

        await actual_training(model_id, training_data)

        print(f"Модель {model_id} успішно натренована")

        await send_training_result(model_id, "SUCCESS")

    except Exception as e:
        error_message = str(e)

        if isinstance(e, httpx.HTTPError):
            print(f"Помилка при отриманні даних: {e}")
        elif isinstance(e, S3Error):
            print(f"Помилка MinIO: {e}")
        else:
            print(f"Помилка тренування: {e}")

        await send_training_result(model_id, "FAILED", error_message)


async def send_training_result(model_id: str, status: str, error_message: str = None) -> None:
    result_message = {
        "modelId": model_id,
        "status": status,
        "errorMessage": error_message,
    }

    if status != "FAILED":
        result_message.update({
            "s3KerasPath": f"model_{model_id}.keras",
            "s3ScalerPath": f"scaler_{model_id}.pkl",
            "s3LabelsPath": f"labels_{model_id}.npy",
        })

    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(result_message).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key="train_results_queue",
    )

    print(f"Результат для моделі {model_id} відправлено")


async def actual_training(
    model_id: str, gestures: dict[str, list[list[list[float]]]]
) -> None:
    """
    Ендпоінт для тренування нової моделі.
    Отримує дані (наприклад, жести), тренує модель і повертає її.
    """
    if not gestures:
        raise Exception("Отримано порожні дані для тренування")

    model = Models.Model(model=None, scaler=None, classes=None)

    samples = []
    labels = []

    # -------------------------------
    # 1. Зчитування JSON
    # -------------------------------
    for label, sequences in gestures.items():
        for seq in sequences:
            df = pd.DataFrame(seq)

            # Перевірка кількості колонок
            if df.shape[1] != EXPECTED_COLUMNS:
                print(
                    f"Пропускаю {label} — неправильна кількість колонок {df.shape[1]}"
                )
                continue

            # Ресемплінг
            df_resampled = Models.resample_sequence(df, SEQUENCE_LENGTH)

            if df_resampled.shape != (SEQUENCE_LENGTH, EXPECTED_COLUMNS):
                print(
                    f"Пропускаю {label} після ресемплінгу — отримано {df_resampled.shape}"
                )
                continue

            samples.append(df_resampled.values.astype(float))
            labels.append(label)

    if len(samples) == 0:
        raise Exception("Немає валідних даних для тренування")

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
    _, counts = np.unique(y, return_counts=True)

    if np.any(counts < 2):
        raise Exception("Кожен клас повинен мати мінімум 2 приклади")

    X_train, X_test, y_train, y_test = train_test_split(
        samples,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # забезпечує правильне розділення по класах
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
    X_test_scaled = np.clip(
        model.scaler.transform(X_test_2d).reshape(N_test, T, F), -1, 1
    )

    # -------------------------------
    # 5. Модель LSTM(32) з Input шаром
    # -------------------------------
    model.model = Sequential(
        [
            Input(shape=(T, F)),  # Забирає попередження input_shape
            LSTM(32, return_sequences=False),  # 32 юніти
            Dropout(0.3),  # трохи більше регуляризації
            Dense(64, activation="relu"),  # Dense шар перед виходом
            Dropout(0.2),
            Dense(len(np.unique(y)), activation="softmax"),  # вихідний шар
        ]
    )

    model.model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # -------------------------------
    # 6. Навчання
    # -------------------------------
    model.model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=4,
        verbose=1,
    )

    # -------------------------------
    # 7. Оцінка
    # -------------------------------
    test_loss, test_accuracy = model.model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Точність моделі на тестових даних: {test_accuracy * 100:.2f}%")

    # -------------------------------
    # 8. Збереження моделі у Minio
    # -------------------------------
    await storage.save_model(model_id, model)


@app.post("/init")
async def init_model(model: Models.InitModelRequest):
    """
    Ендпоінт для ініціалізації моделі.
    Отримує модель (наприклад, збережену Keras-модель) і
    присвоює її змінній current_model.
    """
    try:
        model_id = model.modelId
        local_models[model_id] = await storage.load_model(model_id)
        return {"message": "Model initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize model: {str(e)}",
        )


@app.post("/predict")
def predict_gesture(gesture: Models.GestureData):
    """
    Ендпоінт для передбачення жесту.
    Використовує поточну модель (current_model) для передбачення
    і повертає результат.
    """
    model_id, gesture_data = gesture.modelId, gesture.rawData

    if model_id not in local_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No model initialized"
        )

    if not gesture_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format or empty 'rawData' array",
        )

    current_model = local_models[model_id]

    # Перевірка, що кожен запис має правильну кількість ознак (18)
    if len(gesture_data[0]) != EXPECTED_COLUMNS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {EXPECTED_COLUMNS} \
            columns, but got {len(gesture_data[0])}",
        )

    df = pd.DataFrame(gesture_data)

    # --- Приведення даних до єдиної довжини ---
    df_resampled = Models.resample_sequence(df, SEQUENCE_LENGTH)

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

    return {"predictedLabel": predicted_label, "confidence": float(confidence)}
