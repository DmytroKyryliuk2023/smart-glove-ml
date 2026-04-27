import json
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import aio_pika
from fastapi import FastAPI
import requests

RABBIT_URL = "amqp://guest:guest@localhost:5672/"

base_url = "http://localhost:8000"
data_folder = Path("data")
train_file = data_folder / "gestures_merged.json"
test_file = data_folder / "excuse-me.json"


connection: Optional[aio_pika.RobustConnection] = None
channel: Optional[aio_pika.Channel] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection, channel

    # 🔌 Підключення до RabbitMQ
    connection = await aio_pika.connect_robust(RABBIT_URL)
    channel = await connection.channel()

    # гарантуємо що черги існують
    await channel.declare_queue("train_tasks_queue", durable=True)
    await channel.declare_queue("train_results_queue", durable=True)

    # запускаємо consumer у фоні
    asyncio.create_task(consume_results())

    yield

    # ❌ Закриття
    await connection.close()


app = FastAPI(lifespan=lifespan)


# 📤 Відправка повідомлення
@app.post("/send-training-task")
async def test_send_endpoint(model_id: str = "DEFAULT_SYSTEM_MODEL") -> dict:
    message_body = {
        "taskId": "task_987654",
        "modelId": model_id
    }

    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(message_body).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        ),
        routing_key="train_tasks_queue"
    )

    return {"message": "Повідомлення відправлено в чергу train_tasks_queue"}


# 📥 Отримання training data
@app.get("/api/v1/internal/models/{model_id}/training-data")
async def get_training_data(model_id: str) -> dict:
    if not train_file.exists():
        return {"error": "Файл з даними не знайдено"}

    with train_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# 📡 Consumer (слухає чергу)
async def consume_results():
    global connection

    channel = await connection.channel()
    queue = await channel.declare_queue("train_results_queue", durable=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                print(f"📩 Отримано: {message.body.decode()}")

                # 👉 тут твоя логіка
                
                
@app.post("/init_model")
def init(model_id: str = "DEFAULT_SYSTEM_MODEL"):
    data = {
        "modelId": model_id,
        "modelUrl": "https://example.com/model.h5",
        "scalerUrl": "https://example.com/scaler.pkl",
        "labelsUrl": "https://example.com/labels.json"
    }
    response = requests.post(f"{base_url}/init", json=data)
    
    return response.json()

@app.post("/predict_gesture")
def predict(model_id: str = "DEFAULT_SYSTEM_MODEL"):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    data = {
        "modelId": model_id,
        "rawData": test_data
    }
    response = requests.post(f"{base_url}/predict", json=data)
    
    return response.json()
