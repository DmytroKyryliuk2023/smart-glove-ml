import asyncio
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException, status

import models as Models
from rabbitmq_service import RabbitMQService
from training_service import TrainingService
from prediction_service import PredictionService


RABBIT_URL = "amqp://guest:guest@rabbitmq:5672/"

rabbitmq = RabbitMQService(RABBIT_URL)
training_service = TrainingService()
prediction_service = PredictionService()
local_models: dict[str, Models.Model] = {}


async def process_message(message):
    async with message.process():
        try:
            body = json.loads(message.body.decode())
            model_id = body.get("modelId")
            
            await training_service.train_model(model_id)
            
            result = {
                "modelId": model_id,
                "status": "SUCCESS",
                "errorMessage": None
            }
            await rabbitmq.publish_result("train_results_queue", result)
            
        except Exception as e:
            error_message = str(e)
            print(f"Consumer error: {e}")
            
            result = {
                "modelId": body.get("modelId"),
                "status": "FAILED",
                "errorMessage": error_message,
            }
            await rabbitmq.publish_result("train_results_queue", result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rabbitmq.connect()
    await rabbitmq.declare_queue("train_tasks_queue")
    await rabbitmq.declare_queue("train_results_queue")
    asyncio.create_task(rabbitmq.start_consuming("train_tasks_queue", process_message))
    yield
    await rabbitmq.close()


app = FastAPI(lifespan=lifespan)


@app.post("/models/{model_id}")
async def init_model(model_id: str):
    try:
        local_models[model_id] = await training_service.storage.load_model(model_id)
        print(f"Модель {model_id} завантажена в пам'ять")
        return {"message": "Model initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize model: {str(e)}",
        )


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    if model_id in local_models:
        del local_models[model_id]
        print(f"Модель {model_id} видалена з пам'яті")
        return {"message": "Model deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not found",
        )


@app.post("/predict")
async def predict_gesture(gesture: Models.GestureData):
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

    if len(gesture_data[0]) != training_service.EXPECTED_COLUMNS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {training_service.EXPECTED_COLUMNS} columns, but got {len(gesture_data[0])}",
        )

    current_model = local_models[model_id]
    response = await prediction_service.predict(current_model, gesture_data)
    return response