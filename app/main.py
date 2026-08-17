import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, status
from gesture_detection_service import GestureDetectionService

from . import models
from .division_service import DivisionService
from .gesture_service import GestureService
from .rabbitmq_service import RabbitMQService
from .training_service import TrainingService

RABBIT_URL = os.getenv("RABBITMQ_URL")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
SERVER_ENDPOINT = os.getenv("SERVER_ENDPOINT")

SEQUENCE_LENGTH = 50
NUM_FEATURES = 18

CLOSE_POINTS_THRESHOLD = 30
MIN_GESTURE_LENGTH = 100

CONFIDENCE_THRESHOLD = 0.7
WINDOW_SIZE = 223

rabbitmq = RabbitMQService(RABBIT_URL)
training_service = TrainingService(
    minio_endpoint=MINIO_ENDPOINT,
    minio_access_key=MINIO_ACCESS_KEY,
    minio_secret_key=MINIO_SECRET_KEY,
    server_endpoint=SERVER_ENDPOINT,
)
division_service = DivisionService(
    confidence_threshold=CONFIDENCE_THRESHOLD,
    window_size=WINDOW_SIZE,
    num_features=NUM_FEATURES,
)
gesture_service = GestureService(sequence_length=50)
gesture_detection_service = GestureDetectionService(
    division_service=division_service,
    gesture_service=gesture_service,
    close_points_threshold=CLOSE_POINTS_THRESHOLD,
    min_gesture_length=MIN_GESTURE_LENGTH,
)


async def process_message(message):
    async with message.process():
        try:
            body = json.loads(message.body.decode())
            model_id = body.get("modelId")

            await training_service.train_model(model_id)

            result = {"modelId": model_id, "status": "SUCCESS", "errorMessage": None}
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


@app.post("/models/gesture/{model_id}")
async def init_gesture_model(model_id: str):
    try:
        if model_id in gesture_service.local_models:
            return {"message": "Gesture model already initialized"}

        gesture_service.local_models[
            model_id
        ] = await training_service.storage.load_model(model_id)
        print(f"Gesture model {model_id} loaded into memory")
        return {"message": "Gesture model initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize gesture model: {str(e)}",
        )


@app.delete("/models/gesture/{model_id}")
async def delete_gesture_model(model_id: str):
    if model_id in gesture_service.local_models:
        del gesture_service.local_models[model_id]
        print(f"Gesture model {model_id} deleted from memory")
        return {"message": "Gesture model deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gesture model not found",
        )


@app.post("/models/division")
async def init_division_model(model_id: str = "default"):
    try:
        if model_id in division_service.local_models:
            return {"message": "Division model already initialized"}

        division_service.local_models[model_id] = await training_service.storage.load_model(
            model_id
        )
        print(f"Division model {model_id} loaded into memory")
        return {"message": "Division model initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize division model: {str(e)}",
        )


@app.delete("/models/division")
async def delete_division_model(model_id: str = "default"):
    if model_id in division_service.local_models:
        del division_service.local_models[model_id]
        print(f"Division model {model_id} deleted from memory")
        return {"message": "Division model deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Division model not found",
        )


@app.post("/predict/gesture")
async def predict_gesture(gesture: models.GesturePredictionData):
    gesture_model_id, gesture_data = gesture.gestureModelId, gesture.rawData

    if gesture_model_id not in gesture_service.local_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No gesture model initialized",
        )

    if not gesture_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format or empty 'rawData' array",
        )

    if len(gesture_data[0]) != gesture_service.num_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {gesture_service.num_features} columns, \
                but got {len(gesture_data[0])}",
        )

    gesture_model = gesture_service.local_models[gesture_model_id]
    response = await gesture_service.predict(gesture_model, gesture_data)
    return response


@app.websocket("/predict/sequence")
async def predict_sequence(
    websocket: WebSocket, PredictSequenceData: models.SequencePredictionData
):
    await websocket.accept()
    print("Client connected to WebSocket")

    gesture_model_id, division_model_id = (
        PredictSequenceData.gestureModelId,
        PredictSequenceData.divisionModelId,
    )

    if gesture_model_id not in gesture_service.local_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No gesture model initialized",
        )

    if division_model_id not in division_service.local_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No division model initialized",
        )

    gesture_model = gesture_service.local_models[gesture_model_id]
    division_model = division_service.local_models[division_model_id]

    try:
        detected_starts = []
        detected_ends = []
        stream = []

        while True:
            request = await websocket.receive_json()

            step = request.get("data")

            if step is None:
                await websocket.send_json(
                    {"status": "error", "message": "No data provided"}
                )
                continue

            stream.extend(step)

            is_end_request = request.get("status", "streaming") == "end"

            if len(stream) < DivisionService.WINDOW:
                if is_end_request:
                    needed = DivisionService.WINDOW - len(stream)
                    stream.extend([[0.0] * DivisionService.FEATURES] * needed)
                else:
                    continue

            (
                response,
                stream,
                detected_starts,
                detected_ends,
                should_break,
            ) = await gesture_detection_service.process_window(
                division_model,
                gesture_model,
                stream,
                detected_starts,
                detected_ends,
                is_end_request,
            )

            if response is not None:
                await websocket.send_json(response)

            if should_break:
                break

    except Exception as e:
        print(f"Error in WebSocket stream: {e}")
        await websocket.send_json(
            {"status": "error", "message": f"Server error: {str(e)}"}
        )

    finally:
        print("Client disconnected")
