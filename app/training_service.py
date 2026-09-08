import json

import httpx

from .gesture_service import GestureService
from .rabbitmq_service import RabbitMQService
from .storage_service import MinioStorage


class TrainingService:
    def __init__(
        self,
        rabbitmq: RabbitMQService,
        gesture_service: GestureService,
        storage_service: MinioStorage,
        server_endpoint: str
    ):
        self.rabbitmq = rabbitmq
        self.gesture_service = gesture_service
        self.storage_service = storage_service
        self.server_endpoint = server_endpoint
        self.timeout = httpx.Timeout(
            connect=5.0,
            read=30.0,
            write=10.0,
            pool=5.0,
        )
        
    async def process_message(self, message):
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                model_id = body.get("modelId")

                await self.train_gesture_model(self.gesture_service, model_id)

                result = {"modelId": model_id, "status": "SUCCESS", "errorMessage": None}
                await self.rabbitmq.publish_result("train_results_queue", result)

            except Exception as e:
                error_message = str(e)
                print(f"Consumer error: {e}")

                result = {
                    "modelId": body.get("modelId"),
                    "status": "FAILED",
                    "errorMessage": error_message,
                }
                await self.rabbitmq.publish_result("train_results_queue", result)

    async def train_gesture_model(
        self, gesture_service: GestureService, model_id: str
    ) -> None:
        print(f"Отримано задачу на тренування: modelId={model_id}")

        training_data = await self.fetch_training_data(model_id)
        print(f"Отримано дані для моделі {model_id}")

        model = await gesture_service.train(training_data)
        await self.storage_service.save_gesture_model(model_id, model)
        print(f"Модель {model_id} успішно натренована")
        
    async def fetch_training_data(self, model_id: str):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{self.server_endpoint}/api/v1/internal/models/{model_id}/training-data"
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
