import httpx

from .gesture_service import GestureService
from .storage_service import MinioStorage


class TrainingService:
    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        storage_service: MinioStorage,
        server_endpoint: str,
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.storage_service = storage_service
        self.server_endpoint = server_endpoint
        self.timeout = httpx.Timeout(
            connect=5.0,
            read=30.0,
            write=10.0,
            pool=5.0,
        )

    async def fetch_training_data(self, model_id: str):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{self.server_endpoint}/api/v1/internal/models/{model_id}/training-data"
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def train_gesture_model(
        self, gesture_service: GestureService, model_id: str
    ) -> None:
        print(f"Отримано задачу на тренування: modelId={model_id}")

        training_data = await self.fetch_training_data(model_id)
        print(f"Отримано дані для моделі {model_id}")

        model = await gesture_service.train(training_data)
        await self.storage_service.save_gesture_model(model_id, model)
        print(f"Модель {model_id} успішно натренована")
