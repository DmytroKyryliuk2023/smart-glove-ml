from contextlib import asynccontextmanager
from fastapi import FastAPI
from faststream.rabbit.fastapi import RabbitRouter

rabbit_router = RabbitRouter("amqp://guest:guest@localhost:5672/")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запускаємо брокера при старті
    async with rabbit_router.broker:
        yield

app = FastAPI(lifespan=lifespan)
app.include_router(rabbit_router)

async def test_send() -> None:
    await rabbit_router.broker.publish(
        {
            "taskId": "task_987654",
            "modelId": "DEFAULT_SYSTEM_MODEL"
        },
        queue="train_tasks_queue"
    )

@app.post("/test")
async def test_send_endpoint(data: str) -> dict:
    await test_send()
    return {"message": f"Повідомлення {data} відправлено в чергу train_tasks_queue"}
