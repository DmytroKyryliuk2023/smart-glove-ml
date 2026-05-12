import asyncio
import json

import aio_pika


class RabbitMQService:
    def __init__(self, rabbit_url: str):
        self.rabbit_url = rabbit_url
        self.connection = None
        self.channel = None

    async def connect(self):
        await asyncio.sleep(10)
        self.connection = await aio_pika.connect_robust(self.rabbit_url)
        self.channel = await self.connection.channel()

    async def declare_queue(self, queue_name: str):
        await self.channel.declare_queue(queue_name, durable=True)

    async def start_consuming(self, queue_name: str, callback):
        queue = await self.channel.declare_queue(queue_name, durable=True)
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                asyncio.create_task(callback(message))

    async def publish_result(self, routing_key: str, message: dict):
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=routing_key,
        )

    async def close(self):
        if self.connection:
            await self.connection.close()