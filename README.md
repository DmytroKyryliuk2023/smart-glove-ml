# Smart Glove ML

[![Dev CI Pipeline](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/dev-ci.yml/badge.svg)](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/dev-ci.yml)

[![Prod CI Pipeline](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/prod-ci.yml/badge.svg)](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/prod-ci.yml)

Machine Learning service for Smart Glove gesture recognition. This project provides a FastAPI-based backend that trains TensorFlow/Keras models and performs real-time gesture classification using sensor data collected from a smart glove.

The service is designed as part of a distributed Smart Glove system and integrates with external services such as RabbitMQ, MinIO, MongoDB, and the Smart Glove Backend.

---

# Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Linting](#linting)
- [Project Structure](#project-structure)
- [Docker](#docker)
- [Sequence Normalization](#sequence-normalization)
- [Example API Request](#example-api-request)
- [License](#license)

---

# Project Overview

Smart Glove ML is responsible for the machine learning functionality of the Smart Glove ecosystem.

The service provides:

- Gesture recognition from smart glove sensor data
- Training TensorFlow/Keras neural network models
- Real-time gesture prediction
- Automatic preprocessing of time-series sensor data
- Model storage in MinIO
- Asynchronous training using RabbitMQ
- REST API built with FastAPI

Unlike the previous standalone Flask prototype, this service is designed as a scalable microservice that communicates with other backend components.

---

# Features

- TensorFlow/Keras neural network for gesture recognition
- Automatic preprocessing of sensor sequences
- Fixed-length sequence normalization
- FastAPI REST API
- Real-time gesture prediction
- Asynchronous model training
- RabbitMQ integration
- MinIO model storage
- MongoDB integration
- Docker support
- Unit and integration tests
- GitHub Actions CI pipelines

---

# Architecture

```
                   Smart Glove Backend
                           │
                           ▼
                    FastAPI Application
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Prediction Service                                         │
│  Training Service                                           │
│  RabbitMQ Service                                           │
│  Storage Service                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
      RabbitMQ              MinIO              MongoDB
```

Main components:

- **Prediction Service** – loads trained models and performs inference.
- **Training Service** – trains new neural network models.
- **RabbitMQ Service** – receives and publishes training jobs.
- **Storage Service** – uploads and downloads models from MinIO.

---

# Technology Stack

- Python 3.12
- FastAPI
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy
- RabbitMQ
- MinIO
- MongoDB
- Docker
- Pytest
- Ruff

---

# Requirements

- Python 3.12+
- RabbitMQ
- MinIO
- MongoDB
- Smart Glove Backend

---

# Installation

## Clone the repository

```bash
git clone <repository-url>
cd smart-glove-ml
```

## Create a virtual environment

Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

Install development dependencies

```bash
pip install -r requirements_dev.txt
```

---

# Configuration

Create a `.env` file in the project root.

```env
MONGO_INITDB_ROOT_USERNAME=
MONGO_INITDB_ROOT_PASSWORD=

RABBITMQ_DEFAULT_USER=
RABBITMQ_DEFAULT_PASS=

MINIO_ROOT_USER=
MINIO_ROOT_PASSWORD=

JWT_SECRET_KEY=
JWT_EXPIRATION=
```

---

# Running the Service

## Local

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at

```
http://localhost:8000
```

## Docker

```bash
cd start_docker

docker-compose up -d
```

Stop the services

```bash
docker-compose down
```

---

# API Documentation

After starting the application:

- Swagger UI

```
http://localhost:8000/docs
```

- ReDoc

```
http://localhost:8000/redoc
```

---

## Predict Gesture

```
POST /predict
```

Request

```json
{
  "modelId": "model_123",
  "rawData": [
    [1.0, 2.0, 3.0],
    [1.2, 2.3, 3.4]
  ]
}
```

Response

```json
{
  "predictedLabel": "ok",
  "confidence": 0.95
}
```

---

## Train Model

Training jobs are submitted through RabbitMQ.

Example message:

```json
{
  "modelId": "model_123"
}
```

Training result:

```json
{
  "modelId": "model_123",
  "status": "SUCCESS",
  "errorMessage": null
}
```

---

# Sequence Normalization

Sensor recordings naturally vary in length. Before training or prediction, every sequence is normalized to **50 time steps**.

- Shorter sequences are linearly interpolated.
- Longer sequences are uniformly resampled.
- Sequences of exactly 50 samples remain unchanged.

This ensures consistent model input dimensions.

---

# Testing

Run all tests

```bash
pytest
```

Run with coverage

```bash
pytest --cov=app --cov-report=html
```

Run unit tests

```bash
pytest tests/unit/
```

Run integration tests

```bash
pytest tests/integration/
```

Run a specific test

```bash
pytest tests/unit/test_models.py
```

---

# Linting

The project uses **Ruff** for linting and formatting.

Check the code

```bash
ruff check app tests
```

Automatically fix issues

```bash
ruff check app tests --fix
```

---

# Project Structure

```
smart-glove-ml/
│
├── app/
│   ├── main.py
│   ├── models.py
│   ├── prediction_service.py
│   ├── training_service.py
│   ├── rabbitmq_service.py
│   └── storages.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── data/
│
├── start_docker/
│
├── start_server/
│
├── Dockerfile
├── requirements.txt
├── requirements_dev.txt
├── pytest.ini
├── run_tests.sh
└── README.md
```

---

# Docker

Build the image

```bash
docker build -t smart-glove-ml .
```

Run using Docker Compose

```bash
cd start_docker

docker-compose up -d
```

Services:

| Service | URL |
|----------|-----|
| Smart Glove ML | http://localhost:8000 |
| Smart Glove Backend | http://localhost:8080 |
| RabbitMQ Management | http://localhost:15672 |
| MinIO Console | http://localhost:9001 |
| MongoDB | localhost:27018 |

---

# Example API Request

```python
import requests

gesture = [
    [1.0, 2.0, 3.0],
    [1.1, 2.1, 3.1],
]

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "modelId": "model_123",
        "rawData": gesture
    }
)

prediction = response.json()

print(prediction["predictedLabel"])
print(prediction["confidence"])
```

---

# Development

Install a new dependency

```bash
pip install package-name
```

Update requirements

```bash
pip freeze > requirements.txt
```

Before creating a pull request, it is recommended to run:

```bash
ruff check app tests
pytest
```

---

# License

This project is licensed under the MIT License.

---

# Author

Developed as part of the **Smart Glove** project and coursework at **Lviv Polytechnic National University**.