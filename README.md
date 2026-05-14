# Smart Glove ML

[![CI/CD Pipeline](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/ci-cd.yml)

Машинне навчання для розпізнавання жестів розумної рукавиці. ML-сервіс для розпізнавання та класифікації жестів з гру рукавиці, побудований на FastAPI з використанням TensorFlow/Keras.

## 📋 Таблиця змісту

- [Опис проєкту](#опис-проєкту)
- [Особливості](#особливості)
- [Архітектура](#архітектура)
- [Вимоги](#вимоги)
- [Встановлення](#встановлення)
- [Налаштування](#налаштування)
- [Запуск](#запуск)
- [API документація](#api-документація)
- [Лінтинг](#літинг)
- [Тестування](#тестування)
- [Структура проєкту](#структура-проєкту)
- [Docker](#docker)

## 📝 Опис проєкту

Smart Glove ML — це бекенд-сервіс для машинного навчання, який:

- **Розпізнає жести** з датчиків розумної рукавиці
- **Тренує моделі** на основі історичних даних жестів
- **Робить прогнози** жестів у реальному часі
- **Зберігає моделі** в об'єктному сховищі (MinIO)
- **Обробляє завдання** через чергу повідомлень (RabbitMQ)
- **Нормалізує послідовності** даних датчиків до стандартної довжини

## ✨ Особливості

- 🚀 **FastAPI** — сучасний веб-фреймворк для побудови API
- 🤖 **TensorFlow/Keras** — глибоке навчання для розпізнавання жестів
- 📦 **MinIO** — об'єктне сховище для моделей та даних
- 📨 **RabbitMQ** — асинхронна обробка завдань тренування
- 🧠 **Scikit-learn** — попередня обробка даних і масштабування
- 🐍 **Python 3.12** — сучасна версія Python
- 🐳 **Docker** — контейнеризація для легкого розгортання
- ✅ **Unit тести** — покриття основних компонентів

## 🏗️ Архітектура

```
┌─────────────────────────────────────┐
│      FastAPI Application            │
├─────────────────────────────────────┤
│  ├─ TrainingService                 │
│  ├─ PredictionService               │
│  ├─ RabbitMQService                 │
│  └─ StorageService                  │
├─────────────────────────────────────┤
│        External Services            │
├─────────────────────────────────────┤
│  ├─ SmartGlove Backend              │
│  ├─ RabbitMQ (Message Queue)        │
│  ├─ MinIO (Model Storage)           │
│  └─ MongoDB (Data Storage)          │
└─────────────────────────────────────┘
```

### Компоненти

- **main.py** — точка входу додатка, налаштування FastAPI
- **models.py** — дата-класи моделей, логіка вирівнювання послідовностей
- **training_service.py** — навчання моделей нейронних мереж
- **prediction_service.py** — робота прогнозів на основі навчених моделей
- **rabbitmq_service.py** — інтеграція з RabbitMQ
- **storages.py** — робота з зберіганням в MinIO

## 📦 Вимоги

- Python 3.12+
- SmartGlove Backend
- RabbitMQ
- MinIO
- MongoDB (опціонально)

## 🔧 Встановлення

### 1. Клонування репозиторію

```bash
git clone <repo_url>
cd smart-glove-ml
```

### 2. Створення віртуального окруження

```bash
python3 -m venv .venv
source .venv/bin/activate  # На macOS/Linux
# або
.venv\Scripts\activate  # На Windows
```

### 3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

Для розробки встановіть також залежності для тестування:

```bash
pip install -r requirements.dev.txt
```

## ⚙️ Налаштування

Створіть файл `.env` в кореневій директорії проєкту з наступними змінними:

```bash
MONGO_INITDB_ROOT_USERNAME
MONGO_INITDB_ROOT_PASSWORD

RABBITMQ_DEFAULT_USER
RABBITMQ_DEFAULT_PASS

MINIO_ROOT_USER
MINIO_ROOT_PASSWORD

JWT_SECRET_KEY
JWT_EXPIRATION
```

## 🚀 Запуск

### Локальний запуск

```bash
# Активуйте віртуальне окруження
source .venv/bin/activate

# Запустіть сервер
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Сервер буде доступний за адресою `http://localhost:8000`

### Запуск через Docker

```bash
# Запустіть всі сервіси через docker-compose
cd start_docker
docker-compose up -d

# Перевірте статус контейнерів
docker-compose ps
```

Зупинка сервісів:

```bash
docker-compose down
```

## 📚 API документація

Після запуску сервера, документація доступна за адресами:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Основні ендпоінти

#### Прогноз жесту

```http
POST /predict
Content-Type: application/json

{
  "modelId": "model_123",
  "rawData": [[1.0, 2.0, 3.0, ...], [1.5, 2.1, 3.2, ...], ...]
}
```

**Відповідь:**

```json
{
  "predictedLabel": "ok",
  "confidence": 0.95
}
```

#### Подання завдання на тренування

Завдання на тренування відправляються через RabbitMQ в чергу `train_queue`.

Повідомлення повинно мати структуру:

```json
{
  "modelId": "model_123"
}
```

Результати тренування публікуються в `train_results_queue`:

```json
{
  "modelId": "model_123",
  "status": "SUCCESS",
  "errorMessage": null
}
```

## 🛠️ Лінтинг

У цьому проєкті використовується `ruff` для перевірки стилю, форматування та швидкого виявлення проблем у коді Python.

### Запуск `ruff`

```bash
ruff check app tests
```

### Виправлення проблем автоматично

```bash
ruff check app tests --fix
```

### Рекомендації

- Виконуйте `ruff` разом із `pytest` перед комітом.
- Додайте `ruff` до CI/CD для автоматичної перевірки якості коду.

## ✅ Тестування

### Запуск всіх тестів

```bash
pytest
```

### Запуск з покриттям

```bash
pytest --cov=app --cov-report=html
```

### Запуск специфічних тестів

```bash
# Unit тести
pytest tests/unit/

# Інтеграційні тести
pytest tests/integration/

# Конкретний файл тестів
pytest tests/unit/test_models.py
```

### Запуск через shell скрипт

```bash
bash run_tests.sh
```

## 📁 Структура проєкту

```
smart-glove-ml/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI приложение
│   ├── models.py                    # Дата-класи і утиліти
│   ├── prediction_service.py        # Сервіс прогнозів
│   ├── training_service.py          # Сервіс тренування
│   ├── rabbitmq_service.py          # Інтеграція RabbitMQ
│   └── storages.py                  # Робота зі сховищами
├── tests/
│   ├── unit/                        # Unit тести
│   │   ├── test_main.py
│   │   ├── test_models.py
│   │   ├── test_prediction_service.py
│   │   ├── test_rabbitmq_service.py
│   │   ├── test_training_service.py
│   │   └── test_storages.py
│   ├── integration/                 # Інтеграційні тести
│   │   └── test_backend_api.py
│   └── conftest.py                  # Pytest конфігурація
├── data/
│   ├── excuse-me.json              # Приклади даних
│   └── gestures_merged.json        # Об'єднані дані жестів
├── start_docker/
│   └── docker-compose.yml          # Конфігурація Docker Compose
├── start_server/
│   ├── start_server.sh             # Shell скрипт запуску
│   └── start_server.ps1            # PowerShell скрипт запуску
├── Dockerfile                       # Конфігурація Docker
├── requirements.txt                 # Залежності проєкту
├── requirements_dev.txt             # Залежності розробки
├── pytest.ini                       # Конфігурація pytest
├── run_tests.sh                     # Скрипт для запуску тестів
└── README.md                        # Цей файл
```

## 🐳 Docker

### Побудова образу

```bash
docker build -t smart-glove-ml:latest .
```

### Docker Compose

Для запуску всієї системи (ML сервіс + RabbitMQ + MinIO + MongoDB):

```bash
cd start_docker
docker-compose up -d
```

Сервіси будуть доступні за адресами:

- **Smart Glove ML API**: http://localhost:8000
- **Smart Glove Backend**: http://localhost:8080
- **RabbitMQ Management**: http://localhost:15672
- **MinIO Web UI**: http://localhost:9001
- **MongoDB**: localhost:27018

## 🔍 Нормалізація послідовностей

Сервіс автоматично нормалізує послідовності датчиків до фіксованої довжини (50 точок):

- **Якщо даних менше** → інтерполяція лінійна інтерполяція
- **Якщо даних більше** → рівномірний вибір точок
- **Якщо точно 50** → без змін

Це забезпечує консистентність вхідних даних для моделей.

## 📊 Приклад роботи з API

```python
import requests
import json

# Дані з датчиків жесту
gesture_data = [
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.1, 2.1, 3.1, 4.1, 5.1],
    # ... більше точок
]

# Запит на прогноз
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "modelId": "model_123",
        "rawData": gesture_data
    }
)

result = response.json()
print(f"Прогноз: {result['predictedLabel']}")
print(f"Впевненість: {result['confidence']:.2%}")
```

## 🛠️ Розробка

### Додавання нових залежностей

```bash
# Встановіть пакет
pip install new-package

# Оновіть requirements.txt
pip freeze > requirements.txt
```

## 📝 Ліцензія

MIT License

## 👨‍💻 Автор

Проєкт розроблений як частина навчальної роботи на 6-му семестрі.