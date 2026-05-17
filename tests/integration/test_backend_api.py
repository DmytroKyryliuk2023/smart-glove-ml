import json
import re
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
START_DOCKER_DIR = ROOT / "start_docker"
EXCUSE_ME_FILE = ROOT / "data" / "excuse-me.json"
BASE_URL = "http://localhost:8080"


def docker_compose_command(args, capture_output: bool = False) -> str:
    commands = [
        ["docker", "compose"] + args,
        ["docker-compose"] + args,
    ]

    last_error = None

    for cmd in commands:
        try:
            completed = subprocess.run(
                cmd,
                cwd=START_DOCKER_DIR,
                check=True,
                capture_output=capture_output,
                text=True,
                timeout=120,
            )

            return completed.stdout if capture_output else ""

        except FileNotFoundError as exc:
            last_error = exc

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Docker Compose command timed out: {' '.join(cmd)}"
            ) from exc

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Docker Compose command failed: {' '.join(cmd)}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

    raise RuntimeError(
        "Docker Compose is not installed or available as "
        "'docker compose' or 'docker-compose'."
    ) from last_error


def wait_for_backend_ready(timeout: int = 120):
    """Wait for the backend to be ready by checking logs for initialization message."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["docker", "logs", "smartglove-backend"],
                capture_output=True,
                text=True,
                check=True,
            )
            if re.search(r"Ініціалізація системи успішно завершена", result.stdout):
                return
        except subprocess.CalledProcessError:
            pass  # Container might not be ready yet
        time.sleep(2)
    raise RuntimeError("Backend did not initialize within the timeout period")


@pytest.fixture(scope="session")
def docker_compose():
    docker_compose_command(["up", "-d", "--build"])

    # Чекаємо поки backend не виведе повідомлення про ініціалізацію
    wait_for_backend_ready()

    # Додатковий час для стабілізації системи
    time.sleep(2)

    try:
        yield
    finally:
        docker_compose_command(["down", "-v"])


def load_test_gesture():
    with EXCUSE_ME_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def wait_for_initialization(client: httpx.Client, headers: dict, timeout: int = 60):
    start_time = time.time()
    last_response = None

    while time.time() - start_time < timeout:
        response = client.post(
            f"{BASE_URL}/api/predict/init/default",
            headers=headers,
        )

        if response.status_code == 200:
            return response

        last_response = response
        if response.status_code not in {403, 404, 502, 503, 504}:
            break

        time.sleep(3)

    return last_response


@pytest.mark.integration
def test_full_system_integration(docker_compose):
    email = f"pytest+{uuid.uuid4().hex[:8]}@example.com"
    password = "SecurePass123!"

    raw_data = load_test_gesture()

    with httpx.Client(timeout=30.0) as client:
        register_response = client.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": email,
                "password": password,
            },
        )

        assert register_response.status_code == 200, register_response.text

        token = register_response.json().get("token")

        assert token, register_response.text

        headers = {
            "Authorization": f"Bearer {token}",
        }

        # Додатковий час після реєстрації, щоб бекенд закінчив асинхронні налаштування
        time.sleep(5)

        init_response = wait_for_initialization(client, headers)

        assert init_response is not None, "Init request did not return any response"
        assert init_response.status_code == 200, f"Init failed: {init_response.text}"
        assert init_response.json().get("status") == "SUCCESS"

        time.sleep(5)

        predict_response = client.post(
            f"{BASE_URL}/api/predict/gesture",
            json={
                "modelId": "default",
                "rawData": raw_data,
            },
            headers=headers,
        )

        assert predict_response.status_code == 200, predict_response.text

        predict_body = predict_response.json()

        assert "predictedLabel" in predict_body

        assert 0.0 <= float(predict_body.get("confidence", 0)) <= 1.0
