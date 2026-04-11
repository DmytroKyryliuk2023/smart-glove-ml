import json
import requests
from pathlib import Path

def main():
    # Шляхи до файлів
    data_folder = Path("data")
    train_file = data_folder / "gestures_merged.json"
    test_file = data_folder / "excuse-me.json"
    model_file = data_folder / "trained_model.json"
    
    # Перевірка існування файлів
    if not train_file.exists():
        print(f"Файл тренування не знайдено: {train_file}")
        return
    
    if not test_file.exists():
        print(f"Файл тестування не знайдено: {test_file}")
        return
    
    base_url = "http://localhost:8000"
    
    # Крок 1: Тренування
    print("Тренування моделі...")
    with open(train_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    response = requests.post(f"{base_url}/train", json=training_data)
    if response.status_code != 200:
        print(f"Помилка тренування: {response.status_code}")
        return
    
    # Зберігаємо модель
    with open(model_file, 'w', encoding='utf-8') as f:
        json.dump(response.json(), f)
    print("Модель натренована та збережена")
    
    # Крок 2: Ініціалізація
    print("Ініціалізація моделі...")
    with open(model_file, 'r', encoding='utf-8') as f:
        encoded_model = json.load(f)
    
    response = requests.post(f"{base_url}/init", json=encoded_model)
    if response.status_code != 200:
        print(f"Помилка ініціалізації: {response.status_code}")
        return
    print("Модель ініціалізована")
    
    # Крок 3: Відправляємо excuse-me.json як є на /predict
    print("\nВідправка excuse-me.json на /predict...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Просто відправляємо файл як є
    response = requests.post(f"{base_url}/predict", json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Результат: {result}")
    else:
        print(f"Помилка: {response.status_code}")
        print(f"Деталі: {response.text}")

if __name__ == "__main__":
    main()