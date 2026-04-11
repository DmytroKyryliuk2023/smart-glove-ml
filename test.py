import json
import requests
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

class GestureRecognitionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.train_url = f"{base_url}/train"
        self.init_url = f"{base_url}/init"
        self.predict_url = f"{base_url}/predict"
    
    def load_json_data(self, file_path: str) -> Dict[str, List[List[List[float]]]]:
        """
        Завантажує JSON файл з даними жестів.
        Очікує формат: {"gesture_name": [[[x1,y1,...], [x2,y2,...]], ...]}
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def train_model(self, data_file_path: str, output_json_path: str = "trained_model.json"):
        """
        Відправляє дані на тренування та зберігає отриману модель
        """
        print(f"📚 Завантаження даних з {data_file_path}...")
        training_data = self.load_json_data(data_file_path)
        
        print(f"🚀 Відправка даних на тренування...")
        print(f"   Кількість жестів: {len(training_data)}")
        for gesture_name, sequences in training_data.items():
            print(f"   - {gesture_name}: {len(sequences)} послідовностей")
        
        response = requests.post(self.train_url, json=training_data)
        
        if response.status_code == 200:
            encoded_model = response.json()
            
            # Зберігаємо модель у JSON файл
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(encoded_model, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Модель успішно натренована та збережена в {output_json_path}")
            print(f"   Довжина послідовності: {encoded_model['sequence_length']}")
            print(f"   Очікувана кількість ознак: {encoded_model['expected_columns']}")
            print(f"   Класи: {encoded_model['classes']}")
            
            return encoded_model
        else:
            print(f"❌ Помилка тренування: {response.status_code}")
            print(f"   Деталі: {response.text}")
            return None
    
    def init_model(self, model_json_path: str):
        """
        Ініціалізує модель на сервері з раніше збереженого JSON
        """
        print(f"📂 Завантаження моделі з {model_json_path}...")
        
        with open(model_json_path, 'r', encoding='utf-8') as f:
            encoded_model = json.load(f)
        
        print(f"🔄 Ініціалізація моделі на сервері...")
        response = requests.post(self.init_url, json=encoded_model)
        
        if response.status_code == 200:
            print(f"✅ Модель успішно ініціалізована")
            return True
        else:
            print(f"❌ Помилка ініціалізації: {response.status_code}")
            print(f"   Деталі: {response.text}")
            return False
    
    def predict_gesture(self, gesture_data: List[List[float]]) -> Dict[str, Any]:
        """
        Відправляє дані жесту на передбачення
        """
        payload = {"gesture_data": gesture_data}
        response = requests.post(self.predict_url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Помилка передбачення: {response.status_code}")
            print(f"   Деталі: {response.text}")
            return None
    
    def test_gestures_from_file(self, gestures_file_path: str):
        """
        Тестує модель на всіх жестах з файлу
        """
        print(f"\n🧪 Тестування жестів з {gestures_file_path}...")
        
        gestures_data = self.load_json_data(gestures_file_path)
        
        results = {}
        
        for gesture_name, sequences in gestures_data.items():
            print(f"\n📊 Тестування жесту: {gesture_name}")
            print(f"   Кількість послідовностей: {len(sequences)}")
            
            gesture_results = []
            
            for i, sequence in enumerate(sequences):
                prediction = self.predict_gesture(sequence)
                
                if prediction:
                    gesture_results.append({
                        "sequence_index": i,
                        "prediction": prediction["prediction"],
                        "confidence": prediction["confidence"],
                        "is_correct": prediction["prediction"] == gesture_name
                    })
                    
                    status = "✅" if prediction["prediction"] == gesture_name else "❌"
                    print(f"   {status} Послідовність {i+1}: "
                          f"передбачено '{prediction['prediction']}' "
                          f"(впевненість: {prediction['confidence']:.3f})")
            
            # Статистика для цього жесту
            if gesture_results:
                correct_count = sum(1 for r in gesture_results if r["is_correct"])
                accuracy = correct_count / len(gesture_results)
                results[gesture_name] = {
                    "total": len(gesture_results),
                    "correct": correct_count,
                    "accuracy": accuracy,
                    "details": gesture_results
                }
                print(f"   📈 Точність для '{gesture_name}': {accuracy*100:.1f}% ({correct_count}/{len(gesture_results)})")
        
        # Загальна статистика
        if results:
            total_predictions = sum(r["total"] for r in results.values())
            total_correct = sum(r["correct"] for r in results.values())
            overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
            
            print(f"\n🎯 ЗАГАЛЬНА СТАТИСТИКА:")
            print(f"   Всього передбачень: {total_predictions}")
            print(f"   Всього правильних: {total_correct}")
            print(f"   Загальна точність: {overall_accuracy*100:.1f}%")
            
            # Виведення матриці помилок (спрощено)
            print(f"\n📋 ДЕТАЛІ ПОМИЛОК:")
            for gesture_name, result in results.items():
                errors = [r for r in result["details"] if not r["is_correct"]]
                if errors:
                    print(f"   {gesture_name}:")
                    for err in errors:
                        print(f"      → Посл.{err['sequence_index']+1} передбачено як '{err['prediction']}'")
        
        return results


def main():
    # Шляхи до файлів (змініть якщо потрібно)
    data_folder = Path("data")
    train_file = data_folder / "gestures_merged.json"
    test_file = data_folder / "excuse-me.json"
    model_file = "trained_model.json"
    
    # Перевірка існування файлів
    if not train_file.exists():
        print(f"❌ Файл тренування не знайдено: {train_file}")
        print("   Переконайтеся, що файл gestures_merged.json існує в папці 'data'")
        return
    
    if not test_file.exists():
        print(f"⚠️ Файл тестування не знайдено: {test_file}")
        print("   Пропускаємо тестування жестів")
    
    # Створюємо клієнта
    client = GestureRecognitionClient(base_url="http://localhost:8000")
    
    # Крок 1: Тренування моделі
    print("\n" + "="*60)
    print("КРОК 1: ТРЕНУВАННЯ МОДЕЛІ")
    print("="*60)
    
    encoded_model = client.train_model(str(train_file), str(model_file))
    
    if not encoded_model:
        print("❌ Не вдалося натренувати модель. Завершення роботи.")
        return
    
    # Крок 2: Ініціалізація моделі на сервері
    print("\n" + "="*60)
    print("КРОК 2: ІНІЦІАЛІЗАЦІЯ МОДЕЛІ")
    print("="*60)
    
    if not client.init_model(str(model_file)):
        print("❌ Не вдалося ініціалізувати модель. Завершення роботи.")
        return
    
    # Крок 3: Тестування жестів
    if test_file.exists():
        print("\n" + "="*60)
        print("КРОК 3: ТЕСТУВАННЯ ЖЕСТІВ")
        print("="*60)
        
        client.test_gestures_from_file(str(test_file))
    else:
        # Альтернативне тестування з тренувальних даних
        print("\n" + "="*60)
        print("КРОК 3: ТЕСТУВАННЯ (використовуючи тренувальні дані)")
        print("="*60)
        print("Файл excuse-me.json не знайдено. Тестуємо перший жест з тренувальних даних...")
        
        training_data = client.load_json_data(str(train_file))
        first_gesture_name = list(training_data.keys())[0]
        first_sequence = training_data[first_gesture_name][0]
        
        print(f"\n📊 Тестування жесту: {first_gesture_name}")
        prediction = client.predict_gesture(first_sequence)
        
        if prediction:
            print(f"   Результат: {prediction['prediction']}")
            print(f"   Впевненість: {prediction['confidence']:.3f}")
            print(f"   Правильно: {'✅' if prediction['prediction'] == first_gesture_name else '❌'}")


if __name__ == "__main__":
    # Перевірка, чи запущений сервер
    try:
        response = requests.get("http://localhost:8000/docs", timeout=2)
        print("✅ Сервер FastAPI доступний")
    except:
        print("⚠️ УВАГА: Сервер FastAPI не відповідає!")
        print("   Переконайтеся, що сервер запущено на http://localhost:8000")
        print("   Запустіть сервер командою: uvicorn main:app --reload")
        print()
    
    main()