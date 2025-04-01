import pandas as pd
import requests
import time
import os

start_row = 240  # Строка, с которой начинаем обработку

# Пути к файлам
input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processed_contacts.csv")
output_csv = "processed_contacts_with_results.csv"

# Загружаем исходный DataFrame
df = pd.read_csv(input_csv)

# Загружаем существующие результаты, если файл уже есть, а он есть
if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv)
    
    # Если в существующем файле есть колонка "processed_contacts", используем её для заполнения данных в основном DataFrame
    if "processed_contacts" in df_existing.columns:
        df["processed_contacts"] = df_existing["processed_contacts"].fillna("")
    else:
        df["processed_contacts"] = [""] * len(df)
else:
    df["processed_contacts"] = [""] * len(df)

# Функция для отправки текста в API
def process_text(text):
    """
    Отправляет текст на обработку через API FastAPI, запущенный локально.
    
    Аргументы:
        text (str): Текст для обработки.
        
    Возвращает:
        str: Результат обработки, полученный от API.
             Если произошла ошибка, возвращает None.
    """
    url = "http://127.0.0.1:8000/process"
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("result", "")
    except Exception as e:
        print(f"\nОшибка при обработке текста: {e}")
        return None  # None означает ошибку обработки

# Обрабатываем строки, начиная с `start_row`
for i in range(start_row, len(df)):
    if df.iloc[i]["processed_contacts"]:  # Пропускаем уже обработанные
        continue

    print(f"Обработка строки {i + 1}/{len(df)}")

    # Три попытки обработки
    for attempt in range(3):
        result = process_text(df.iloc[i]["1"])
        if result is not None:  # Успешная обработка
            df.at[i, "processed_contacts"] = result
            break
        time.sleep(5)  # Пауза перед повторной попыткой
    else:
        print(f"\nНе удалось обработать строку {i + 1} после 3 попыток")
    
    # Промежуточное сохранение каждые 5 строк или в конце
    if (i + 1) % 5 == 0 or i == len(df) - 1:
        df.to_csv(output_csv, index=False)
        print(f"\nСохранение прогресса до строки {i + 1}")

print(f"\nОбработка завершена. Результаты сохранены в {output_csv}")
