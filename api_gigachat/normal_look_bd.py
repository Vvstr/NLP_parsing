import pandas as pd
import re
import json

# Функция для извлечения JSON из строки
def extract_json(text):
    """
    Извлекает JSON из строки, где он может быть обернут в блок markdown (```json ... ```).

    Аргументы:
        text (str): Исходная строка с данными контактов.

    Возвращает:
        dict или None: Если JSON успешно извлечён и распарсен, возвращает словарь,
                       иначе возвращает None.
    """
    if not isinstance(text, str):
        return None

    # Ищем блок, начинающийся с ```json и заканчивающийся на ```
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            print(f"Ошибка при разборе JSON: {e}")
            return None
    return None

# Загрузка исходного DataFrame (укажите правильный путь к CSV)
df = pd.read_csv("processed_contacts_with_results.csv")

# Предполагается, что URL находится в первом столбце с именем "0"
# Применяем функцию extract_json к столбцу processed_contacts
df["parsed_contacts"] = df["processed_contacts"].apply(extract_json)

# Создаем новый DataFrame, содержащий URL и распаршенные JSON-данные
new_df = df[["0", "parsed_contacts"]].copy()

# Сохраняем новый DataFrame в CSV для дальнейшей работы
new_df.to_csv("final_data.csv", index=False)

print("Новый датасет сохранён в final_data.csv")