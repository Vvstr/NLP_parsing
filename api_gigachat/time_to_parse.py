from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import uuid
import base64
from time import time
import logging
import json
import logging.config
from bs4 import BeautifulSoup, Comment
import re


# Здесь через config просто потому что я вспомнил что конфиги это хорошо
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logging_config.json"), "r") as f:
        logging_config = json.load(f)
        logging.config.dictConfig(logging_config)
except Exception as e:
    logging.basicConfig(level=logging.INFO)
    logging.error(f"Ошибка загрузки конфигурации логирования: {e}")


logger = logging.getLogger(__name__)
# .env у меня находится в корневой папке, хоть он и используется только здесь, решил так оставить
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path)
# Всё возможно, вдруг больше 5 секунд будем ждать(вроде timeout по defaultу = 5)
app = FastAPI(timeout=60)

# Константы для APIшки
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_CLIENT_ID = os.getenv("GIGACHAT_CLIENT_ID")
GIGACHAT_CLIENT_SECRET = os.getenv("GIGACHAT_CLIENT_SECRET")
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
TOKEN_CACHE = {"token": None, "expires_at": 0}
MAX_RETRIES = 3
TOKEN_REFRESH_MARGIN = 300

# Валидация это хорошо
class TaskRequest(BaseModel):
    text: str


def get_access_token():
    """
    Получает токен доступа для взаимодействия с GigaChat API.
    
    - Проверяет наличие валидного токена в кэше и возвращает его, если он ещё не истёк.
    - Если токен отсутствует или близок к истечению, делает запрос на AUTH_URL
      для получения нового токена.
    - Кэширует новый токен и время его истечения.
    
    Возвращает:
        str: Токен доступа.
    
    В случае ошибки запроса выбрасывает HTTPException с кодом 500.
    """
    current_time = time()
    if TOKEN_CACHE["token"] and (current_time < (TOKEN_CACHE["expires_at"] - TOKEN_REFRESH_MARGIN)):
        return TOKEN_CACHE["token"]
    
    # Обнуляем кэш, т.к. у меня вылетало во время работы из-за этого
    TOKEN_CACHE["token"] = None
    TOKEN_CACHE["expires_at"] = 0

    # Из документации сбера
    credentials = f"{GIGACHAT_CLIENT_ID}:{GIGACHAT_CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    # Из документации сбера
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {encoded_credentials}'
    }
    
    data = {'scope': 'GIGACHAT_API_PERS'}
    
    try:
        logger.info("Запрос токена доступа к GigaChat API")
        response = requests.post(AUTH_URL, headers=headers, data=data, verify=False)
        response.raise_for_status()
        token_data = response.json()
        TOKEN_CACHE['token'] = token_data['access_token']
        # Unix время вроде
        TOKEN_CACHE['expires_at'] = current_time + token_data['expires_at']
        return TOKEN_CACHE['token']
    except Exception as e:
        logger.error(f"Ошибка при запросе токена: {e}")
        logger.error(f"Ответ сервера: {response.status_code} {response.text}")
        raise HTTPException(status_code=500, detail="Ошибка аутентификации в GigaChat API")


def preprocess_html(html: str, max_length: int = None) -> str:
    """
    Предобработка HTML-содержимого.
    
    Функция выполняет следующие шаги:
    - Заменяет заэкранированные символы (&nbsp; и &amp;) на обычные.
    - Используя BeautifulSoup, удаляет теги script, style, комментарии и скрытые элементы.
    - Извлекает текст из основных блоков (main, article, body, div, footer).
    - Убирает лишние пробелы.
    - При наличии параметра max_length, обрезает текст до указанного количества символов.
    
    Аргументы:
        html (str): Исходный HTML-контент.
        max_length (int, optional): Максимальная длина итогового текста.
    
    Возвращает:
        str: Очищенный и обработанный текст.
    """
    try:
        # Замена заэкранированных символов
        html = html.replace('&nbsp;', ' ').replace('&amp;', '&')
        soup = BeautifulSoup(html, 'html.parser')

        # Удаляем теги script и style
        for tag in soup(['script', 'style']):
            tag.decompose()

        # Удаляем комментарии
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Удаляем скрытые элементы (display:none или visibility:hidden, там могут только пасхалки хранится)
        for tag in soup.find_all(style=re.compile(r'display:\s*none|visibility:\s*hidden')):
            tag.decompose()

        # Извлекаем текст из основных блоков
        content_tags = soup.find_all(['main', 'article', 'body', 'div', 'footer'])
        text_parts = []
        for tag in content_tags:
            text_parts.extend(tag.stripped_strings)

        text = ' '.join(text_parts)
        text = re.sub(r'\s+', ' ', text).strip()

        # Обрезаем текст до указанного максимума(нужно чтобы порезать htmlки как у доменныого сайта с 525к токенов)
        if max_length and len(text) > max_length:
            text = text[:max_length]

        return text
    except Exception as e:
        logger.error(f"Ошибка предобработки HTML: {str(e)}")
        return ""


@app.post("/process")
async def process_task(request: TaskRequest):
    """
    Обработчик POST-запроса на эндпоинте /process.
    
    Получает текст из запроса, определяет, содержит ли он HTML-контент,
    и если да, выполняет его предобработку. Затем формируется промпт для GigaChat API,
    который требует извлечения контактных данных из текста. При успешном запросе к API
    возвращает результат в формате JSON.
    
    Аргументы:
        request (TaskRequest): Объект запроса, содержащий поле text.
    
    Возвращает:
        dict: Словарь с ключом "result", содержащим ответ от GigaChat API.
    """
    text = request.text
    if "<html" in text.lower():
        logger.info("Обнаружен HTML-контент, выполняется предобработка")
        text = preprocess_html(text, max_length=10000)

    # Формирование промпта для Gigachat API с требованиями к извлечению контактов
    prompt = f"""
                Извлеки **ВСЕ** контактные данные из текста, включая:
                1. Телефоны (в любом формате: +7, 8, международные)
                2. Email-адреса (учти регистр)
                3. Физические адреса (с индексом, городом, улицей)
                4. Ссылки на соцсети (Instagram, Telegram, VK, WhatsApp и т.д.)
                5. Ссылки на карты (Яндекс/Google Maps)
                6. Другие контакты (например, ссылки на профили, мессенджеры)

                **Требования:**
                - Верни ТОЛЬКО валидные данные.
                - Нормализуй телефоны в формат +7XXXXXXXXXX.
                - Разделяй адреса на компоненты (город, улица, дом).
                - Игнорируй дубликаты.

                Формат ответа (строго JSON):
                ```json
                {{
                    "phones": ["+79528190301", ...],
                    "emails": ["contact@example.com", ...],
                    "addresses": ["Москва, ул. Пушкина 15", ...],
                    "social_links": ["https://instagram.com/username", "https://t.me/username", "https://vk.com/username", ...],
                    "map_links": ["https://yandex.ru/maps/org/123", ...],
                    "other_contacts": {{"telegram": "username", "whatsapp": "phone_number"}}
                }}
                ```
                """


    full_prompt = prompt + "\n\n" + text
    logger.info(f"Сформированный промпт для Gigachat: {full_prompt[:200]}...")  # логируем начало промпта, раз уж логи везде где только можно

    # Попытки нужны для токенов, чтобы не случилось так, что во время работы программы мы упёрлись в 500 ошибку из-за токена
    # PS ну по крайней мере я так сделал, может можно лучше, не придумал как
    for attempt in range(MAX_RETRIES):
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {get_access_token()}'
            }

            payload = {
                "model": "GigaChat-2",
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 1,
                "max_tokens": 4000
            }

            logger.info(f"Попытка {attempt + 1}/{MAX_RETRIES}")
            response = requests.post(
                GIGACHAT_API_URL,
                headers=headers,
                json=payload,
                verify=False,
                timeout=30
            )
            logger.info(f"Ответ от GigaChat API: {response.status_code} - {response.text}")
            if response.status_code == 200:
                break 
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Если получен статус 401 (не авторизован), пробуем обновить токен
            if response.status_code == 401 and attempt < MAX_RETRIES - 1:
                logger.warning("Токен истек, обновляем...")
                TOKEN_CACHE["token"] = None  # Принудительно обновляем токен
                continue
            else:
                logger.error(f"HTTP Error: {e}")
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к GigaChat API: {e}")
            logger.error(f"Ответ сервера: {response.status_code} {response.text}")
            raise HTTPException(status_code=500, detail=f"Ошибка GigaChat API: {str(e)}")

    try:
        response_json = response.json()
        choices = response_json.get("choices", [])

        # Если ответ не содержит данных, логируем и генерируем ошибку, но если честно, 
        # тут это скорее для красоты, чтобы всё было лаконично было
        if not choices:
            logger.error("Ответ от GigaChat не содержит данных")
            raise HTTPException(status_code=500, detail="GigaChat вернул пустой ответ")

        result = choices[0].get("message", {}).get("content", "")
        if not result:
            logger.error("GigaChat API вернул пустой content")
            raise HTTPException(status_code=500, detail="GigaChat API вернул пустой content")

    except (KeyError, IndexError) as e:
        logger.error("Ошибка парсинга ответа GigaChat")
        raise HTTPException(status_code=500, detail="Ошибка парсинга ответа GigaChat")

    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Хост для запуска сервера")
    parser.add_argument("--port", default=8000, type=int, help="Порт для запуска сервера")
    args = parser.parse_args()

    logger.info(f"Запуск сервера на {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=60)
