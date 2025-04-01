# NLP_parsing

Этот репозиторий содержит несколько экспериментов и реализаций по извлечению контактных данных и локаций из текстов с использованием различных подходов и моделей.

## 📌 Содержание
- [API Gigachat](#api-gigachat)
- [Папка trash](#папка-trash)
- [Файл main.nlp](#файл-mainnlp)
- [Общие пояснения](#общие-пояснения)

---

## 🚀 API Gigachat

Папка `api_gigachat` содержит минималистичное FastAPI-приложение для обработки текстов через [GigaChat API](https://gigachat.devices.sberbank.ru).  
Код организован таким образом, чтобы работать в режиме однопоточного обращения к API.  

### 🔧 Запуск сервера  
Перейдите в папку `api_gigachat` и запустите сервер с помощью uvicorn:

```bash
cd api_gigachat
uvicorn main:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 60
```

## 📊 Обработка данных
Код делает запросы к GigaChat API для обработки текста. По моим подсчётам, обработка 10,000 строк заняла бы около 10:30 часов, однако уже обработанные строки обрабатывались примерно за 3 часа.(Там не Null 670 строк,
ещё в этом и моя вина, что грубо предобработал и там где оставалось по 1 токену, мне нужно было прям всю строку без предобработки пихать, это сидело у меня в голове но я постоянно забывал)

## 🗑️ Папка trash
В папке trash хранятся ноутбуки и эксперименты, которые не имеют практической пользы для основного проекта, но сохранены ради историчности и дальнейших экспериментов.

## 🔹 Особенности:
Notebook parsing-with-gemma3.ipynb
Этот ноутбук задумывался для запуска на Kaggle, однако из-за сложностей с зависимостями модель так и не была запущена.

Остальные файлы в папке — экспериментальные наработки, которые помогут в будущем при повторном рассмотрении идей.

## ❌ Файл main.nlp
Файл main.nlp представляет эксперимент по извлечению локаций из текстов.

Изначально я предполагал, что модель с лёгкостью выделит полные адреса или локации из текста. Но я забыл что это не LLM а NLP для задачи NER

На деле модель отрабатывала именно на задаче NER, извлекая отдельные сущности (например, названия городов или улиц), а не полный адрес.

Этот эксперимент соответствует датасету processed_contacts.csv, где результатом является не полноценный адрес, а набор отдельных локационных элементов.

## API Gigachat
Папка для этого дела содержит .py скрипт для запуска на 127.0.0.1 сервера и другой скрипт который будет дёргатьт его за одну деинственную ручку  

Чтобы получить токен а затем посмотреть модели есть postman_collection где хранятся 2 запроса POST->GET 

## 🔑 Файл .env
Для работы с GigaChat API требуется создать файл .env, который должен содержать следующие переменные:

GIGACHAT_CLIENT_ID=your_client_id
GIGACHAT_CLIENT_SECRET=your_client_secret
Как это работает?
GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET передаются в API.

API кодирует их и генерирует токен для аутентификации.

Токен используется в последующих запросах для доступа к GigaChat.

Ещё тут должна быть ссылка на csv файл 1.9 гб после NLP модели, чтобы вы увидели, о чём я именно говорю, но ему ещё загружается на диск 3 часа, как загрузиться - прикреплю  
К сожалению Git LFS мне не дался

