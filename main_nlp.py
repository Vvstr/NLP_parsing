import pandas as pd
import re
import json
import logging
import multiprocessing
import time
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import psutil
import unicodedata

# Настройка логирования: вывод логов в файл extraction.log с указанием времени, уровня и сообщения
logging.basicConfig(
    filename='extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Начало выполнения программы.")

class ContactExtractorNLP:
    """
    Класс для извлечения контактной информации из HTML-текста с использованием методов NLP и регулярных выражений.
    Обратите внимание: NER-модель используется для извлечения сущностей типа LOC/ADDRESS,
    что зачастую приводит к извлечению только отдельных слов (например, названий городов или улиц),
    а не полноценных адресов.
    """
    def __init__(self):
        self._init_model()
        
    def _init_model(self):
        """
        Инициализация NER-модели и компиляция регулярных выражений для поиска контактов.
        
        Если модель еще не загружена, происходит:
         - Загрузка токенизатора и модели для NER из библиотеки transformers.
         - Создание пайплайна для NER с агрегацией сущностей.
         - Компиляция регулярных выражений для телефонов, email, ссылок на соцсети, карты, а также для Skype и ICQ.
        """
        if not hasattr(self, 'tokenizer'):
            logging.info("Загрузка токенизатора...")
            self.tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
            
            logging.info("Загрузка NER-модели...")
            self.ner_pipeline = pipeline(
                "ner",
                model="Davlan/bert-base-multilingual-cased-ner-hrl",
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",  # Объединение смежных сущностей
                device=-1,  # Использование CPU
                batch_size=4
            )
            
        # Компиляция регулярных выражений для поиска различных типов контактов:
        self.phone_pattern = re.compile(r'(?:\+7|8|7)?\s?[\(\-]?\d{3}[\)\-]?\s?\d{3}[\-]?\d{2}[\-]?\d{2}')
        self.email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w{2,6}\b')
        self.social_pattern = re.compile(r'https?://(?:t\.me|vk\.com|facebook\.com)/\S+')
        self.map_pattern = re.compile(r'https?://(?:yandex\.ru/maps|google\.com/maps)/\S+')
        self.skype_pattern = re.compile(r'skype:\s*\S+')
        self.icq_pattern = re.compile(r'ICQ:\s*\d+')
        logging.info("Модель и регулярки инициализированы.")

    def preprocess_html(self, html):
        """
        Предобработка HTML-текста:
         - Заменяет заэкранированные символы (&nbsp;, &amp;) на пробел и символ '&'.
         - Использует BeautifulSoup для парсинга HTML.
         - Удаляет скрытые элементы (с display:none или visibility:hidden).
         - Извлекает текст из основных блоков (main, article, body, div, footer).
         
        Аргументы:
            html (str): Исходный HTML-текст.
        
        Возвращает:
            str: Очищенный текст для дальнейшей обработки.
        """
        try:
            # Замена заэкранированных символов
            html = html.replace('&nbsp;', ' ').replace('&amp;', '&')
            # Используем парсер для быстрого анализа HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Удаляем скрытые элементы, где стиль содержит display:none или visibility:hidden
            for tag in soup.find_all(style=re.compile(r'display:\s*none|visibility:\s*hidden')):
                tag.decompose()
                
            # Извлечение текстовых блоков из ключевых тегов
            content_blocks = soup.find_all(['main', 'article', 'body', 'div', 'footer'])
            text_parts = []
            for block in content_blocks:
                # Проходим по всем текстовым фрагментам в блоке
                for elem in block.stripped_strings:
                    # Замена специальных символов и переносов строк
                    text = elem.replace('\xa0', ' ').replace('\n', ' ')
                    text_parts.append(text)
            
            # Объединяем фрагменты в единый текст
            return ' '.join(text_parts)
        except Exception as e:
            logging.error(f"HTML preprocessing error: {str(e)}")
            return ""

    def _ner_processing(self, text):
        """
        Обработка текста с использованием NER-модели.
        
        Выполняет прямой прогон текста через NER-пайплайн и собирает сущности с метками 'LOC' или 'ADDRESS'.
        При этом найденные сущности объединяются в последовательности, что может приводить к тому, что
        результатом будут отдельные слова (например, только название города или улицы), а не полный адрес.
        
        Аргументы:
            text (str): Текст для извлечения локаций.
        
        Возвращает:
            list: Список уникальных "адресов" (на самом деле наборов сущностей LOC/ADDRESS).
        """
        entities = self.ner_pipeline(text)
        addresses = []
        current_address = []
        # Проходим по найденным сущностям
        for entity in entities:
            # Если сущность относится к локациям или адресам, добавляем слово в текущий набор
            if entity['entity_group'] in ['LOC', 'ADDRESS']:
                current_address.append(entity['word'])
            # Если встречается другая сущность, а предыдущая группа накопила данные,
            # объединяем их в строку и добавляем в список адресов
            elif current_address:
                addresses.append(' '.join(current_address))
                current_address = []
        if current_address:
            addresses.append(' '.join(current_address))
        # Возвращаем уникальные результаты
        return list(set(addresses))

    def extract_contacts(self, text):
        """
        Извлечение различных контактов из текста с использованием регулярных выражений и NER-модели.
        
        Выполняет следующие шаги:
         1. Нормализация текста с помощью unicodedata.
         2. Поиск номеров телефонов: извлекаются цифры и нормализуются в формат +7XXXXXXXXXX.
         3. Поиск email-адресов с последующей фильтрацией (исключаются ссылки на изображения).
         4. Поиск ссылок на соцсети и карты.
         5. Извлечение "адресов" с помощью метода _ner_processing (результат может быть не полноценным адресом).
         6. Поиск контактов Skype и ICQ.
        
        Аргументы:
            text (str): Текст для извлечения контактов.
        
        Возвращает:
            defaultdict: Словарь, содержащий списки найденных контактов по ключам:
                        phones, emails, social_links, map_links, addresses, other_contacts.
        """
        result = defaultdict(list)
        try:
            # Нормализация текста для унификации символов
            text = unicodedata.normalize('NFKC', text)
            
            # Извлечение номеров телефонов
            phones = []
            for match in self.phone_pattern.finditer(text):
                # Убираем все нецифровые символы
                phone = re.sub(r'\D', '', match.group())
                # Если номер состоит из 11 цифр и начинается с 7, 8 или 7, нормализуем его
                if len(phone) == 11 and phone.startswith(('7', '8')):
                    phones.append(f"+7{phone[1:]}")
                elif len(phone) == 10:
                    phones.append(f"+7{phone}")
            result["phones"] = phones
            
            # Извлечение email-адресов
            emails = []
            for email in self.email_pattern.findall(text):
                # Фильтрация: исключаем email, оканчивающиеся на расширения изображений
                if not email.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    emails.append(email)
            result["emails"] = emails
            
            # Извлечение ссылок на соцсети и карты
            result["social_links"] = self.social_pattern.findall(text)
            result["map_links"] = self.map_pattern.findall(text)
            
            # Извлечение "адресов" с помощью NER (результат может содержать только отдельные названия городов или улиц)
            result["addresses"] = self._ner_processing(text)
            
            # Извлечение прочих контактов: Skype и ICQ
            result["other_contacts"] = {
                "skype": self.skype_pattern.findall(text)[:3],
                "icq": self.icq_pattern.findall(text)[:3]
            }
        except Exception as e:
            logging.error(f"Extraction error: {str(e)}")
        return result

# Глобальная переменная для экземпляра модели, используемая в worker-процессах
global_extractor = None

def init_worker():
    """
    Инициализатор для воркер-процессов.
    Создает глобальный экземпляр ContactExtractorNLP, чтобы не перегружать модель при каждом вызове.
    """
    global global_extractor
    global_extractor = ContactExtractorNLP()

def process_row(row):
    """
    Обработка одной строки CSV-файла.
    
    Извлекает HTML-контент из строки, выполняет его предобработку, затем извлекает контакты.
    Результаты возвращаются в виде словаря с ключами, соответствующими типам извлеченных данных.
    
    Аргументы:
        row (tuple): Кортеж, содержащий:
                     - row[0]: оригинальный индекс,
                     - row[1]: URL (не используется),
                     - row[2]: HTML-текст.
    
    Возвращает:
        dict: Словарь с результатами обработки строки.
    """
    try:
        start_time = time.time()
        # Извлечение значений: индекс и HTML-контент
        index = row[0]
        html = row[2]
        if index % 50 == 0:
            print(f"Обработка строки {index}...", flush=True)
            logging.info(f"Обработка строки {index}...")
        # Предобработка HTML
        text = global_extractor.preprocess_html(html)
        # Извлечение контактов из очищенного текста
        contacts = global_extractor.extract_contacts(text)
        result = {
            'index': index,
            'extracted_phones': ";".join(contacts["phones"]),
            'extracted_emails': ";".join(contacts["emails"]),
            'extracted_addresses': ";".join(contacts["addresses"]),
            'extracted_social_links': ";".join(contacts["social_links"]),
            'extracted_map_links': ";".join(contacts["map_links"]),
            'extracted_other_contacts': json.dumps(contacts["other_contacts"])
        }
        return result
    except Exception as e:
        logging.error(f"Строка {row[0]}: {str(e)}")
        return {
            'index': row[0],
            'extracted_phones': None,
            'extracted_emails': None,
            'extracted_addresses': None,
            'extracted_social_links': None,
            'extracted_map_links': None,
            'extracted_other_contacts': None
        }
    finally:
        if row[0] % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Строка {row[0]} обработана за {elapsed:.2f} сек", flush=True)

def process_csv_in_chunks(input_csv, output_csv, chunksize=100, num_workers=6):
    """
    Обрабатывает CSV-файл чанками с использованием multiprocessing.
    
    Порядок действий:
      1. Чтение CSV-файла чанками (пакетами строк).
      2. Для первого чанка сохраняется первая строка как заголовок.
      3. Каждый чанк обрабатывается параллельно с помощью пула воркеров.
      4. Результаты каждого чанка объединяются и сохраняются в итоговый CSV-файл.
    
    Аргументы:
        input_csv (str): Путь к входному CSV-файлу.
        output_csv (str): Путь для сохранения результатов.
        chunksize (int): Количество строк в одном чанке.
        num_workers (int): Количество параллельных воркеров.
    """
    print("Начало обработки")
    print(f"Настройки: chunksize={chunksize}, workers={num_workers}")
    print(f"CPU: {psutil.cpu_count()} ядер, Память: {psutil.virtual_memory().total//1024**3} GB")
    start_total = time.time()
    ctx = multiprocessing.get_context('spawn')
    
    header_row = None
    chunk_idx = 0
    with ctx.Pool(processes=num_workers, initializer=init_worker, maxtasksperchild=50) as pool:
        for chunk in pd.read_csv(
            input_csv,
            header=None,         # Чтение всех строк как данных (без автоматического заголовка)
            chunksize=chunksize,
            engine='c',
            dtype='string',
            memory_map=True
        ):
            chunk_idx += 1
            print(f"\nОбработка чанка № {chunk_idx} (строк: {len(chunk)})...")
            logging.info(f"Начало обработки чанка № {chunk_idx}, строк: {len(chunk)}")
            chunk_start = time.time()
            
            # Для первого чанка сохраняем первую строку как заголовок
            if chunk_idx == 1:
                header_row = chunk.iloc[0:1].copy()
                data_chunk = chunk.iloc[1:].copy()
            else:
                data_chunk = chunk.copy()
            
            # Сброс индексов для корректного объединения результатов
            data_chunk = data_chunk.reset_index(drop=False).rename(columns={'index': 'orig_index'})
            rows = list(data_chunk.itertuples(index=False, name=None))
            
            results = []
            with tqdm(total=len(data_chunk), desc=f"Чанк {chunk_idx}") as pbar:
                for res in pool.imap_unordered(process_row, rows, chunksize=10):
                    results.append(res)
                    pbar.update(1)
                    pbar.set_postfix_str(f"Speed: {pbar.n/(time.time()-chunk_start):.1f} rows/sec")
            
            results_df = pd.DataFrame(results)
            merged_chunk = data_chunk.merge(results_df, left_on='orig_index', right_on='index').drop(columns=['orig_index', 'index'])
            
            # Если это первый чанк, объединяем его с заголовком
            if chunk_idx == 1:
                header_row.columns = range(header_row.shape[1])
                merged_chunk = pd.concat([header_row, merged_chunk], ignore_index=True)
            
            # Запись чанка в итоговый файл (перезапись для первого чанка, добавление для последующих)
            mode = 'w' if chunk_idx == 1 else 'a'
            header = (chunk_idx == 1)
            merged_chunk.to_csv(output_csv, mode=mode, header=header, index=False)
            
            chunk_time = time.time() - chunk_start
            print(f"Чанк № {chunk_idx} обработан за {chunk_time:.1f} сек")
            logging.info(f"Чанк № {chunk_idx} обработан за {chunk_time:.1f} сек")
            
    total_time = time.time() - start_total
    print(f"\n{'='*40}")
    print(f"Обработка завершена за {total_time//3600:.0f} ч {total_time%3600//60:.0f} мин")
    avg_speed = (chunksize * (chunk_idx - 1) + (chunksize if header_row is None else chunksize - 1)) / total_time
    print(f"Средняя скорость: {avg_speed:.1f} строк/сек")
    logging.info(f"Общее время выполнения: {total_time:.2f} сек")

if __name__ == '__main__':
    # Рекомендуемые параметры для системы (например, Ryzen 7 4800HS)
    process_csv_in_chunks(
        input_csv="analyzer_analyzer_urls.csv",
        output_csv="processed_contacts.csv",
        chunksize=100,
        num_workers=6
    )
