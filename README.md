# ALPACA Test Bench - Испытательный стенд для тестирования библиотек оцифровки документов

## 🎯 Цель проекта

ALPACA Test Bench - это комплексная система для тестирования и сравнения различных библиотек по оцифровке и очистке документов перед их дальнейшей обработкой системами типа Dify. Основная задача - получить идеальный Markdown из различных форматов документов.

## 🏗️ Архитектура системы

### Компоненты системы

```
ALPACA Test Bench/
├── src/                    # Основной код
│   ├── processors/         # Модули обработки документов
│   ├── ui/                # Веб-интерфейсы
│   ├── utils/             # Утилиты и хелперы
│   └── core/              # Ядро системы
├── configs/               # Конфигурации процессоров
├── test_files/           # Тестовые документы
├── outputs/              # Результаты обработки
└── tests/                # Unit тесты
```

### Поддерживаемые форматы

- **PDF файлы** (.pdf) - PyPDF, PDFPlumber, PyMuPDF, PDFMiner, Camelot
- **Word документы** (.doc, .docx) - Unstructured Partition (rus+eng OCR), docx2txt  
- **PowerPoint** (.ppt, .pptx) - python-pptx
- **Excel таблицы** (.xls, .xlsx) - openpyxl, pandas, xlrd
- **Изображения** (.jpg, .jpeg, .png, .tiff) - Tesseract OCR, EasyOCR
- **Структурированное извлечение** – unstructured.partition_* (PDF/DOCX/PPTX/MD/HTML) с классификацией элементов и автоочисткой шумов

### Pipeline обработки

1. **Загрузка документа** - валидация и определение типа
2. **Извлечение контента** - применение специализированных процессоров
3. **Постобработка** - очистка и нормализация текста
4. **Конвертация в Markdown** - финальное форматирование
5. **Оценка качества** - метрики и сравнение результатов

### Структурированное извлечение

Пайплайн поддерживает `UnstructuredPartitionExtractor`, который использует `unstructured.partition_*` для получения элементов (Title, NarrativeText, ListItem, Table и т.д.) с сохранением структуры, фильтрацией водяных знаков и номеров страниц. Все элементы передаются в `UnstructuredLLMCleaner`, что позволяет подключать LLM для контекстной нормализации и выдавать готовый Markdown без потери смысла.

## 🚀 Быстрый старт

### Установка

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd "ALPACA test bench"

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\\Scripts\\activate    # Windows

# Установите зависимости
pip install -r requirements.txt

# Скопируйте настройки окружения
cp .env.example .env
```

## 🧩 Интеграция с Unstructured

Блок `unstructured` добавляет в стенд полноценную работу с разметкой документа до уровня элементов.

- `UnstructuredPartitionExtractor` автоматически выбирает нужную функцию `partition_*` (например, `partition_pdf`, `partition_docx`, `partition_md`) и возвращает элементы с типами (`Title`, `NarrativeText`, `ListItem`, `Table`, `Header`, `Footer` и др.).
- Шум (повторяющиеся заголовки, номера страниц, водяные знаки) исключается на этапе извлечения за счёт фильтров `drop_types`.
- Layout PDF/DOCX/HTML/Markdown сохраняется: таблицы возвращаются с HTML-структурой, списки — с признаками вложенности, заголовки — с уровнем (`category_depth`).
- `UnstructuredLLMCleaner` применяет `unstructured.cleaners.clean_text`/`process_element` и может дополнительно вызвать LLM для контекстной нормализации текста.

### Установка

```bash
pip install "unstructured[all-docs]"

# (опционально) пакет с hi_res моделями
pip install "unstructured-inference[torch]"
```

### Пример использования

```python
from pathlib import Path
from src.core.pipeline import DocumentPipeline
from src.processors import UnstructuredPartitionExtractor, UnstructuredLLMCleaner

pipeline = DocumentPipeline()

pipeline.register_extractor(
    ['.pdf', '.docx', '.md'],
    UnstructuredPartitionExtractor({
        'supported_types': ['.pdf', '.docx', '.md'],
        'strategy': 'hi_res',
        'chunking_strategy': 'by_title',
        'infer_table_structure': True,
    })
)

def llm_normalizer(text: str, element_type: str, metadata: dict, context: str = "") -> str:
    # Здесь можно вызвать OpenAI, Azure OpenAI, Together и т.д.
    return text.strip()

pipeline.register_cleaner(
    UnstructuredLLMCleaner({
        'use_llm_cleaning': True,
        'llm_callable': llm_normalizer,
        'cleaner_functions': [
            'clean_extra_whitespace',
            'clean_multiple_newlines',
            'clean_non_ascii_chars',
        ],
    })
)

result = pipeline.process_document(Path('test_files/pdf/sample.pdf'))
markdown = result['conversion_results']
```

### Быстрое применение partition_* и cleaners

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_text

elements = partition_pdf('sample.pdf', strategy='hi_res', infer_table_structure=True)

for element in elements:
    element.apply(clean_text)
    # Дополнительно можно вызвать:
    # element = process_element(element, filters=[custom_filter])
```

LLM можно подключить через параметр `llm_callable` в `UnstructuredLLMCleaner` — функция получает `text`, `element_type`, `metadata`, `context` и должна вернуть очищенный фрагмент Markdown.

### Настройка Tesseract (для OCR)

**Windows:**
1. Скачайте и установите Tesseract из [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Добавьте путь к tesseract.exe в переменную PATH
3. Или укажите путь в `.env`: `TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe`

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-rus
```

### Запуск

```bash
# Веб-интерфейс (Streamlit)
streamlit run src/ui/streamlit_app.py

# API сервер (FastAPI)  
uvicorn src.api.main:app --reload --host localhost --port 8000
```

## 📊 Использование системы

### 1. Загрузка тестовых файлов
- Поместите документы в папку `test_files/`
- Рекомендуется организовать по подпапкам по типам файлов

### 2. Настройка процессоров
- Отредактируйте `configs/processors_config.py` для настройки параметров
- Включите/выключите нужные процессоры через флаг `enabled`
- Настройте приоритет обработки через параметр `priority`

### 3. Тестирование через веб-интерфейс
1. Откройте Streamlit приложение (обычно http://localhost:8501)
2. Выберите файл для обработки
3. Настройте параметры процессоров
4. Запустите обработку
5. Сравните результаты разных методов

### 4. Пакетное тестирование
```python
from src.core.pipeline import DocumentPipeline
from src.utils.batch_processor import BatchProcessor

# Создание pipeline
pipeline = DocumentPipeline()

# Обработка всех файлов в папке
batch = BatchProcessor(pipeline)
results = batch.process_directory("test_files/pdf/")
```

## 🔧 Конфигурация процессоров

### Настройка PDF процессоров

```python
PDF_PROCESSORS = {
    "pdfplumber": ProcessorConfig(
        name="PDF Plumber",
        enabled=True,
        priority=2,
        parameters={
            "extract_tables": True,
            "table_settings": {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines"
            }
        }
    )
}
```

### Настройка OCR

```python
IMAGE_OCR_PROCESSORS = {
    "tesseract": ProcessorConfig(
        name="Tesseract OCR", 
        enabled=True,
        parameters={
            "lang": "rus+eng",
            "config": "--oem 3 --psm 6"
        }
    )
}
```

## 📈 Метрики качества

Система автоматически вычисляет следующие метрики:

- **Длина текста** - объем извлеченного контента
- **Читаемость** - индексы Flesch-Kincaid, ARI
- **Сохранение структуры** - наличие заголовков, списков, таблиц
- **Качество форматирования** - корректность Markdown разметки
- **Уровень ошибок** - количество артефактов и некорректных символов

## 🛠️ Расширение системы

### Добавление нового процессора

1. Создайте класс процессора в `src/processors/`:

```python
from src.core.base_processor import BaseProcessor

class NewProcessor(BaseProcessor):
    def process(self, file_path: str) -> str:
        # Ваша логика обработки
        return extracted_text
```

2. Зарегистрируйте в `configs/processors_config.py`
3. Добавьте в соответствующий раздел UI

### Добавление новых метрик

```python
from src.utils.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def calculate(self, text: str, original_file: str) -> float:
        # Ваша метрика
        return score
```

## 🧪 Тестирование

```bash
# Запуск всех тестов
pytest

# Запуск с покрытием
pytest --cov=src --cov-report=html

# Тестирование конкретного модуля
pytest tests/test_processors.py
```

## 📝 Лучшие практики

### Для тестирования
1. **Используйте разнообразные документы** - разные стили, языки, форматы
2. **Тестируйте пограничные случаи** - поврежденные файлы, нестандартные форматы
3. **Документируйте результаты** - ведите журнал лучших конфигураций

### Для настройки процессоров
1. **Начинайте с базовых настроек** - затем тонко настраивайте
2. **Комбинируйте процессоры** - используйте сильные стороны каждого
3. **Мониторьте производительность** - некоторые процессоры медленнее других

### Для обработки больших файлов
1. **Настройте лимиты** - `MAX_FILE_SIZE_MB`, `MAX_PROCESSING_TIME_SECONDS`
2. **Используйте параллельную обработку** - `PARALLEL_WORKERS`
3. **Очищайте временные файлы** - настройте автоочистку

## 🤝 Вклад в развитие

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Создайте Pull Request

## 📄 Лицензия

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Поддержка

- GitHub Issues: [Создать issue](../../issues)
- Email: team@alpaca.dev
- Документация: [Wiki](../../wiki)

---

**Примечание**: Этот проект находится в активной разработке. Некоторые функции могут быть изменены в будущих версиях.