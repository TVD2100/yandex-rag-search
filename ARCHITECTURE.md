# Архитектура (GitHub-версия)

Самодостаточная версия приложения «Yandex Docs Search» для публикации.
Содержит только то, что нужно для запуска: Streamlit UI, модули поиска и
генерации, пример базы документации.

## Обзор

Приложение - это RAG-агент для поиска документации. Пользователь задаёт
вопрос, агент итеративно ищет фрагменты документации в SQLite-базе через
векторный поиск по эмбеддингам, накапливает полезные фрагменты в контексте
и формирует ответ с цитатами `[#id]`.

## Слои

### 1. UI-слой (`app.py`)

Streamlit-приложение, единственная точка входа. Отвечает за:

- **Автоопределение баз** - `discover_bases()` сканирует `data/<имя>/index.db`
  и возвращает `{name: {path, label}}`. Добавление новой базы = создание
  папки `data/<имя>/` с `index.db` (без правки кода).
- **Креденшелы** - `get_credentials()` читает `SAGAAI_YANDEXAI_KEY`
  (API-ключ) и `SAGAAI_YANDEXAI_KEY2` (Folder ID) из переменных окружения.
- **Сборка пайплайна** - `get_retriever()` создаёт `Retriever` с
  `use_db_vectors=True` (векторы документов из SQLite), `get_searcher()`
  создаёт `IterativeSearch` с параметрами из настроек.
- **Страницы** - «Новый запрос» (селектор базы, вопрос, живой прогресс
  через `st.status`, экспорт ответа) и «Настройки» (форма параметров,
  инструкция по ключам).
- **Светлая тема** - CSS-инъекция через `apply_light_theme()`.

### 2. Поисковый слой

#### `src/retriever.py` - векторный ретривер

- Загружает чанки и векторы документов из SQLite (`load_chunks`,
  `load_vectors`).
- Нормализует матрицу документов (L2) при инициализации
  (`use_db_vectors=True`).
- `search_vector(query, top_n)` - эмбеддинг запроса через
  `text-search-query`, косинусная близость с матрицей документов,
  возвращает `[(chunk_id, score)]`.
- `build_index()` - программное построение векторов документов через
  `text-search-doc` (для баз без таблицы `embeddings`).

#### `src/db.py` - доступ к SQLite

- `load_chunks()` - чтение таблицы `chunks` (id, text, source, chunk_index),
  очистка метаданных-префиксов (`Документ:`, `Продукт:`, `Заголовок:` и т.п.).
- `load_vectors()` - чтение таблицы `embeddings` (chunk_id, vector BLOB
  float32[256]), возвращает `(ids, matrix)`.

#### `src/embeddings.py` - клиент эмбеддингов

- `embed_doc()` - эмбеддинг документа (`text-search-doc`), кеш в npz.
- `embed_query()` - эмбеддинг запроса (`text-search-query`), JSON-кеш по MD5
  текста запроса.
- Троттлинг (RPS), ретраи при 429.

### 3. Генерация ответа

#### `src/iterative_search.py` - итеративный RAG-цикл

Основной компонент. Реализует паттерн function calling:

1. Вопрос отправляется LLM вместе с описанием инструмента `search_rag`.
2. LLM возвращает вызов функции с `query` и `keep_chunk_ids`.
3. Система выполняет векторный поиск (`Retriever.search_vector`),
   возвращает до `top_n` фрагментов с индексами `#id`.
4. LLM либо отвечает с цитатами `[#id]`, либо вызывает `search_rag` снова,
   указывая, какие фрагменты сохранить в контексте.
5. Цикл повторяется до `max_iterations` (по умолчанию 7).
6. Если LLM не дала ответ - fallback через `_answer_from_chunks()`
   (финальный ответ по накопленным фрагментам без инструментов).

Ключевые параметры: `max_iterations`, `top_n`, `max_chunks`,
`temperature`, `reasoning_effort`. API: `chat/completions`, заголовки
`Api-Key` / `x-project`, модель `gpt://{folder_id}/<model>`.

#### `src/answerer.py` - генерация ответа с цитатами

Используется как fallback-путь: формирует ответ по фиксированному набору
фрагментов без итеративного поиска. Парсит цитаты `#<id>` из текста,
возвращает список процитированных чанков.

### 4. Настройки (`src/settings.py`)

- `MODELS` - список моделей из справочника Yandex AI Studio.
- `REASONING_OPTIONS` - допустимые уровни рассуждений для каждой модели.
- `DEFAULT_SETTINGS` - значения по умолчанию: база `rag_base`, модель
  `deepseek-v4-flash`, reasoning `high`, итерации 7, top_n 10,
  max_chunks 10, температура 0.2, show_debug False.
- `load_settings()` / `save_settings()` - JSON в `~/.yaagent_search_settings.json`
  слиянием с дефолтами.

## Поток выполнения запроса

1. Пользователь выбирает базу и вводит вопрос.
2. `app.py` проверяет креденшелы, создаёт `Retriever` и `IterativeSearch`.
3. `IterativeSearch.search()` запускает цикл function calling.
4. На каждой итерации: LLM -> `search_rag` -> векторный поиск -> фрагменты
   -> LLM (ответ или новый вызов).
5. Прогресс отображается через `progress_callback` в `st.status`.
6. Ответ с цитатами показывается пользователю; доступны экспорт в Markdown
   и копирование в буфер.
7. При `show_debug=True` показываются итерации, цитированные фрагменты и
   найденные чанки.

## Данные

- `data/<имя_базы>/index.db` - SQLite-база: таблица `chunks` (id, text,
  source, chunk_index) и таблица `embeddings` (chunk_id, vector BLOB
  float32[256]).
- `rag/query_cache.json` - кеш эмбеддингов запросов (MD5 -> вектор).
- `~/.yaagent_search_settings.json` - настройки пользователя.

## Зависимости

- `streamlit` - UI
- `requests` - HTTP-клиент для API Yandex AI Studio
- `numpy` - векторные операции

## Отличия от полной версии

- Нет скриптов оценки (`scripts/`), тестов, `src/completeness.py`,
  `src/metrics.py`, `src/query_rewriter.py`.
- Нет второй базы `yandex_cloud_docs` (только пример `rag_base`).
- Нет внутренних путей и артефактов разработки.
