# API анализа тональности

Продолжение [первой части](https://github.com/lmaoooo45/ml-engineering-1). API использует ту же модель `cointegrated/rubert-tiny-sentiment-balanced`. `model.py` содержит класс из текстовой задачи. Веса не обучаются и загружаются один раз при запуске сервера.

ТЗ: принимать русский текст через HTTP, определять тональность и возвращать JSON. Входной текст должен быть строкой от 1 до 5000 символов и содержать хотя бы один непробельный символ. Модель обрабатывает до 512 токенов, более длинный текст обрезается. Для некорректного запроса возвращается код 422.

Установка и запуск из папки проекта на Windows с Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

При первом запуске нужен интернет для скачивания модели. Документация с формой отправки запроса: http://127.0.0.1:8000/docs. Проверка готовности: `GET /health`. Основной запрос: `POST /predict`, тело:

```json
{"text": "Мне понравился этот фильм"}
```

Ответ содержит `text`, `label`, `score`, `scores` и `processing_seconds`. Метки: negative, neutral, positive. Оценки sigmoid не обязаны суммироваться в единицу; это не измеренная точность модели.

Все тесты:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

В `tests/test_api.py` проверяются HTTP-запросы, ошибки и однократная загрузка модели. Вместо сети используется простой тестовый объект. `tests/test_model_api.py` проверяет реальную модель через API на трёх фразах. Эти примеры проверяют работоспособность, а не точность на большом наборе. Для запуска только быстрых тестов можно добавить `-m "not integration"`.

Файл `.github/workflows/tests.yml` запускает все тесты при push в GitHub, в pull request и вручную. Локальный commit без push не запускает GitHub Actions. Результаты запусков доступны во вкладке [Actions](https://github.com/lmaoooo45/ml-engineering-api/actions).

Локальная проверка 12 октября 2026 года: 16 тестов прошли на Windows с Python 3.12.2. Проверены запуск Uvicorn и запросы через `/docs`: обычная фраза возвращает 200 и positive, пустой текст возвращает 422.

Источники: [FastAPI](https://fastapi.tiangolo.com/tutorial/), [тестирование](https://fastapi.tiangolo.com/tutorial/testing/), [GitHub Actions](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

Третья часть: [рекомендательная система фильмов](https://github.com/lmaoooo45/ml-engineering-recsys).
