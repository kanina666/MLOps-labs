# Практическая работа «Наблюдаемость»

API и пакетный источник публикуют задачи в ограниченную очередь RabbitMQ. FastStream-воркер загружает модель из MLflow по alias `champion` и сохраняет результат в PostgreSQL. Дашборд «Конвейер» показывает состояние системы и позволяет перейти от лога задачи к её трейсу.

## Стек и выбор инструментов

| Сигнал | Как собирается | Почему так |
| --- | --- | --- |
| Логи | `logging` → JSON в stdout → Grafana Alloy → Loki | Общий форматтер обслуживает приложение и библиотеки без дополнительного клиента логов. Alloy читает контейнеры текущего Compose-проекта через `docker.sock:ro`. Loki индексирует только `service`, `level`, `env` и поддерживает переходы к Tempo в Grafana. |
| Трейсы | OpenTelemetry SDK → OTLP → Collector → Tempo | Collector ограничивает память перед пакетированием и централизует экспорт. Tempo поддерживает запросы по спанам и переходы к Loki. На стенде используются локальные именованные тома. |
| Метрики | Push по OTLP через Collector в Prometheus. RabbitMQ опрашивается на `15692` | Короткоживущий источник отправляет финальные метрики перед выходом. RabbitMQ публикует `ready`, `unacked` и глубину DLQ через встроенный плагин. Prometheus вычисляет скорости, перцентили и алерты по этим данным. |

`LOG_LEVEL` управляет всеми тремя сервисами. `task_id` передаётся в AMQP `correlation_id` и добавляется в логи через контекст обработки. `trace_id` и `span_id` берутся из активного спана. Идентификаторы задач, пакетов и трейсов остаются полями JSON. Индексация уникального ID на каждую задачу создала бы отдельный поток почти для каждой записи. Тексты исключений с входными значениями и DSN отбрасываются, для диагностики сохраняются тип и кадры стека. Успешные health-check записываются на `DEBUG`.

## Чистый запуск

Требуются Python 3.12, `uv` и Docker Compose с поддержкой `include` и `configs.content`. Команды для PowerShell выполняются из корня репозитория. `down -v` удаляет данные выбранного Compose-проекта, поэтому для защиты используется отдельное имя проекта.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
$env:COMPOSE_PROJECT_NAME = 'mlops-observability-demo'
$env:MODEL_URI = 'models:/mlops_observability_demo@champion'
uv sync --locked
docker compose build app mlflow
docker compose --profile observability --profile tools down -v --remove-orphans
docker compose --profile observability up -d
uv run --locked python scripts/demo_observability.py init
docker compose --profile observability up -d --wait --wait-timeout 120
```

Команда `init` регистрирует демонстрационную модель после запуска MLflow. Источники данных и дашборд загружаются из [provisioning](observability/grafana/provisioning), правила — из [Compose-конфигурации Prometheus](observability/compose.yaml). Откройте [дашборд «Конвейер»](http://localhost:3000/d/mlops-pipeline) под пользователем `admin` с паролем `GRAFANA_PASSWORD` из `.env`. Правила доступны в **Alerting → Alert rules → Data source-managed** и в [Prometheus Alerts](http://localhost:9090/alerts).

| Команда | Сервисы наблюдаемости |
| --- | --- |
| `docker compose up -d` | Логи в stdout |
| `docker compose --profile logs up -d` | Alloy и Loki |
| `docker compose --profile traces up -d` | Collector и Tempo |
| `docker compose --profile metrics up -d` | Collector и Prometheus |
| `docker compose --profile observability up -d` | Все сигналы и Grafana |

Источник запускается по запросу через `docker compose run source`. Настройки `OTEL_*`, `LOG_LEVEL` и версии образов приведены в [.env.example](.env.example). При недоступном Collector приложение продолжает обработку. Очереди и повторы фонового экспорта ограничены, поэтому часть сигналов может потеряться. `OTEL_SDK_DISABLED=true` отключает SDK, а `OTEL_TRACES_EXPORTER=none` и `OTEL_METRICS_EXPORTER=none` позволяют выключить отдельный экспорт.

## Трейсы и метрики

У API, источника и воркера разные `service.name`. `service.version`, `deployment.environment.name` и `service.instance.id` задаются через `OTEL_RESOURCE_ATTRIBUTES`. По умолчанию ID экземпляра берётся из hostname, поэтому реплики различимы. Заголовок `traceparent` связывает `POST /api/v1/tasks → publish → process → inference / db.write_result`. Для inference контекст явно переносится в executor. Отказы и паузы источника видны в `source.backoff`, повторные доставки — в отдельных спанах `process` того же трейса с номером попытки. Пакет содержит отдельный трейс на каждую задачу.

На стенде `OTEL_TRACES_SAMPLER=parentbased_always_on` записывает все локально начатые трейсы и сохраняет решение родителя. Для сотен задач в секунду подойдёт `parentbased_traceidratio` с долей по замерам нагрузки. Для сохранения всех ошибок Collector с tail sampling должен получить все спаны каждого трейса и дождаться его исхода.

| Компонент | Метрики |
| --- | --- |
| API | Счётчик запросов по маршруту и коду, гистограмма длительности |
| Источник | Счётчики публикаций, nack, повторов и суммарного времени ожидания |
| Воркер | Счётчики `ok`, `error`, `dlq` и повторных доставок, текущие задачи в работе, гистограммы inference и времени до результата |
| Инфраструктура | RSS процессов из `psutil`, ready и unacked рабочей очереди, глубина DLQ |

Бакеты inference масштабируются от `INFERENCE_DELAY_MS` и при 200 мс сгущены около 180–300 мс. Для API бакеты сгущены в интервале 5–200 мс. Графики p50/p95/p99 используют `histogram_quantile` после суммирования скоростей по бакетам. Панель SLO показывает долю терминальных задач, успешно сохранённых в пределах `N`. По умолчанию `N=3 с`, цель — 99%, error budget — 1%. Текущий объём незавершённых задач виден по глубине очереди и in-flight.

## Оценка числа временных рядов

Оценка рассчитана для одного API, одного источника, двух воркеров и двух очередей. Число рядов равно произведению числа значений лейблов, для гистограммы оно дополнительно умножается на число бакетов и рядов суммы и количества.

Для одного API с 12 комбинациями `(method, route, status_code)` получается `12 × (1 счётчик + 9 конечных бакетов + +Inf + sum + count) = 156` рядов. Источник даёт 5 рядов, два воркера — `2 × (3 исхода + redelivery + in-flight + RSS + 12 рядов inference + 14 рядов времени задачи) = 64`. RSS API и две очереди добавляют 9 рядов. Итого для перечисленных метрик получается `156 + 5 + 64 + 9 = 234`, то есть около **240 рядов**. Это оценка выбранных метрик конвейера, а не всех рядов хранилища. Служебные метрики, дополнительные HTTP-комбинации и новые экземпляры увеличивают итоговое число. После пересоздания контейнеров старые ряды сохраняются до истечения retention. `task_id` добавил бы до 6000 рядов к одному счётчику или до `6000 × 12 = 72 000` к гистограмме inference.

## Сценарии защиты

Начните с одного воркера. Для короткого показа пакета установите `INFERENCE_DELAY_MS=35` до запуска стенда. При стандартных 200 мс пакет займёт больше времени.

```powershell
uv run --locked python scripts/demo_observability.py batch --tasks 6000
docker compose run -d --name mlops-demo-source source `
    python -m mlops_labs.source --input /data/batch.jsonl --batch-id demo-6000
```

Для нового прогона задайте новые `--name` и `--batch-id`. `published_total` в `source.progress` показывает число подтверждённых публикаций. Чтобы сверить число завершённых задач по пакету, используйте в Loki Explore диапазон, охватывающий весь прогон.

```logql
sum by (event) (count_over_time({service=~"source|worker"} | json | batch_id="demo-6000" | event=~"source.published|task.completed|task.dead_letter" [15m]))
```

1. **Нагрузка и масштабирование.** В Loki Explore найдите пакет запросом `{service="source"} | json | batch_id="demo-6000" | event=~"source.progress|source.complete"`. На дашборде покажите `ready`, `unacked`, лимит `x-max-length=10` и скорости. Через минуту устойчивой нагрузки выполните `docker compose --profile observability up -d --scale worker=2`. Сравните скорости до и после на устойчивых интервалах через `sum(rate(mlops_source_published_total[1m]))` и `sum by (instance) (rate(mlops_worker_tasks_total{outcome="ok"}[1m]))`.
2. **Одна задача.** Возьмите `task_id` из PostgreSQL или из ответа `POST /api/v1/tasks`. В Loki выполните `{service=~"api|source|worker"} | json | task_id="<id>"`. Раскройте запись и нажмите **Tempo**, затем у спана — **Logs for this span**. `messaging.rabbitmq.queue_wait_seconds` оценивает время до доставки, включая сеть. Длительность inference показывает его спан. Exemplars связывают графики задержки с трейсами. За последние 15 минут ошибки видны по `{service=~"api|source|worker",level=~"ERROR|CRITICAL"}`.
3. **Переполнение и `503`.** После опустошения очереди остановите воркеры командой `docker compose stop worker`. Запустите `uv run --locked python scripts/demo_observability.py api --requests 40 --concurrency 8`. При лимите 10 ожидаются 10 ответов `202` и 30 ответов `503`. Удерживайте очередь полной до `Firing`, затем запустите воркеры. Трейсы отказов находятся в Tempo по `{ span.http.response.status_code = 503 }`.
4. **DLQ.** После дренирования очереди отправьте poison из блока ниже. Первая команда показывает три попытки и `task.dead_letter` в одном трейсе, вторая — немедленный DLQ для невалидного payload.
5. **SIGTERM и отказ экспорта.** Под нагрузкой остановите воркер через `docker compose stop -t 45 worker`. Покажите `worker.shutdown_started`, `task.completed` и `worker.stopped`. По отдельности остановите `otel-collector` и `loki`. После каждой остановки отправьте валидную задачу, проверьте её статус `COMPLETED` через `GET /api/v1/tasks/{task_id}` и восстановите сервис.

Для команды `poison` используются опубликованные порты RabbitMQ и Collector. Блок рассчитан на порты по умолчанию.

```powershell
$oldRabbit = $env:RABBITMQ_URL
$oldOtlp = $env:OTEL_EXPORTER_OTLP_ENDPOINT
try {
    $env:RABBITMQ_URL = 'amqp://guest:guest@127.0.0.1:5672/'
    $env:OTEL_EXPORTER_OTLP_ENDPOINT = 'http://127.0.0.1:4318'
    uv run --locked python scripts/demo_observability.py poison
    uv run --locked python scripts/demo_observability.py poison --malformed
} finally {
    $env:RABBITMQ_URL = $oldRabbit
    $env:OTEL_EXPORTER_OTLP_ENDPOINT = $oldOtlp
}
```

## Алерты и бюджет памяти

| Правило | Порог | Выдержка |
| --- | --- | --- |
| `TaskQueueNearCapacity` | Средняя глубина за 10 с превышает 90% лимита. Короткое окно сглаживает освобождение отдельных слотов. | `1m`, чтобы краткое заполнение буфера не вызывало тревогу |
| `TaskDeadLetters` | В DLQ есть хотя бы одно сообщение. Каждая такая задача требует разбора. | `10s`, чтобы кратковременное появление при ручной проверке не вызывало тревогу |

Loki, Tempo и Prometheus хранят данные 2 дня. Суммарный лимит профиля наблюдаемости — **1440 МиБ**. Collector и Alloy тратят память на обработку и доставку, их лимиты входят в бюджет.

```powershell
$ids = docker compose --profile observability ps -q alloy loki otel-collector tempo prometheus grafana
docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}' $ids
```

| Компонент | `docker stats` под нагрузкой, МиБ | Текущий лимит, МиБ |
| --- | ---: | ---: |
| Alloy | 57,49 | 128 |
| Loki | 73,48 | 256 |
| Collector | 38,55 | 128 |
| Tempo | 179,10 | 416 |
| Prometheus | 38,54 | 256 |
| Grafana | 94,35 | 256 |
| **Всего** | **481,51** | **1440** |

Таблица передаёт измеренный расход памяти из `docker stats` во время предыдущего прогона 1300 задач. Это исторический замер, а не гарантированный максимум и не результат нового запуска. Лимит 1440 МиБ относится только к шести сервисам наблюдаемости; приложение, RabbitMQ, PostgreSQL и MLflow в него не входят. После публикации пакета из 6000 задач стек занимал 666,01 МиБ. В этом прогоне Loki насчитал 6000 публикаций и 6000 завершений, PostgreSQL подтвердил 6000 задач с одной попыткой. При 35 мс inference устойчивая скорость обработки выросла с 16,70 до 33,09 задачи/с после подключения второго воркера.
