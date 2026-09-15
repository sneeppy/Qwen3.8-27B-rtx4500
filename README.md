# Qwen3.8-27B на NVIDIA RTX 4500 Ada Generation

Стек инференса **Qwen3.8-27B** на GPU архитектуры **Ada Lovelace (sm_89)**: RTX 4500 Ada Generation 24 ГБ. Внутри образа — **vLLM 0.28.0** с патчами из этого репозитория.

Запуск: `check-env.sh` → `.env` → `docker compose up -d --build` → Open WebUI.

Канонический запуск — **только Docker** на Linux-хосте с RTX 4500 Ada. Open WebUI даёт чат в браузере, а vLLM — OpenAI-совместимый API. С хоста: чат `http://127.0.0.1:3000`, API `http://127.0.0.1:8080`. Оба порта слушают только loopback; снаружи чат открывает Caddy.

Машиночитаемые требования: [`stack-requirements.txt`](stack-requirements.txt). Что делает стек иначе, чем stock vLLM: [docs/optimizations.md](docs/optimizations.md). Подводные камни при отладке: [docs/gotchas.md](docs/gotchas.md).

```bash
./check-env.sh
cp .env.example .env                # CTX, API_KEY, PORT, CADDY_NETWORK, …
# В .env: WEBUI_SECRET_KEY=$(openssl rand -hex 32)
docker compose up -d --build
```

| Этап | Что происходит |
|---|---|
| `docker compose up -d --build` | Сборка vLLM-образа на сервере и загрузка закреплённого образа Open WebUI. |
| Первый старт контейнера | Скачивание и requant модели (~20 ГБ) в `./models`, затем torch.compile / CUDA graphs. Healthcheck ждёт до 15 минут. |
| Повторный старт | Веса, compile-кэш и база Open WebUI остаются в Docker volumes. |

Проверка GPU в Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

Если команда падает — установите [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) и перезапустите Docker.

---

## Быстрый старт

1. На **хосте с RTX 4500 Ada** (драйвер ≥ 575, Docker, NVIDIA Container Toolkit):

   ```bash
   ./check-env.sh
   ```

2. Конфиг:

   ```bash
   cp .env.example .env
   echo "WEBUI_SECRET_KEY=$(openssl rand -hex 32)" >> .env
   ```

   Обязательно задайте `CADDY_NETWORK`. Рекомендуется также сгенерировать `API_KEY`.

3. Сборка и запуск на этом хосте:

   ```bash
   docker compose up -d --build
   ```

4. Откройте чат: `http://127.0.0.1:3000`. Первый зарегистрированный пользователь становится администратором.

5. Проверка API:

   ```bash
   curl http://127.0.0.1:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "Ответь одним словом: привет"}],
       "temperature": 0.7,
       "chat_template_kwargs": {"enable_thinking": false}
     }'
   ```

Дефолты контейнера: `MODE=single`, `SPEC=dflash2`, `PREFIX_CACHE=1`, контекст **32768**, API-порт **8080**, UI-порт **3000**. Поднимайте `CTX` в `.env` только после успешного прогона на 32K.

Веса **не кладутся в образ** — только volume. Хосту не нужны nvcc, GCC и нативный vLLM.

Если образ уже есть локально и Caddy не нужен в этом запуске — `CADDY_NETWORK` в compose всё равно обязателен. Задайте сеть, даже если сайт ещё не описан в Caddyfile.

---

## Модель

Нужен checkpoint **W4A16 AutoRound** в формате `compressed-tensors`, не GGUF:

`dbirks/Qwen3.8-27B-W4A16-AutoRound` → после `prepare` каталог
`models/Qwen3.8-27B-W4A16-AutoRound` (и `-fast`, если не отключён `FAST_VARIANT=0`).

Для `SPEC=dflash2` контейнер также качает drafter `Qwen3.8-27B-DFlash2-W4A16`.

Контейнер готовит веса сам, если их нет в `MODEL_DIR` (по умолчанию `./models`). Можно положить уже подготовленный каталог туда заранее.

Только скачать и requant, без сервера:

```bash
docker compose run --rm qwen prepare
```

---

## Требования

Машиночитаемый манифест: [`stack-requirements.txt`](stack-requirements.txt).

### Хост (для запуска)

| Компонент | Требование | Заметки |
|---|---|---|
| GPU | NVIDIA **sm_89**, **24 ГБ** | RTX 4500 Ada Generation |
| Драйвер | **≥ 575** (лучше 580+) | Образ CUDA **13** |
| Docker | Engine + Compose | Образ: CUDA **13.0.3** |
| NVIDIA Container Toolkit | обязателен | Иначе `--gpus` не работает |
| CUDA toolkit / GCC на хосте | **не нужны** | nvcc в образе для JIT FlashInfer/Triton |

TGP карты ~210 W; частоты GPU трогать не нужно.

Шина памяти ~**432 ГБ/с**. После прогрева второго бута измерьте tok/s сами клиентом.

---

## Конфигурация (`.env`)

Шаблон: `.env.example`. Файл `.env` рядом с `docker-compose.yml`. Секреты в git не коммитить.

| Переменная | Смысл | По умолчанию |
|---|---|---|
| `PORT` | Порт на хосте и в контейнере | `8080` |
| `CTX` | Окно контекста | `32768` (число токенов). Пресеты: `fast` / `long` / `huge` |
| `MODE` | `single` — чат/агент; `batch` — много запросов | `single` |
| `SPEC` | Drafter: `dflash2` / `mtp` / `off` | `dflash2` |
| `PREFIX_CACHE` | Кэш общего префикса между запросами | `1` |
| `MODEL_DIR` | Каталог весов на хосте → `/app/models` | `./models` |
| `API_KEY` | Если задан — ключ vLLM | пусто (API открыт на loopback) |
| `CADDY_NETWORK` | Имя Docker-сети Caddy | **обязательно** |
| `WEBUI_SECRET_KEY` | Подпись сессий Open WebUI | **обязательно**, случайные 32 байта |
| `WEBUI_PORT` | Loopback-порт чата | `3000` |
| `WEBUI_NAME` | Название интерфейса | `Qwen3.8-27B` |
| `WEBUI_ENABLE_SIGNUP` | Разрешить регистрацию | `True` для первого входа |
| `HF_TOKEN` | Если Hugging Face режет анонимные скачивания | пусто |
| `EXTRA_ARGS` | Доп. флаги `vllm serve` | пусто |

`HOST` внутри контейнера всегда `0.0.0.0` (иначе проброс порта не работает). С хоста слушает только `127.0.0.1`.

Числовой `CTX` мапится в entrypoint: ≤65536 → профиль `fast`, ≤131072 → `long`, иначе `huge`, плюс `MAX_LEN` равный числу. Именованный `CTX=fast` без числа даёт окно launcher’а (64k).

---

## Два режима

Пик скорости зависит от того, один это пользователь или поток API.

```
                    Serving Qwen3.8-27B
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
   MODE=single (по умолчанию)              MODE=batch
   speculative decode (SPEC=dflash2)       без спекуляции, широкий батч
   1–несколько чатов                       много независимых запросов
   PREFIX_CACHE=1                          INT8 GEMM, до 64 seqs
```

**`MODE=single`.** DFlash2 предлагает блок токенов, target проверяет. Для одного оператора за картой.

**`MODE=batch`.** Без спекуляции, высокая совокупная пропускная способность. На одной GPU не поднимать оба режима сразу: в `.env` смените `MODE` и `docker compose up -d --force-recreate`.

Имя модели в API всегда `qwen3.8-27b`.

---

## Клиентский API

Сервер — OpenAI-совместимый REST (`/v1/chat/completions`, `/health`, …).
Open WebUI обращается к нему внутри Docker-сети по `http://qwen:8080/v1`; ключ остаётся на backend UI и не передаётся браузеру.

### cURL

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Ты полезный ассистент по коду."},
      {"role": "user", "content": "Напиши быстрый CUDA reduction kernel на C++."}
    ],
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Если задан `API_KEY`, добавьте заголовок `Authorization: Bearer <ключ>`.

### Python (SDK OpenAI)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[
        {"role": "system", "content": "Ты полезный ассистент по коду."},
        {"role": "user", "content": "Объясни grouped-query attention в двух предложениях."}
    ],
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": false}},
)
print(response.choices[0].message.content)
```

Пока контейнер не запущен на Ada-хосте, слушать нечего.

---

## Open WebUI и Caddy

Open WebUI хранит пользователей, настройки и историю в volume `open-webui-data`. Порт 3000 слушает только `127.0.0.1`; Caddy достигает UI по общей Docker-сети. API vLLM остаётся отдельно на loopback 8080.

1. Узнать сеть Caddy:

   ```bash
   docker inspect <caddy-container> --format '{{range $k, $_ := .NetworkSettings.Networks}}{{println $k}}{{end}}'
   ```

2. В `.env`: `CADDY_NETWORK=<это-имя>`.

3. В Caddyfile — **отдельный сайт**. DNS A-запись на IP сервера, порты 80/443 как у остальных:

   ```caddy
   qwen.4500.dev.econdata.ru {
           encode gzip

           reverse_proxy qwen38-open-webui:8080 {
                   flush_interval -1
           }
   }
   ```

4. `docker compose up -d` (пересоздаст контейнер qwen в сети Caddy), затем перезагрузить Caddy.

Чат: `https://qwen.4500.dev.econdata.ru`. Первый зарегистрированный пользователь получает роль администратора; после этого рекомендуется поставить `WEBUI_ENABLE_SIGNUP=False` и пересоздать UI:

```bash
docker compose up -d --force-recreate open-webui
```

Сырой API не публикуется через этот домен и остаётся на `http://127.0.0.1:8080/v1`.

---

## Операционные ловушки

1. **`CADDY_NETWORK` обязателен** — compose без него не стартует.
2. **Контекст 32K по умолчанию** — безопасный первый старт. `CTX=fast` / `long` / `huge` — после того, как 32K уже живёт. `huge` — lossy KV (KVarN).
3. **Не поднимать `MAX_SEQS` и `KV_MEM` наугад.** Лимит — пул recurrent state и 24 ГБ.
4. **Первый boot медленный.** Цифры скорости снимайте со второго/третьего старта (прогретый `qwen-cache`).
5. **WSL2:** `VLLM_WSL2_ENABLE_PIN_MEMORY=1` в `.env`, иначе V2 runner падает на UVA. На native Linux переменная безвредна.
6. **Ключ.** Loopback без ключа допустим. Как только сайт в Caddy торчит наружу — задайте `API_KEY`.
7. **Регистрация.** После создания администратора выключите `WEBUI_ENABLE_SIGNUP`, если публичная регистрация не нужна.
8. **Конфигурация Open WebUI хранится в volume.** Если позже изменить endpoint через `.env`, сохранённая настройка может иметь приоритет; проверьте Admin Panel → Connections.

Проверка установки внутри контейнера: `docker compose run --rm qwen verify`.

---

## Лицензия и происхождение

- **vLLM** 0.28.0 и патчи в `patches/`.
- Оптимизации: [docs/optimizations.md](docs/optimizations.md). Подводные камни: [docs/gotchas.md](docs/gotchas.md).
