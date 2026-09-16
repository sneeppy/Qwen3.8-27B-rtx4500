# Qwen3.8-27B на NVIDIA RTX 4500 Ada Generation

Стек инференса **Qwen3.8-27B** на GPU архитектуры **Ada Lovelace (sm_89)**: RTX 4500 Ada Generation 24 ГБ. Внутри образа — **vLLM 0.28.0** с патчами из этого репозитория.

Запуск: `check-env.sh` → `.env` → `docker compose up -d --build` → Open WebUI.

Канонический запуск — **только Docker** на Linux-хосте с RTX 4500 Ada. Open WebUI даёт чат в браузере, vLLM — OpenAI-совместимый API. Чат: `http://<IP-сервера>:3000` (по умолчанию на всех интерфейсах). API: `http://127.0.0.1:8080` (только loopback). Caddy не нужен для проверки в LAN.

Машиночитаемые требования: [`stack-requirements.txt`](stack-requirements.txt). Что делает стек иначе, чем stock vLLM: [docs/optimizations.md](docs/optimizations.md). Подводные камни при отладке: [docs/gotchas.md](docs/gotchas.md).

```bash
./check-env.sh
cp .env.example .env                # CTX, API_KEY, WEBUI_SECRET_KEY, …
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

   Задайте `WEBUI_SECRET_KEY`. Рекомендуется также `API_KEY`. Caddy не нужен.

3. Сборка и запуск на этом хосте:

```bash
   docker compose up -d --build
```

4. Откройте чат: `http://192.168.1.35:3000` (IP сервера) или `http://127.0.0.1:3000` с самого хоста. Первый зарегистрированный пользователь становится администратором.

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

Шина памяти ~**432 ГБ/с**. На этой карте в `MODE=single` + DFlash2 один чат даёт примерно **80 tok/s** decode (см. [Скорость](#скорость-toks)).

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
| `API_KEY` | Ключ vLLM и Open WebUI | пусто (API открыт на loopback) |
| `WEBUI_SECRET_KEY` | Подпись сессий Open WebUI | **обязательно**, случайные 32 байта |
| `WEBUI_PORT` | Порт чата на хосте | `3000` |
| `WEBUI_BIND` | Адрес проброса UI | `0.0.0.0` (LAN). Loopback: `127.0.0.1` |
| `WEBUI_NAME` | Название интерфейса | `Qwen3.8-27B` |
| `WEBUI_ENABLE_SIGNUP` | Разрешить регистрацию | `True` для первого входа |
| `HF_TOKEN` | Если Hugging Face режет анонимные скачивания | пусто |
| `REQ_METRICS` | Тайминги и `usage` в каждом JSON ответа vLLM | `0`. `1` — для замера с сервера, UI не меняет |
| `EXTRA_ARGS` | Доп. флаги `vllm serve` | пусто |

`HOST` внутри контейнера vLLM всегда `0.0.0.0`. С хоста API проброшен только на `127.0.0.1:8080`. Чат слушает `WEBUI_BIND` (по умолчанию все интерфейсы, порт 3000).

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

Open WebUI хранит пользователей и историю в volume `open-webui-data`. По умолчанию чат доступен с LAN: `http://<IP>:3000`. API vLLM — только `http://127.0.0.1:8080`.

Чтобы слушать только loopback: `WEBUI_BIND=127.0.0.1` в `.env` и `docker compose up -d`.

Caddy не обязателен. Когда появится, подключите UI к его сети и проксируйте контейнер:

```bash
docker inspect <caddy-container> --format '{{range $k, $_ := .NetworkSettings.Networks}}{{println $k}}{{end}}'
docker network connect <имя-сети> qwen38-open-webui
```

```caddy
qwen.4500.dev.econdata.ru {
        encode gzip
        reverse_proxy qwen38-open-webui:8080 {
                flush_interval -1
  }
}
```

После первого входа поставьте `WEBUI_ENABLE_SIGNUP=False` и пересоздайте UI:

```bash
docker compose up -d --force-recreate open-webui
```

---

## Скорость (tok/s)

Цифры ниже — **RTX 4500 Ada**, прогретый `MODE=single`, `SPEC=dflash2`, контекст 32K, `--max-concurrency 1`, thinking выключен, выход 256 токенов. Снимайте со второго бута (`qwen-cache`).

| Нагрузка | Output tok/s | Acceptance DFlash2 | Mean TTFT |
|---|---|---|---|
| random, 128 in / 256 out, 8 запросов | **69** | 30% (длина 3.12) | ~110 мс |
| 8 осмысленных чат-запросов (стих, код, SQL, …) | **80** | 38% (длина 3.64) | ~125 мс |

ITL шага GPU ~**43 мс** (~23 раунда/с). Throughput выше за счёт спекуляции: без DFlash2 было бы около этих 23 tok/s. На живом тексте acceptance выше, чем на random. С thinking и на длинном контексте цифры ниже.

`--dataset-name random` с `--random-input-len 128 --random-output-len 256` даёт нижнюю оценку (acceptance хуже). Смотрите **Output token throughput** и блок **Speculative Decoding**. Peak output tok/s у бенча считается по ITL и при спекуляции часто *ниже* среднего — это не регресс.

`REQ_METRICS=1` в `.env`, затем `docker compose up -d --force-recreate qwen`. В ответе API появятся тайминги; чат не изменится.

Лог движка: `docker compose logs -f qwen` — периодические `Avg generation throughput`.

---

## Операционные ловушки

1. **Контекст 32K по умолчанию** — безопасный первый старт. `CTX=fast` / `long` / `huge` — после того, как 32K уже живёт. `huge` — lossy KV (KVarN).
2. **Не поднимать `MAX_SEQS` и `KV_MEM` наугад.** Лимит — пул recurrent state и 24 ГБ.
3. **Первый boot медленный.** Цифры скорости снимайте со второго/третьего старта (прогретый `qwen-cache`).
4. **WSL2:** `VLLM_WSL2_ENABLE_PIN_MEMORY=1` в `.env`, иначе V2 runner падает на UVA. На native Linux переменная безвредна.
5. **Чат на LAN.** Порт 3000 открыт на `0.0.0.0`. Задайте `API_KEY`, после первого админа выключите `WEBUI_ENABLE_SIGNUP`. API на 8080 с LAN не торчит.
6. **Конфигурация Open WebUI хранится в volume.** Если позже изменить endpoint через `.env`, сохранённая настройка может иметь приоритет; проверьте Admin Panel → Connections.
7. **`vllm bench serve --model qwen3.8-27b` ходит на Hugging Face и падает 404.** Нужны `--model` / `--tokenizer` на `/app/models/Qwen3.8-27B-W4A16-AutoRound-fast` и `--served-model-name qwen3.8-27b`. Drafter `Qwen3.8-27B-DFlash2-W4A16` в `--tokenizer` не ставить.

Проверка установки внутри контейнера: `docker compose run --rm qwen verify`.

---

## Лицензия и происхождение

- **vLLM** 0.28.0 и патчи в `patches/`.
- Оптимизации: [docs/optimizations.md](docs/optimizations.md). Подводные камни: [docs/gotchas.md](docs/gotchas.md).
