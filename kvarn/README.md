# KV-кэш KVarN, порт на vLLM 0.28.0

[KVarN](https://github.com/huawei-csl/KVarN) — схема сжатия KV-кэша: вращение Адамара, итеративная нормализация дисперсии, 4-бит ключи / 2-бит значения на тайл 128 токенов. Изначально это нативный attention backend внутри форка vLLM 0.23.0. Этот каталог — порт того backend на vLLM 0.28.0, который крутит этот репозиторий: только dense-путь (не MLA), заточенный под Qwen3.8-27B / GPU 24 ГБ.

Что внутри:

- `files/vllm/...` — модули KVarN (backend, ядра Triton, конфиг, эталон Sinkhorn), скопированные из KVarN и подогнанные под API backend 0.28.0 (маркеры исходной адаптации в исходниках сохранены).
- `kvarn-0.28.0.patch` — небольшие hunk’и, чтобы апстрим-vLLM знал новые `kvarn_*` cache dtype (литералы dtype, карта dtype, реестр backend + приоритет, `KVQuantMode.KVARN`, ветка KV-спеки в attention-слое и выравнивание страниц гибридной модели).
- `kvarn-v2-runner-0.28.0.patch` — V2 runner, sliding-cache и правки корректности DFlash2 поверх базового порта.
- `install.sh` — копирует модули в `venv/lib/python3.12/site-packages/vllm` и накладывает патч (можно запускать повторно).

Заметки по порту — тому, кто будет бампать vLLM:

- Спека attention в 0.28.0 ставит `cache_dtype_str="auto"` для спек с `kv_quant_mode` = `NONE`. Форма KVarN зависит от пресета, поэтому порт добавляет `KVQuantMode.KVARN` и передаёт размер packed-слота через `FullAttentionSpec(state_content_bytes=...)`. Без этого движок падает на инициализации KV-кэша.
- Связка impl→builder идёт через `get_layers_from_vllm_config`, а не через hunk `impl.layer_name` в `attention.py` у KVarN, плюс маленький owner-реестр, чтобы MTP draft-слой не сбрасывали два builder’а.
- Пулы материализуются во время `profile_run` (forward с `attn_metadata=None`), чтобы профайлер памяти vLLM их учитывал — hunk в `gpu_worker.py` не нужен.
- Padding слота на токен до степени двойки (у KVarN это было ради смешанных head_dim у Gemma-4) здесь **выключен** по умолчанию (`KVARN_POW2_SLOT=1` возвращает): при head_dim 256 это 840 Б/токен/слой вместо 1024 (fp8: 2048).
- Гибридное выравнивание делает блок attention 2048 токенов (страница должна совпасть со страницей Gated DeltaNet 1.63 МБ); vLLM режет его на kernel-тайлы по 128 токенов, инвариант KVarN `tile == kernel block` выполняется.
- Мелкие правки устойчивости: защита от NaN в online-softmax на полностью маскированных чанках / пустых split-K строках; без перекомпиляции packed-KV ядра на каждый контекст; padding verify-плана обнуляется для replay CUDA-графов.
- Не портировано: путь MLA, `TQSlidingWindowSpec` (здесь нет sliding-window слоёв), hunk конфига Gemma-4.

Замеры на GPU 24 ГБ: контекст 262k помещается (пул 420k токенов при 4 слотах против ~200k у fp8), needle-in-a-haystack корректен на 4k…240k, perplexity +0.16%, decode ~на 20% медленнее fp8 на контексте 100k, MTP работает, throughput коротких запросов ниже (блоки по 2048 токенов стоят как 800-токенный блок fp8, плюс время на flush prefill).
