# prepare/ — одноразовая подготовка модели

Публичный W4A16 quant Qwen3.8-27B на 24 ГБ как есть не встаёт: две bf16-матрицы эмбеддингов по 2.5 ГБ и неквантованный MTP draft-модуль. Эти скрипты чинят это **на месте**, на CPU, один раз. `docker compose run --rm qwen prepare`
(см. [docker/prepare.sh](../docker/prepare.sh)) запускает ровно их; шаг пропускается, если результат уже есть в каталоге модели.

Запускать из корня репозитория, **по порядку** — сначала `quant_lm_head.py`, потому что `build_draft_vocab.py` режет его строки:

```bash
V=venv/bin/python; M=models/Qwen3.8-27B-W4A16-AutoRound
$V prepare/quant_lm_head.py $M      # lm_head -> int8 group-128, на месте: ~1.3 ГБ
$V prepare/quant_embed.py   $M      # embed_tokens так же (untied): ещё ~1.3 ГБ
$V prepare/quant_mtp.py     $M      # модуль mtp.* (~850 МБ bf16) -> int8
$V prepare/build_draft_vocab.py $M --ids prepare/draft_vocab_ids.json
$V prepare/fetch_fast_variant.py    # опционально, ~1 ГБ: «fast»-вариант для single-user
$V prepare/fetch_dflash2.py         # опционально, 1.2 ГБ: DFlash2 drafter (SPEC=dflash2)
```

`build_draft_vocab.py` пишет срез `lm_head` на 40 960 строк — MTP-drafter считает по нему, а не по полному словарю 248k. `draft_vocab_ids.json` — поставленный список id; `--corpus` считает свой. Нужен патч [patches/qwen3_5-mtp-draft-vocab.patch](../patches/qwen3_5-mtp-draft-vocab.patch).

## Другой checkpoint

`quant_heads_stream.py` делает работу `quant_lm_head.py` + `quant_embed.py` + `quant_mtp.py` за один проход — для checkpoint’ов, которые те три не открывают: **single-shard** (они читают шард целиком в RAM; uncensored-сборка кладёт один `model.safetensors` на 18.6 ГБ) и **асимметричный AWQ** (клонируют `config_groups.group_0` на симметричные тензоры, после чего vLLM ищет несуществующий `weight_zero_point`). Та же математика, те же выходные тензоры, пик RSS заметно ниже размера шарда (на примере 18.6 ГБ измерили 9.7 ГБ — всё равно не инструмент для мало RAM).

```bash
$V prepare/fetch_thirdparty.py                          # ~18.6 ГБ (или: fetch_thirdparty.py <hf-repo>)
$V prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16
$V prepare/build_draft_vocab.py  models/Qwen3.8-27B-Uncensored-W4A16 \
  --ids prepare/draft_vocab_ids.json
```

Дальше `MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 bash single-user/start_qwen.sh`.
Для `SPEC=dflash2` ещё нужно увеличить закреплённый пул: этот checkpoint тяжелее того, на котором мерили константы, примерно на 1 ГБ.
`--mtp-bits 4` и `--keep-fc` — для экспериментов с draft-модулем; проверены дефолты (int8, `mtp.fc` квантован).

Два скрипта `fetch_*` только качают: fast-вариант — int4-GPTQ lm_head и drafter плюс draft-словарь по собственным выходам модели; `fetch_dflash2.py` — блочный drafter DFlash2 в W4A16.

`bash verify.sh --no-server` проверяет каждый шаг выше по каталогу модели и называет скрипт, которого не хватает. Каждый in-place скрипт кладёт бэкап рядом с оригиналом (`.bak*`), шаг можно откатить без повторной загрузки 19.5 ГБ. Зачем каждый шаг и с цифрами:
[docs/optimizations.md](../docs/optimizations.md).
