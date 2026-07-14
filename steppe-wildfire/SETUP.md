# Как воспроизвести это исследование с нуля

Весь пайплайн запускается в Google Colab (GEE и Google Drive требуют
Google-аккаунт). Локально можно запускать только `models/`, `scripts/train.py --synthetic`
и `scripts/spectral_indices.py` — там просто PyTorch/NumPy, интернет не нужен.

## Что понадобится

- Google-аккаунт
- Google Cloud проект, зарегистрированный под Earth Engine (бесплатно):
  https://code.earthengine.google.com/register — выбери Unpaid usage →
  Academic/Individual → Community Tier
- Бесплатный API-ключ NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/api/

## Порядок запуска в Colab

1. **`notebooks/00_find_real_cases.py`** — ищет реальные случаи пожаров через
   архив NASA FIRMS, отфильтровывает нефтяные факелы и стационарные
   источники. Вставь свой FIRMS-ключ вместо `YOUR_FIRMS_MAP_KEY`.
   Для большого набора кейсов используй `notebooks/05_bulk_find_cases.py`
   (сканирует несколько лет разом).

2. **`notebooks/01_gee_download.py`** — выгружает снимки Sentinel-2 через
   Earth Engine. Нужно:
   - `ee.Initialize(project="...")` — подставь свой GCP project ID
   - Заполнить `FIRE_CASES` координатами из шага 1
   - При большом числе случаев (50+) экспортируй партиями через
     `BATCH_INDEX` (см. комментарий в файле), иначе упрёшься в лимиты GEE

3. **`notebooks/06_bulk_masks.py`** — конвертирует скачанные GeoTIFF в
   `.npy` и генерирует маски пожара через FIRMS (буфер вокруг hotspot).
   Тоже требует FIRMS-ключ. Список `FIRE_CASES` должен совпадать с шагом 2.

4. **`scripts/patchify.py --data-dir data --out-dir data_patches`** — режет
   крупные снимки на патчи 256×256. Работает локально, интернет не нужен.

5. **`scripts/train.py`** — обучение:
   ```
   python scripts/train.py --data-dir data_patches --in-channels 3 --crop-size 256 --epochs 50 --save-path model_baseline.pth
   python scripts/train.py --data-dir data_patches --in-channels 5 --crop-size 256 --epochs 50 --save-path model_ndvi_nbr.pth
   ```
   `--in-channels 3` — baseline (SWIR2, SWIR1, NIR), `5` — с NDVI/NBR.
   На CPU обучение на патчах будет медленным — переключи Colab runtime на GPU
   (Runtime → Change runtime type → T4 GPU).

## Быстрая проверка без реальных данных

```
python scripts/train.py --synthetic --epochs 5 --in-channels 3
```
Прогоняет весь пайплайн (Dataset → модель → лосс → IoU) на синтетике —
полезно, чтобы убедиться, что код не сломан, до похода в GEE/FIRMS.
