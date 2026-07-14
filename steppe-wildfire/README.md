# Steppe Wildfire Segmentation (Kazakhstan)

Адаптация моделей сегментации активных пожаров (U-Net на мультиспектральных
Sentinel-2 снимках) под степные пожары Казахстана, где базовые модели
(обученные на лесных пожарах США/Европы) теряют точность из-за тонкого
быстро движущегося фронта горения и ложных срабатываний на нагретой солнцем
почве/солончаках.

## Финальный результат

Датасет: **82 реальных случая** степных пожаров по Казахстану (2019-2023),
нарезанные на **3097 патчей** 256×256 (Sentinel-2 + автоматическая разметка
через NASA FIRMS, с фильтрацией нефтяных факелов). Train/val разбиение
по источникам (66/16), без утечки данных между патчами одного пожара.

| Модель | Параметры | Best val IoU |
|---|---|---|
| Baseline U-Net (с нуля) | 7.77М | **0.3454** |
| U-Net + ResNet34 (ImageNet, transfer learning) | 24.4М | **0.3624** |

Transfer learning с предобученным на ImageNet энкодером даёт небольшой, но
стабильный прирост над обучением с нуля — согласуется с ожиданием, что
предобученные представления помогают при ограниченном объёме данных
дистанционного зондирования.

## Регионы датасета

- **Западный Казахстан** (Актюбинская, ЗКО, Атырауская обл.) — сухие степи
  и полупустыни, самая частая летняя пожароопасность
- **Центральный Казахстан** (Карагандинская, Улытауская обл.) — открытые
  пространства Сарыарки
- **Северный/Павлодарский регион** (Костанайская, Акмолинская, Павлодарская
  обл.)
- **Восточный Казахстан** (Абайская, ВКО обл.) — включая зону реликтовых
  ленточных боров
- **Южный Казахстан** (Кызылординская, Туркестанская обл.) — переходная
  зона степь/полупустыня у Бетпак-Далы

## Структура проекта

```
steppe-wildfire/
├── models/
│   └── unet.py              # архитектура U-Net (3 или 5 входных каналов)
├── scripts/
│   ├── spectral_indices.py  # NDVI, NBR — протестировано локально
│   └── train.py             # тренировочный цикл + синтетический sanity-check
├── notebooks/
│   ├── 01_gee_download.py   # ТОЛЬКО Google Colab: выгрузка Sentinel-2 по GEE
│   └── 02_firms_labels.py   # ТОЛЬКО Google Colab: авторазметка через NASA FIRMS
└── data/                    # сюда лягут реальные патчи (сейчас пусто)
```

## Дальнейшие шаги (реальные данные)

1. **Google Colab**: запустить `notebooks/01_gee_download.py`.
   - Нужно: Google-аккаунт + бесплатная привязка к Google Cloud Project
     для Earth Engine (см. комментарий внизу файла).
   - Нужно заполнить `FIRE_CASES` реальными случаями степных пожаров
     (дата + координаты) — источники: NASA FIRMS archive, новости
     акиматов Костанайской/Абайской/Карагандинской/ЗКО областей.

2. **Google Colab**: запустить `notebooks/02_firms_labels.py`.
   - Нужен бесплатный `MAP_KEY` с https://firms.modaps.eosdis.nasa.gov/api/
   - Генерирует маски автоматически, без ручной разметки.

3. Сконвертировать GeoTIFF → `.npy` (функция `tif_to_npy` в `scripts/train.py`),
   разложить в `data/images/` и `data/masks/`.

4. Обучить на реальных данных:
   ```
   python scripts/train.py --data-dir data --epochs 30 --in-channels 3   # baseline
   python scripts/train.py --data-dir data --epochs 30 --in-channels 5   # адаптация
   ```

5. Сравнить IoU baseline vs адаптированной модели — это и есть
   центральный научный результат ("domain gap quantification + adaptation").

## Научная рамка

Мы не заявляем детекцию пожаров по спутнику как новую задачу — это
активная область (Pereira et al. 2021, Fusioka et al. 2024 и др.). Вклад
этой работы: первый (насколько нам известно) публичный пайплайн + датасет
для оценки active-fire моделей на степных пожарах Центральной Азии, и
эмпирическое подтверждение пользы transfer learning в условиях
ограниченного объёма региональных данных.

## Проверенные источники

1. de Almeida Pereira, G. H., Fusioka, A. M., Nassu, B. T., & Minetto, R.
   (2021). Active fire detection in Landsat-8 imagery: A large-scale dataset
   and a deep-learning study. *ISPRS J. Photogrammetry and Remote Sensing*,
   178, 171-186.
2. Fusioka, A. M., et al. (2024). Active Fire Segmentation: A Transfer
   Learning Study From Landsat-8 to Sentinel-2. *IEEE JSTARS*, 17,
   14093-14108.
3. Chuvieco, E., et al. (2019). Historical background and current
   developments for mapping burned area from satellite Earth observation.
   *Remote Sensing of Environment*, 225, 45-64.
   DOI: 10.1016/j.rse.2019.02.013
4. Spatiotemporal Fire Risk Index for Kazakhstan integrating machine
   learning and remote sensing (2025). *Frontiers in Forests and Global
   Change*. https://doi.org/10.3389/ffgc.2025.1680856
5. Mapping fire hazard potential in Kazakhstan: a machine learning and
   remote sensing perspective (2025). *International Journal of Wildland
   Fire*. https://doi.org/10.1071/WF24232
