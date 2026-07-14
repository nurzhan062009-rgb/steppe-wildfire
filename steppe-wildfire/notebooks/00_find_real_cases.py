"""
Поиск реальных случаев пожаров через архив NASA FIRMS (вместо гадания по
новостям). Запускать в Google Colab.

Идея: вместо того чтобы искать координаты в новостях (там их обычно нет),
качаем ВСЕ точки активного огня FIRMS по Казахстану за нужный период,
и сами выбираем кластеры точек в открытой степи (не в лесу/не рядом с
городами - это можно проверить глазами на карте по координатам).

Получи бесплатный MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/
"""

import io
import pandas as pd
import requests

FIRMS_MAP_KEY = "YOUR_FIRMS_MAP_KEY"

# Bounding box всего Казахстана (грубо)
KZ_BBOX = (46.0, 40.5, 87.5, 55.5)  # lon_min, lat_min, lon_max, lat_max

# Степные/засушливые области - для справки, куда потом смотреть на карте:
# Костанайская, Акмолинская, Павлодарская, Карагандинская, ЗКО, Актюбинская


def fetch_country_archive(source: str, start_date: str, days: int = 5) -> pd.DataFrame:
    """
    Качает точки FIRMS по всему bbox Казахстана за период.

    source: "VIIRS_SNPP_SP" (архивные данные, для дат старше ~2 месяцев)
    start_date: "YYYY-MM-DD" - начало периода
    days: сколько дней подряд качать. ВАЖНО: у FIRMS Area API жёсткий лимит
          1..5 дней за один запрос (не 10!). Для более длинного периода нужно
          делать несколько последовательных запросов и склеивать результат
          (см. fetch_multi_week ниже).
    """
    lon_min, lat_min, lon_max, lat_max = KZ_BBOX
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/"
        f"{source}/{lon_min},{lat_min},{lon_max},{lat_max}/{days}/{start_date}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    return df


def fetch_multi_week(source: str, start_date: str, total_days: int = 31) -> pd.DataFrame:
    """
    Качает данные за произвольный период (например, весь август = 31 день),
    разбивая на чанки по 5 дней (лимит FIRMS Area API) и склеивая результат.
    """
    import datetime
    import time

    chunks = []
    current = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    remaining = total_days

    while remaining > 0:
        chunk_days = min(5, remaining)
        date_str = current.strftime("%Y-%m-%d")
        print(f"  Качаю {date_str} + {chunk_days} дн...")
        try:
            df_chunk = fetch_country_archive(source, date_str, days=chunk_days)
            chunks.append(df_chunk)
        except requests.HTTPError as e:
            print(f"  Ошибка на {date_str}: {e}")
        current += datetime.timedelta(days=chunk_days)
        remaining -= chunk_days
        time.sleep(1)  # вежливая пауза, чтобы не упереться в rate limit

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


# Известные нефтегазоносные области Казахстана (грубые bbox), где VIIRS
# массово ловит факелы сжигания попутного газа, а не пожары. Это не научная
# точность, а быстрый практический фильтр по географии.
OIL_REGIONS_BBOX = [
    # (lon_min, lat_min, lon_max, lat_max, название)
    (50.0, 42.0, 59.5, 46.5, "Мангистауская обл. (Узень/Жетыбай/Бузачи/Каракия)"),
    (46.5, 44.0, 54.5, 48.0, "Атырауская обл. (Тенгиз/Кашаган/Прорва)"),
    (48.0, 48.5, 55.0, 52.0, "ЗКО (Карачаганак)"),
]


def is_in_oil_region(lat: float, lon: float) -> bool:
    for lon_min, lat_min, lon_max, lat_max, _name in OIL_REGIONS_BBOX:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return True
    return False


def filter_likely_flares(df: pd.DataFrame, round_decimals: int = 1) -> pd.DataFrame:
    """
    Отсеивает вероятные газовые факелы двумя эвристиками:
      1) Точка лежит в известном нефтегазоносном регионе (см. OIL_REGIONS_BBOX).
      2) Точка "стационарна" - почти те же координаты (округлённые до
         ~1 км) встречаются в датасете на 2+ РАЗНЫХ датах. Настоящий степной
         пожар вспыхивает и гаснет за 1-3 дня и не сидит месяцами на одном
         пятне - а факел горит постоянно.

    Возвращает только точки, прошедшие оба фильтра (вероятные реальные пожары).
    """
    df = df.copy()
    df["in_oil_region"] = df.apply(lambda r: is_in_oil_region(r["latitude"], r["longitude"]), axis=1)

    df["lat_round"] = df["latitude"].round(round_decimals)
    df["lon_round"] = df["longitude"].round(round_decimals)
    n_unique_dates = df.groupby(["lat_round", "lon_round"])["acq_date"].transform("nunique")
    df["is_stationary"] = n_unique_dates >= 2

    n_before = len(df)
    filtered = df[~df["in_oil_region"] & ~df["is_stationary"]].drop(
        columns=["in_oil_region", "is_stationary", "lat_round", "lon_round"]
    )
    print(f"Отфильтровано: {n_before} -> {len(filtered)} точек "
          f"(убраны нефтяные регионы + стационарные факелы)")
    return filtered


def find_high_confidence_clusters(df: pd.DataFrame, min_confidence: str = "n") -> pd.DataFrame:
    """
    Фильтрует точки по уверенности детекции и сортирует по FRP
    (Fire Radiative Power - чем выше, тем интенсивнее горение = крупнее пожар,
    легче найти на снимке).
    """
    if "confidence" in df.columns:
        # VIIRS confidence: l/n/h (low/nominal/high)
        conf_order = {"l": 0, "n": 1, "h": 2}
        min_level = conf_order.get(min_confidence, 1)
        df = df[df["confidence"].map(lambda c: conf_order.get(c, 0)) >= min_level]

    df = df.sort_values("frp", ascending=False)
    return df[["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]]


if __name__ == "__main__":
    # Известные засушливые/пожароопасные периоды: июль-сентябрь
    # Пример: весь август 2023 (степной сезон пожаров), качаем чанками по 5 дней
    df = fetch_multi_week(source="VIIRS_SNPP_SP", start_date="2023-08-01", total_days=31)
    print(f"\nВсего точек FIRMS по Казахстану за август 2023: {len(df)}")

    df_clean = filter_likely_flares(df)
    top_clusters = find_high_confidence_clusters(df_clean, min_confidence="n")
    print("\nТоп-20 вероятных настоящих пожаров (по FRP), с координатами:")
    print(top_clusters.head(20).to_string(index=False))
    print(
        "\nЭто уже отфильтрованный список (без известных нефтяных регионов и "
        "без стационарных точек). Но эвристика грубая - всё равно быстро "
        "глянь координаты глазами на Google Maps перед тем как выгружать снимки."
    )

    print(
        "\nДальше:\n"
        "  1. Возьми 5-10 координат из top_clusters с наибольшим FRP.\n"
        "  2. Вбей координаты в Google Maps/Google Earth - глазами проверь,\n"
        "     что это ОТКРЫТАЯ СТЕПЬ (поле/трава), а не лес и не населённый пункт.\n"
        "  3. Эти (acq_date, longitude, latitude) вставляешь в FIRE_CASES\n"
        "     в 01_gee_download.py.\n"
    )
