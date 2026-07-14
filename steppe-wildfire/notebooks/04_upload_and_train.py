from google.colab import files
uploaded = files.upload()  # выбери steppe-wildfire.zip в открывшемся диалоге

# Берём РЕАЛЬНОЕ имя загруженного файла (а не жёстко прописанное), какое бы
# оно ни было - "steppe-wildfire.zip", "steppe-wildfire (5) (2).zip" и т.п.
zip_name = list(uploaded.keys())[0]
print(f"Загружен файл: {zip_name}")

!rm -rf steppe-wildfire
!unzip -o "{zip_name}" -d .

import os
assert os.path.exists("steppe-wildfire/scripts/train.py"), "Не нашли train.py - проверь структуру архива"
print("Распаковано ОК, train.py на месте")

!mkdir -p steppe-wildfire/data
!cp -r data/images steppe-wildfire/data/
!cp -r data/masks steppe-wildfire/data/

!cd steppe-wildfire && python3 scripts/train.py --data-dir data --in-channels 3 --batch-size 1 --epochs 20
