from google.colab import files
uploaded = files.upload()  # pick steppe-wildfire.zip in the dialog

# Uses the ACTUAL uploaded filename (not hardcoded), whatever Colab
# ends up calling it ("steppe-wildfire.zip", "steppe-wildfire (2).zip", etc.)
zip_name = list(uploaded.keys())[0]
print(f"Uploaded file: {zip_name}")

!rm -rf steppe-wildfire
!unzip -o "{zip_name}" -d .

import os
assert os.path.exists("steppe-wildfire/scripts/train.py"), "train.py not found - check the archive structure"
print("Unpacked OK, train.py is in place")

!mkdir -p steppe-wildfire/data
!cp -r data/images steppe-wildfire/data/
!cp -r data/masks steppe-wildfire/data/

!cd steppe-wildfire && python3 scripts/train.py --data-dir data --in-channels 3 --batch-size 4 --epochs 30
