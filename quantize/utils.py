from pathlib import Path
from zipfile import ZipFile


def extract_coco128(zipfile_root_path: str = "") -> None:
    if not (Path(zipfile_root_path) / "coco128/images/train2017").exists():
        with ZipFile("coco128.zip", "r") as zip_ref:
            zip_ref.extractall(zipfile_root_path)
