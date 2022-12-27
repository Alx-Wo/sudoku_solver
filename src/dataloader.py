import json
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.images = self.data_dir.glob("*.jpg")

    def img2label(self, img_path: Path) -> Path:
        label_path = img_path.with_suffix(".json")
        return label_path

    def __iter__(self):
        for image in self.images:
            img_path = image
            label_path = self.img2label(img_path=img_path)
            with open(label_path, "r", encoding="utf-8") as fid:
                labels = json.load(fid)
            yield {
                "img_path": img_path,
                "input": labels["input"],
                "solution": labels["solution"],
            }

    def __next__(self):
        pass
