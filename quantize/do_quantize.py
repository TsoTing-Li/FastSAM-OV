import pickle
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import cv2
import nncf
import numpy as np
import openvino as ov
import torch
import torch.utils
from tqdm.auto import tqdm
from ultralytics import FastSAM

import utils


class CalibrationDateCollector:
    def __init__(self) -> None:
        self.collecting = False
        self.calibration_data = list()

    @contextmanager
    def collect(self):
        try:
            self.collecting = True
            yield
        finally:
            self.collecting = False

    def add(self, data):
        if self.collecting:
            self.calibration_data.append(data)


class NNCFWrapper:
    def __init__(
        self,
        core: ov.Core,
        ov_model: str,
        stride: int = 32,
        device: str = "CPU",
        data_collector=None,
    ) -> None:
        self.model = core.read_model(ov_model)
        self.compiled_model = core.compile_model(self.model, device_name=device)

        self.stride = stride
        self.pt = True
        self.fp16 = False
        self.names = {0: "object"}
        self.data_collector = data_collector

    def __call__(self, im, **_) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data_collector:
            self.data_collector.add(im)

        result = self.compiled_model(im)
        return torch.from_numpy(result[0]), torch.from_numpy(result[1])


class COCOLoader(torch.utils.data.Dataset):
    def __init__(self, images_path: str) -> None:
        self.images_list = list(Path(images_path).iterdir())

    def __getitem__(self, index) -> None:
        if isinstance(index, slice):
            return [
                self.read_image(str(img_path)) for img_path in self.images_list[index]
            ]
        return self.read_image(str(self.images_list[index]))

    def read_image(self, image_path: str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return len(self.images_list)


def collect_calibration_data_for_decoder(
    model, calibration_dataset_size: int, calibration_cache_path: Path, data_collector
) -> list:
    if not calibration_cache_path.exists():
        utils.extract_coco128()
        coco_dataset = COCOLoader(Path("coco128/images/train2017"))

        with data_collector.collect():
            for image in tqdm(
                coco_dataset[:calibration_dataset_size],
                desc="Collecting calibration data",
            ):
                model(
                    image,
                    retina_masks=True,
                    imgsz=640,
                    conf=0.6,
                    iou=0.9,
                    verbose=False,
                )

            calibration_cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(calibration_cache_path, "wb") as f:
                pickle.dump(data_collector.calibration_data, f)
    else:
        with open(calibration_cache_path, "rb") as f:
            data_collector.calibration_data = pickle.load(f)

    return data_collector.calibration_data


def quantize(
    model,
    save_model_path: Path,
    calibration_cache_path: Path,
    calibration_dataset_size: int,
    preset: nncf.QuantizationPreset,
    data_collector,
) -> None:
    calibration_data = collect_calibration_data_for_decoder(
        model=model,
        calibration_dataset_size=calibration_dataset_size,
        calibration_cache_path=calibration_cache_path,
        data_collector=data_collector,
    )
    quantized_ov_decoder = nncf.quantize(
        model=model.predictor.model.model,
        calibration_dataset=nncf.Dataset(calibration_data),
        preset=preset,
        subset_size=len(calibration_data),
        fast_bias_correction=True,
        ignored_scope=nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],
            names=[
                "/model.22/dfl/conv/Conv",  # in the post-processing subgraph
                "/model.22/Add",
                "/model.22/Add_1",
                "/model.22/Add_2",
                "/model.22/Add_3",
                "/model.22/Add_4",
                "/model.22/Add_5",
                "/model.22/Add_6",
                "/model.22/Add_7",
                "/model.22/Add_8",
                "/model.22/Add_9",
                "/model.22/Add_10",
            ],
        ),
    )
    ov.save_model(quantized_ov_decoder, save_model_path)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    model = parser.add_argument_group("Model")
    model.add_argument("--model_name", type=str, default="FastSAM-x")
    model.add_argument("--ov_model_path", required=True, type=str)

    calibration = parser.add_argument_group("Calibration")
    calibration.add_argument(
        "--dataset_size", type=int, default=128, help="Number of calibration dataset"
    )
    calibration.add_argument("--cache_path", type=str, default="calibration_data")

    return parser


def main(args: ArgumentParser) -> None:
    ori_model = FastSAM(args.model_name)
    dummy_input = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    ori_model(
        dummy_input, device="CPU", retina_masks=True, imgsz=640, conf=0.6, iou=0.9
    )

    data_collector = CalibrationDateCollector()
    wrapped_model = NNCFWrapper(
        core=ov.Core(), ov_model=args.ov_model_path, data_collector=data_collector
    )
    ori_model.predictor.model = wrapped_model

    quantized_model_path = Path(f"{args.model_name}_quantized_model")
    if not quantized_model_path.is_dir():
        quantized_model_path.mkdir(parents=True, exist_ok=True)

    if not (quantized_model_path / f"{args.model_name}.xml").exists():
        quantize(
            model=ori_model,
            save_model_path=quantized_model_path / f"{args.model_name}.xml",
            calibration_cache_path=Path(args.cache_path)
            / f"coco{args.dataset_size}.pkl",
            calibration_dataset_size=args.dataset_size,
            preset=nncf.QuantizationPreset.MIXED,
            data_collector=data_collector,
        )


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
