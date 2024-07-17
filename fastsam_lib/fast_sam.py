from pathlib import Path
from typing import Tuple

import numpy as np
import openvino as ov
import torch
from ultralytics import FastSAM
from ultralytics.engine.results import Results


class OVWrapper:
    def __init__(
        self, core: ov.Core, ov_model: str, device: str = "CPU", stride: int = 32
    ) -> None:
        self.model = core.compile_model(ov_model, device_name=device)
        self.stride = stride
        self.pt = True
        self.fp16 = False
        self.names = {0: "object"}

    def __call__(self, im, **_) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.model(im)
        return torch.from_numpy(result[0]), torch.from_numpy(result[1])


class FASTSAM_OV:
    def __init__(
        self,
        ov_model_path: Path,
        model_name: str = "FastSAM-x",
        device: str = "CPU",
        imgsz: int = 1024,
        conf: float = 0.6,
        iou: float = 0.9,
        retina_masks: bool = True,
    ) -> None:
        self.model = FastSAM(model_name)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.retina_masks = retina_masks

        dummy_input = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.model(
            dummy_input,
            device=device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )

        if not ov_model_path.exists():
            self.convert(model=self.model, ov_model_path=ov_model_path)

        wrapper_model = OVWrapper(
            core=ov.Core(),
            ov_model=ov_model_path,
            device=device,
            stride=self.model.predictor.model.stride,
        )
        self.model.predictor.model = wrapper_model

    def convert(self, model: FastSAM, ov_model_path: Path) -> None:
        ov_model = model.export(format="openvino", dynamic=True, half=False)

    def infer(
        self,
        img_src: np.array,
        retina_masks: bool = None,
        imgsz: int = None,
        conf: float = None,
        iou: float = None,
    ) -> Results:
        results = self.model(
            img_src,
            device=self.device,
            retina_masks=self.retina_masks if retina_masks is None else retina_masks,
            imgsz=self.imgsz if imgsz is None else imgsz,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
        )
        return results[0]

    def postprocess_bbox(self, results: Results) -> list:
        boxes_list, scores_list = (
            results.boxes.xyxy.tolist(),
            results.boxes.conf.tolist(),
        )
        assert len(boxes_list) == len(scores_list)
        return [list(item) for item in zip(boxes_list, scores_list)]

    def postprocess_mask(self, results: Results) -> torch.Tensor:
        return results.masks.data
