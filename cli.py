from argparse import ArgumentParser
from pathlib import Path

import cv2

from fastsam_lib.fast_sam import FASTSAM_OV


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument(
        "--img", required=True, type=str, help="Path to image as input"
    )
    general.add_argument(
        "--output", type=str, default=None, help="Path to save inference result"
    )

    model = parser.add_argument_group("Model")
    model.add_argument("--model_name", type=str, default="FastSAM-x")
    model.add_argument(
        "--quantize",
        default=False,
        action="store_true",
        help="Use quantized model to inference",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.quantize:
        ov_model_path = f"{args.model_name}_quantized_model"
    else:
        ov_model_path = f"{args.model_name}_openvino_model"

    FASTSAM_OV_SERVER = FASTSAM_OV(
        ov_model_path=Path(ov_model_path) / f"{args.model_name}.xml",
        model_name=args.model_name,
    )

    img = cv2.imread(args.img)
    ov_result = FASTSAM_OV_SERVER.infer(img)
    mask_result = FASTSAM_OV_SERVER.postprocess_mask(ov_result)
    print(type(mask_result))

    # img_array = ov_result.plot()
    # save_path = f"{args.output}.jpg" if args.output is not None else f"{Path(args.img).stem}_result.jpg"
    # cv2.imwrite(save_path, img_array)
    """
    python3 cli.py --img test1.png --quantize
    """
