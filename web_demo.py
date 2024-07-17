from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from fastsam_lib.fast_sam import FASTSAM_OV
from utils import utils


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    model = parser.add_argument_group("Model")
    model.add_argument("--model_name", type=str, default="FastSAM-x")
    model.add_argument(
        "--quantize",
        default=False,
        action="store_true",
        help="Use quantized model to inference",
    )

    return parser


EXAMPLES = [
    ["component/test1.png"],
    ["component/test2.png"],
    ["component/test3.png"],
    ["component/test4.png"],
    ["component/test5.png"],
    ["component/test6.png"],
]

object_points = []
background_points = []
bbox_points = []
last_image = EXAMPLES[0][0]


def select_point(
    img: Image.Image, task_type: str, point_type: str, event: gr.SelectData
) -> Image.Image:
    """Gradio select callbask"""
    if task_type != "Get mask":
        return img

    img = img.convert("RGBA")
    x, y = event.index[0], event.index[1]
    point_radius = np.round(max(img.size) / 100)

    if point_type == "Object point":
        object_points.append((x, y))
        color = (30, 255, 30, 200)
    elif point_type == "Background point":
        background_points.append((x, y))
        color = (255, 30, 30, 200)
    elif point_type == "Bounding box":
        bbox_points.append((x, y))
        color = (10, 10, 255, 255)

        if len(bbox_points) % 2 == 0:
            # Draw a rectangle if number of points is even
            new_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
            _draw = ImageDraw.Draw(new_img)
            x0, y0, x1, y1 = *bbox_points[-2], *bbox_points[-1]
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            # Save sorted order
            bbox_points[-2] = (x0, y0)
            bbox_points[-1] = (x1, y1)
            _draw.rectangle((x0, y0, x1, y1), fill=(*color[:-1], 90))
            img = Image.alpha_composite(img, new_img)

    ImageDraw.Draw(img).ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=color,
    )
    return img


def clear_points() -> Tuple[Image.Image, None]:
    """Gradio clear points callback."""
    global object_points, background_points, bbox_points
    object_points = []
    background_points = []
    bbox_points = []
    return last_image, None


def save_last_picked_image(img: Image.Image) -> None:
    global last_image
    last_image = img
    clear_points()
    return None


def update_visual_type(task_type: str) -> gr.update:
    if task_type == "Get bbox":
        return gr.update(value="bbox")
    elif task_type == "Get mask":
        return gr.update(value="mask")


def update_point_type(task_type: str) -> gr.update:
    if task_type == "Get mask":
        return gr.update(interactive=True, value="Object point")
    else:
        return gr.update(interactive=False, value=None)


def update_model_type(FASTSAM_OV__QUANTIZED_SERVER) -> gr.update:
    if FASTSAM_OV__QUANTIZED_SERVER:
        return gr.update(
            choices=["FP32 model", "Quantized model"],
            value="FP32 model",
            interactive=True,
        )
    else:
        return gr.update(
            choices=["FP32 model", "Quantized model"],
            value="FP32 model",
            interactive=False,
        )


def segment_image(original_img, model_type, conf, iou):
    global \
        FASTSAM_OV_SERVER, \
        FASTSAM_OV_QUANTIZED_SERVER, \
        object_points, \
        background_points, \
        bbox_points
    if model_type == "FP32 model":
        model = FASTSAM_OV_SERVER
    elif model_type == "Quantized model":
        model = FASTSAM_OV_QUANTIZED_SERVER

    result = utils.segment(
        model=model,
        image=original_img,
        object_points=object_points,
        background_points=background_points,
        bbox_points=bbox_points,
        conf=conf,
        iou=iou,
    )
    clear_points()
    return result


def bbox_image(original_img, model_type, conf, iou):
    global FASTSAM_OV_SERVER, FASTSAM_OV_QUANTIZED_SERVER
    if model_type == "FP32 model":
        model = FASTSAM_OV_SERVER
    elif model_type == "Quantized model":
        model = FASTSAM_OV_QUANTIZED_SERVER

    return utils.bbox(model=model, image=original_img, conf=conf, iou=iou)


def inference(original_img, task_type, model_type, conf, iou):
    if task_type == "Get mask":
        return segment_image(
            original_img=original_img, model_type=model_type, conf=conf, iou=iou
        )
    elif task_type == "Get bbox":
        return bbox_image(
            original_img=original_img, model_type=model_type, conf=conf, iou=iou
        )


def main(args):
    global FASTSAM_OV_SERVER, FASTSAM_OV_QUANTIZED_SERVER
    FASTSAM_OV_SERVER = None
    FASTSAM_OV_QUANTIZED_SERVER = None
    if args.quantize:
        ov_quantized_model_path = (
            Path(f"{args.model_name}_quantized_model") / f"{args.model_name}.xml"
        )
        FASTSAM_OV_QUANTIZED_SERVER = FASTSAM_OV(
            ov_model_path=ov_quantized_model_path, model_name=args.model_name
        )

    ov_model_path = Path(f"{args.model_name}_openvino_model") / f"{args.model_name}.xml"
    FASTSAM_OV_SERVER = FASTSAM_OV(
        ov_model_path=ov_model_path, model_name=args.model_name
    )

    with gr.Blocks(title="FastSAM OpenVINO") as demo:
        with gr.Row(variant="panel"):
            original_img = gr.Image(label="Input", type="pil")
            segmented_img = gr.Image(label="Segmentation", type="pil")
        with gr.Row(variant="panel"):
            task_type = gr.Radio(
                ["Get bbox", "Get mask"], value="Get bbox", label="Select task"
            )
            point_type = gr.Radio(
                ["Object point", "Background point", "Bounding box"],
                label="Select point type (Only supports 'Get mask' task)",
            )
            model_type = gr.Radio(
                ["FP32 model", "Quantized model"],
                value="FP32 model",
                label="Select model inference",
            )
            visual_type = gr.Radio(["bbox", "mask"], label="Visualize type")
        with gr.Row(variant="panel"):
            with gr.Column(variant="panel"):
                conf = gr.Slider(value=0.6, step=0.05, maximum=1.0, label="conf")
                iou = gr.Slider(value=0.9, step=0.05, maximum=1.0, label="iou")
            with gr.Column(variant="panel"):
                infer_button = gr.Button("Inference", value="primary")
                clear_botton = gr.Button("Clear points", variant="secondary")
        gr.Examples(EXAMPLES, inputs=original_img, outputs=segmented_img)

        # Callbacks
        original_img.select(
            select_point,
            inputs=[original_img, task_type, point_type],
            outputs=[original_img],
        )
        original_img.upload(
            save_last_picked_image, inputs=original_img, outputs=segmented_img
        )
        clear_botton.click(clear_points, outputs=[original_img, segmented_img])
        task_type.change(update_visual_type, inputs=task_type, outputs=visual_type)
        task_type.change(update_point_type, inputs=task_type, outputs=point_type)

        demo.load(
            lambda: update_model_type(FASTSAM_OV_QUANTIZED_SERVER),
            inputs=None,
            outputs=model_type,
        )

        infer_button.click(
            inference,
            inputs=[original_img, task_type, model_type, conf, iou],
            outputs=segmented_img,
        )

    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7788)


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
