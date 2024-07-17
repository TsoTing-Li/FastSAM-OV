import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def bbox(
    model,
    image,
    imgsz: int = 1024,
    conf: float = 0.6,
    iou: float = 0.9,
    use_retina: bool = True,
):
    w, h = image.size
    scale = imgsz / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h))

    results = model.infer(
        img_src=image, retina_masks=use_retina, imgsz=imgsz, conf=conf, iou=iou
    )
    anno_info = model.postprocess_bbox(results)
    return draw_bbox(image=image, anno_info=anno_info)


def draw_bbox(image, anno_info: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for bbox, _ in anno_info:
        x1, y1, x2, y2 = bbox
        draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline="red", width=5)
    return image


def segment(
    model,
    image,
    object_points: list,
    background_points: list,
    bbox_points: list,
    imgsz: int = 1024,
    conf: float = 0.6,
    iou: float = 0.9,
    use_retina: bool = True,
    with_contours: bool = True,
    mask_random_color: bool = True,
    better_quality: bool = True,
):
    w, h = image.size
    scale = imgsz / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h))

    results = model.infer(
        img_src=image, retina_masks=use_retina, imgsz=imgsz, conf=conf, iou=iou
    )
    masks = model.postprocess_mask(results)

    if not (object_points or bbox_points):
        annotations = masks.cpu().numpy()
    else:
        annotations = []

    if object_points:
        all_points = object_points + background_points
        labels = [1] * len(object_points) + [0] * len(background_points)
        scaled_points = [[int(x * scale) for x in point] for point in all_points]
        h, w = masks[0].shape[:2]
        assert max(h, w) == imgsz
        onemask = np.zeros((h, w))
        for mask in sorted(masks, key=lambda x: x.sum(), reverse=True):
            mask_np = (mask == 1.0).cpu().numpy()
            for point, label in zip(scaled_points, labels):
                if mask_np[point[1], point[0]] == 1 and label == 1:
                    onemask[mask_np] = 1
                if mask_np[point[1], point[0]] == 1 and label == 0:
                    onemask[mask_np] = 0
        annotations.append(onemask >= 1)
    if len(bbox_points) >= 2:
        scaled_bbox_points = []
        for i, point in enumerate(bbox_points):
            x, y = int(point[0] * scale), int(point[1] * scale)
            x = max(min(x, new_w), 0)
            y = max(min(y, new_h), 0)
            scaled_bbox_points.append((x, y))

        for i in range(0, len(scaled_bbox_points) - 1, 2):
            x0, y0, x1, y1 = *scaled_bbox_points[i], *scaled_bbox_points[i + 1]

            intersection_area = torch.sum(masks[:, y0:y1, x0:x1], dim=(1, 2))
            masks_area = torch.sum(masks, dim=(1, 2))
            bbox_area = (y1 - y0) * (x1 - x0)

            union = bbox_area + masks_area - intersection_area
            iou = intersection_area / union
            max_iou_index = torch.argmax(iou)

            annotations.append(masks[max_iou_index].cpu().numpy())

    return fast_process(
        annotations=np.array(annotations),
        image=image,
        scale=(1024 // imgsz),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        with_contours=with_contours,
    )


def fast_process(
    annotations,
    image,
    scale,
    better_quality=False,
    mask_random_color=True,
    bbox=None,
    use_retina=True,
    with_contours=True,
) -> Image.Image:
    original_h = image.height
    original_w = image.width

    if better_quality:
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )
            annotations[i] = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)
            )

    inner_mask = fast_show_mask(
        annotations,
        plt.gca(),
        random_color=mask_random_color,
        bbox=bbox,
        retinamask=use_retina,
        target_height=original_h,
        target_width=original_w,
    )

    if with_contours:
        contour_all = []
        temp = np.zeros((original_h, original_w, 1))
        for i, mask in enumerate(annotations):
            annotation = mask.astype(np.uint8)
            if not use_retina:
                annotation = cv2.resize(
                    annotation,
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            contours, _ = cv2.findContours(
                annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                contour_all.append(contour)
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2 // scale)
        color = np.array([0 / 255, 0 / 255, 255 / 255, 0.9])
        contour_mask = temp / 255 * color.reshape(1, 1, -1)

    image = image.convert("RGBA")
    overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), "RGBA")
    image.paste(overlay_inner, (0, 0), overlay_inner)

    if with_contours:
        overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), "RGBA")
        image.paste(overlay_contour, (0, 0), overlay_contour)

    return image


# CPU post process
def fast_show_mask(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    #
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::1]
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    if random_color:
        color = np.random.random((mask_sum, 1, 1, 3))
    else:
        color = np.ones((mask_sum, 1, 1, 3)) * np.array(
            [30 / 255, 144 / 255, 255 / 255]
        )
    transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    mask = np.zeros((height, weight, 4))

    h_indices, w_indices = np.meshgrid(
        np.arange(height), np.arange(weight), indexing="ij"
    )
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    mask[h_indices, w_indices, :] = mask_image[indices]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="b", linewidth=1
            )
        )

    if not retinamask:
        mask = cv2.resize(
            mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )

    return mask
