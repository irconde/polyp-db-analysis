"""Utility submodules for :mod:`polypdb`.

The original :mod:`polypdb.utils` module has been split into dedicated
submodules.  Selected functions are re-exported here for backwards
compatibility.
"""

from .image_ops import (
    adjust_bounding_box,
    crop_black_padding_band,
    crop_image_padding,
    draw_bounding_boxes,
    get_image_size,
    normalize_image,
    resize_and_maintain_aspect,
    resize_with_padding,
)
from .file_ops import move_images, move_images_parallel, validate_conversion
from .masking import (
    mask_corner_artifacts,
    crop_and_mask_corner_artifacts,
    merge_rects,
    preprocess_for_text,
    detect_text_boxes,
    detect_ui_square_row,
    detect_ui_squares,
    draw_mask_contours,
    mask_text_regions,
    create_text_mask,
    create_text_masks,
    estimate_background_color,
    is_color_similar,
    mask_full_viewport_by_expanded_box,
    mask_viewport_auto_color_safe,
)

__all__ = [
    # image_ops
    "adjust_bounding_box",
    "crop_black_padding_band",
    "crop_image_padding",
    "draw_bounding_boxes",
    "get_image_size",
    "normalize_image",
    "resize_and_maintain_aspect",
    "resize_with_padding",
    # file_ops
    "move_images",
    "move_images_parallel",
    "validate_conversion",
    # masking
    "mask_corner_artifacts",
    "crop_and_mask_corner_artifacts",
    "merge_rects",
    "preprocess_for_text",
    "detect_text_boxes",
    "detect_ui_square_row",
    "detect_ui_squares",
    "draw_mask_contours",
    "mask_text_regions",
    "create_text_mask",
    "create_text_masks",
    "estimate_background_color",
    "is_color_similar",
    "mask_full_viewport_by_expanded_box",
    "mask_viewport_auto_color_safe",
]

