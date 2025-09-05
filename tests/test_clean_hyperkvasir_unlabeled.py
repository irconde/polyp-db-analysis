import sys
import types

sys.modules.setdefault("cv2", types.SimpleNamespace())

import numpy as np
from polypdb import clean_hyperkvasir_unlabeled as ch


def test_clean_frame_masks_ui_squares():
    img = np.full((40, 40, 3), 255, dtype=np.uint8)
    img[10:20, 10:20] = 100
    img[5:15, 7:13] = 150

    def fake_detect_text_boxes(_img, _ocr, **kwargs):
        return [(10, 10, 20, 20)], None

    def fake_detect_ui_squares(_img):
        mask = np.zeros((_img.shape[0], _img.shape[1]), dtype=np.uint8)
        mask[5:15, 7:13] = 255
        return mask

    def fake_viewport(img):
        return img, None, None

    orig_detect = ch.detect_text_boxes
    orig_ui = ch.detect_ui_squares
    orig_viewport = ch.mask_viewport_auto_color_safe
    try:
        ch.detect_text_boxes = fake_detect_text_boxes
        ch.detect_ui_squares = fake_detect_ui_squares
        ch.mask_viewport_auto_color_safe = fake_viewport
        cleaned = ch._clean_frame(img, ocr=None)
    finally:
        ch.detect_text_boxes = orig_detect
        ch.detect_ui_squares = orig_ui
        ch.mask_viewport_auto_color_safe = orig_viewport

    assert (cleaned[5:15, 5:15] == 0).all()
    assert (cleaned[0:10, 2:8] == 0).all()


def test_clean_frame_can_draw_ui_bbox():
    img = np.full((60, 60, 3), 255, dtype=np.uint8)
    img[20:30, 20:30] = 100

    def fake_detect_text_boxes(_img, _ocr, **kwargs):
        return [(20, 20, 30, 30)], None

    def fake_detect_ui_squares(_img):
        mask = np.zeros((_img.shape[0], _img.shape[1]), dtype=np.uint8)
        mask[15:25, 17:23] = 255
        return mask

    def fake_viewport(img):
        return img, None, None

    orig_detect = ch.detect_text_boxes
    orig_ui = ch.detect_ui_squares
    orig_viewport = ch.mask_viewport_auto_color_safe
    try:
        ch.detect_text_boxes = fake_detect_text_boxes
        ch.detect_ui_squares = fake_detect_ui_squares
        ch.mask_viewport_auto_color_safe = fake_viewport
        cleaned = ch._clean_frame(img, ocr=None, draw_ui_bbox=True)
    finally:
        ch.detect_text_boxes = orig_detect
        ch.detect_ui_squares = orig_ui
        ch.mask_viewport_auto_color_safe = orig_viewport

    green = np.array([0, 255, 0], dtype=np.uint8)
    assert (cleaned[10, 12:18] == green).all()
    assert (cleaned[10:20, 12] == green).all()


def test_clean_frame_removes_br_padding_glyph():
    img = np.full((20, 30, 3), 200, dtype=np.uint8)
    img[:, 20:] = 0  # right padding
    img[15:20, 25:30] = 255  # glyph inside padding

    def fake_detect_text_boxes(_img, _ocr, **kwargs):
        return [], None

    def fake_detect_ui_squares(_img):
        return None

    def fake_viewport(img):
        return img, None, None

    orig_detect = ch.detect_text_boxes
    orig_ui = ch.detect_ui_squares
    orig_viewport = ch.mask_viewport_auto_color_safe
    try:
        ch.detect_text_boxes = fake_detect_text_boxes
        ch.detect_ui_squares = fake_detect_ui_squares
        ch.mask_viewport_auto_color_safe = fake_viewport
        cleaned = ch._clean_frame(img, ocr=None)
    finally:
        ch.detect_text_boxes = orig_detect
        ch.detect_ui_squares = orig_ui
        ch.mask_viewport_auto_color_safe = orig_viewport

    assert cleaned.shape[1] < 20
    assert cleaned.shape[0] < 20
    assert cleaned.max() <= 200
