import pytest
from PIL import Image, ImageDraw

from polypdb.utils import mask_corner_artifacts


def _make_image(overlays):
    """Create an image with coloured corner overlays and artefacts inside."""
    size = (100, 100)
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size
    for corner, (ow, oh) in overlays.items():
        if corner == "tl":
            poly = [(0, 0), (ow, 0), (0, oh)]
            draw.polygon(poly, fill=(0, 0, 0))
            draw.rectangle([1, 1, max(1, ow // 2), max(1, oh // 2)], fill=(255, 0, 0))
        elif corner == "tr":
            poly = [(w, 0), (w - ow, 0), (w, oh)]
            draw.polygon(poly, fill=(0, 0, 0))
            draw.rectangle(
                [w - ow + 2, 1, w - 2, max(1, oh // 2)], fill=(255, 0, 0)
            )
        elif corner == "bl":
            poly = [(0, h), (ow, h), (0, h - oh)]
            draw.polygon(poly, fill=(0, 0, 0))
            draw.rectangle(
                [1, h - oh + 2, max(1, ow // 2), h - 2], fill=(255, 0, 0)
            )
        elif corner == "br":
            poly = [(w, h), (w - ow, h), (w, h - oh)]
            draw.polygon(poly, fill=(0, 0, 0))
            draw.rectangle(
                [w - ow + 2, h - oh + 2, w - 2, h - 2], fill=(255, 0, 0)
            )
    return img


def test_mask_corner_artifacts_auto_detect():
    overlays = {
        "tl": (10, 15),
        "tr": (20, 12),
        "bl": (6, 18),
        "br": (14, 9),
    }
    img = _make_image(overlays)
    masked = mask_corner_artifacts(img, auto_detect=True, threshold=1)
    pixels = masked.load()
    w, h = masked.size

    # Verify artefacts removed and outside area untouched
    inside_outside = {
        "tl": ((1, 1), (overlays["tl"][0], overlays["tl"][1])),
        "tr": ((w - 1, 1), (w - overlays["tr"][0] - 1, overlays["tr"][1])),
        "bl": ((1, h - 1), (overlays["bl"][0], h - overlays["bl"][1] - 1)),
        "br": ((w - 1, h - 1), (w - overlays["br"][0] - 1, h - overlays["br"][1] - 1)),
    }

    for corner, (inside, outside) in inside_outside.items():
        assert pixels[inside] == (0, 0, 0)
        assert pixels[outside] == (255, 255, 255)
