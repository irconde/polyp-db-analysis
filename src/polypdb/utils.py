import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import tifffile
from typing import Optional, Tuple
import typing as t
try:  # Optional dependency
    import cv2  # type: ignore
    if not hasattr(cv2, "imread"):
        cv2 = None  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    cv2 = None  # type: ignore


logger = logging.getLogger(__name__)


def _save_image(img, path):
    """Save ``img`` to ``path`` with TIFF fallback using ``tifffile``."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tif", ".tiff"}:
        tifffile.imwrite(path, np.array(img))
    else:
        with open(path, "wb") as f:
            img.save(f)


def get_image_size(image_path):
    """Return the width and height of an image.

    The image is opened using :mod:`PIL.Image` instead of OpenCV.

    Args:
        image_path: Path to the image.

    Returns:
        Tuple ``(width, height)`` representing the size of the image.
    """
    with Image.open(image_path) as img:
        return img.width, img.height


def crop_image_padding(
    img_path,
    cropped_img_path,
    mask_path=None,
    cropped_mask_path=None,
    padding_color=(15, 15, 15),
    threshold=20,
):
    """Crop an image and optional mask, stripping away padding pixels.

    ``img_path`` is validated before attempting to open it.  Passing a non
    existing path (or an invalid file descriptor) previously resulted in a
    rather opaque ``OSError: [Errno 9] Bad file descriptor`` bubbling up from
    :func:`PIL.Image.open`.  By explicitly checking for path existence we can
    raise a clearer ``FileNotFoundError`` and avoid confusing descriptor
    errors.

    Parameters
    ----------
    img_path:
        Location of the input image to be cropped.
    cropped_img_path:
        Output location where the cropped image will be written.
    mask_path:
        Optional path to a segmentation mask corresponding to ``img_path``.
    cropped_mask_path:
        Destination for the cropped mask when ``mask_path`` is provided.
    padding_color:
        RGB colour considered as padding and removed from the image. Defaults
        to ``(15, 15, 15)``.
    threshold:
        Permissible channel deviation from ``padding_color`` to still be
        regarded as padding. Defaults to ``20``.
    """

    img_path = os.fspath(img_path)
    cropped_img_path = os.fspath(cropped_img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    with Image.open(img_path) as img:
        # Convert to RGB to consistently evaluate padding colour even when the
        # source image has an alpha channel or a single channel.  The original
        # ``img`` object is preserved so that the cropped output retains its
        # initial mode.
        rgb_img = img.convert("RGB")
        pixels = rgb_img.load()

        def is_close_to_color(pixel, color, threshold):
            return all(abs(int(p) - c) <= threshold for p, c in zip(pixel, color))

        # Determine the number of padding columns on each side
        left_padding = 0
        for x in range(rgb_img.width):
            if all(
                is_close_to_color(pixels[x, y], padding_color, threshold)
                for y in range(rgb_img.height)
            ):
                left_padding += 1
            else:
                break

        right_padding = 0
        for x in range(rgb_img.width - 1, -1, -1):
            if all(
                is_close_to_color(pixels[x, y], padding_color, threshold)
                for y in range(rgb_img.height)
            ):
                right_padding += 1
            else:
                break

        # Determine the number of padding rows on each side
        top_padding = 0
        for y in range(rgb_img.height):
            if all(
                is_close_to_color(pixels[x, y], padding_color, threshold)
                for x in range(rgb_img.width)
            ):
                top_padding += 1
            else:
                break

        bottom_padding = 0
        for y in range(rgb_img.height - 1, -1, -1):
            if all(
                is_close_to_color(pixels[x, y], padding_color, threshold)
                for x in range(rgb_img.width)
            ):
                bottom_padding += 1
            else:
                break

        if (
            left_padding + right_padding >= rgb_img.width
            or top_padding + bottom_padding >= rgb_img.height
        ):
            left, upper, right, lower = 0, 0, img.width, img.height
        else:
            left = left_padding
            upper = top_padding
            right = rgb_img.width - right_padding
            lower = rgb_img.height - bottom_padding

        # Crop the image to the bounding box to remove padding
        cropped_img = img.crop((left, upper, right, lower))
        _save_image(cropped_img, cropped_img_path)

    if mask_path is not None:
        mask_path = os.fspath(mask_path)
        cropped_mask_path = os.fspath(cropped_mask_path)
        if os.path.exists(mask_path):
            with Image.open(mask_path) as mask_img:
                # Crop the mask to the same bounding box
                cropped_mask_img = mask_img.crop((left, upper, right, lower))
                _save_image(cropped_mask_img, cropped_mask_path)


def resize_and_maintain_aspect(img_path, output_path, target_size=(854, 480), fill_color=(0, 0, 0)):
    """
    Resize an image to a target size maintaining the aspect ratio.
    If the image is smaller than the target, pad it with a specified color.
    :param img_path: Path to the input image.
    :param output_path: Path to the output image.
    :param target_size: Tuple (width, height) of the target size.
    :param fill_color: Tuple (R, G, B) of the color to fill the padded area.
    :return: None. The image is saved to the output path.
    """
    with Image.open(img_path) as img:
        # Maintain aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # If the image is smaller than the target, pad it
        if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
            new_img = Image.new("RGB", target_size, fill_color)
            new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                                (target_size[1] - img.size[1]) // 2))
        else:
            new_img = img

        new_img.save(output_path)


def resize_with_padding(img, target_size=(224, 224), fill_color=(0, 0, 0)):
    """Resize ``img`` to ``target_size`` maintaining aspect ratio by padding.

    :param img: ``PIL.Image`` to be resized.
    :param target_size: Desired output size ``(width, height)``.
    :param fill_color: RGB color tuple used for padding.
    :return: Resized and padded ``PIL.Image``.
    """
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, fill_color)
    new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                        (target_size[1] - img.size[1]) // 2))
    return new_img


def mask_corner_artifacts(
    img: Image.Image,
    width: int = 50,
    height: int = 50,
    auto_detect: bool = True,
    threshold: int = 10,
    max_colour_run: int = 25,
    dark_thresh: int = 40,
    max_auto_fraction: float = 0.15,
    fill_mode: str = "base",
):
    """Mask camera overlays or logos in the four image corners.

    This implementation mirrors the standalone script that reliably removes the
    triangular corner pads used in colonoscopy frames.  Auto-detection is guided
    both by colour similarity and by a grayscale ``dark`` mask so that the mask
    does not extend into the valid field of view.  Short runs of non-matching
    pixels are tolerated which ensures small artefacts touching the image edge
    are still covered.

    Parameters
    ----------
    img:
        ``PIL.Image`` to be processed. The image is not modified in place.
    width, height:
        Fallback dimensions for the corner overlays when ``auto_detect`` is
        ``False``.
    auto_detect:
        When ``True`` (default), the size of the overlay in each corner is
        determined automatically by scanning outward from the corner until
        pixel values diverge from the base corner colour by ``threshold`` and
        remain darker than ``dark_thresh``.  If the diagonally opposite corner of
        the measured region has the same colour, a rectangular overlay is
        assumed; otherwise a triangular region is masked.
    threshold:
        Maximum per-channel colour distance to still be considered part of the
        overlay during auto detection.
    max_colour_run:
        Allow short sequences of differing pixels when measuring the overlay
        size.  This helps cover logos or indicators touching the image edges.
    dark_thresh:
        Grayscale value below which a pixel is considered part of the dark
        padding.  Defaults to ``40``.
    max_auto_fraction:
        Maximum fraction of the image dimension that auto detection may mask.
        This prevents large dark regions from being fully covered.  Defaults to
        ``0.15``.
    fill_mode:
        Fill colour for the masked region. ``"base"`` (default) reuses the
        corner colour.  A fixed RGB colour can be provided via
        ``"rgb:R,G,B"``.

    Returns
    -------
    ``PIL.Image``
        Image with corner artifacts filled.
    """

    img = img.convert("RGB")
    pixels = img.load()
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    gray = cv2.cvtColor(np.array(img)[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    dark = gray < dark_thresh

    fixed_fill: Optional[Tuple[int, int, int]] = None
    if fill_mode.startswith("rgb:"):
        try:
            fixed_fill = tuple(
                int(c) for c in fill_mode.split("rgb:")[1].split(",")
            )
            if len(fixed_fill) != 3:
                fixed_fill = None
        except Exception:
            fixed_fill = None

    def _differs(p1, p2):
        return any(abs(int(a) - int(b)) > threshold for a, b in zip(p1[:3], p2[:3]))

    def _measure(x0: int, y0: int, x_step: int, y_step: int, max_w: int, max_h: int):
        base = pixels[x0, y0]
        # horizontal run along row y0
        run_w = 0
        x = x0
        miss = 0
        while 0 <= x < img_w and run_w < max_w:
            if _differs(pixels[x, y0], base) or not dark[y0, x]:
                miss += 1
                if miss > max_colour_run:
                    break
            else:
                miss = 0
            run_w += 1
            x += x_step
        if miss:
            run_w -= miss
            if miss > max_colour_run:
                run_w += 1
        run_w = max(0, run_w)

        # vertical run along column x0
        run_h = 0
        y = y0
        miss = 0
        while 0 <= y < img_h and run_h < max_h:
            if _differs(pixels[x0, y], base) or not dark[y, x0]:
                miss += 1
                if miss > max_colour_run:
                    break
            else:
                miss = 0
            run_h += 1
            y += y_step
        if miss:
            run_h -= miss
            if miss > max_colour_run:
                run_h += 1
        run_h = max(0, run_h)
        return run_w, run_h, base

    corners = [
        ("tl", 0, 0, 1, 1, lambda W, H: [(0, 0), (W, 0), (0, H)]),
        (
            "tr",
            img_w - 1,
            0,
            -1,
            1,
            lambda W, H: [(img_w, 0), (img_w - W, 0), (img_w, H)],
        ),
        (
            "bl",
            0,
            img_h - 1,
            1,
            -1,
            lambda W, H: [(0, img_h), (W, img_h), (0, img_h - H)],
        ),
        (
            "br",
            img_w - 1,
            img_h - 1,
            -1,
            -1,
            lambda W, H: [
                (img_w, img_h),
                (img_w - W, img_h),
                (img_w, img_h - H),
            ],
        ),
    ]

    for name, x0, y0, x_step, y_step, poly_fn in corners:
        if auto_detect:
            w, h, base = _measure(x0, y0, x_step, y_step, img_w, img_h)
            max_w = int(img_w * max_auto_fraction)
            max_h = int(img_h * max_auto_fraction)
            w = min(w, max_w)
            h = min(h, max_h)
        else:
            w, h, base = width, height, pixels[x0, y0]
        if w <= 0 or h <= 0:
            continue

        x1 = x0 + x_step * (w - 1)
        y1 = y0 + y_step * (h - 1)
        color = tuple(
            int(c) for c in (fixed_fill if fixed_fill is not None else base[:3])
        )

        if (
            0 <= x1 < img_w
            and 0 <= y1 < img_h
            and not _differs(pixels[x1, y1], base)
            and dark[y1, x1]
        ):
            if name == "tl":
                rect = [0, 0, w - 1, h - 1]
            elif name == "tr":
                rect = [img_w - w, 0, img_w - 1, h - 1]
            elif name == "bl":
                rect = [0, img_h - h, w - 1, img_h - 1]
            else:  # br
                rect = [img_w - w, img_h - h, img_w - 1, img_h - 1]
            draw.rectangle(rect, fill=color)
        else:
            draw.polygon(poly_fn(w, h), fill=color)

    return img


def crop_and_mask_corner_artifacts(
    src_dir,
    dest_dir,
    padding_color=(15, 15, 15),
    padding_threshold=20,
    threshold=10,
    dark_thresh=40,
    max_auto_fraction=0.15,
):
    """Mask corner overlays then crop padding using auto detection.

    Parameters
    ----------
    src_dir : str
        Directory with input images.
    dest_dir : str
        Directory where processed images will be saved.
    padding_color : tuple
        RGB colour regarded as padding and removed after masking. Defaults to
        ``(15, 15, 15)``.
    padding_threshold : int
        Permissible channel deviation for padding detection. Defaults to
        ``20``.
    threshold : int
        Threshold for auto-detecting overlay size in :func:`mask_corner_artifacts`.
    dark_thresh : int
        Gray threshold used to classify padding during corner masking.
    max_auto_fraction : float
        Maximum fraction of the image dimension that auto detection may mask
        when calling :func:`mask_corner_artifacts`.
    """

    src_dir = os.fspath(src_dir)
    dest_dir = os.fspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    for name in os.listdir(src_dir):
        if not name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        ):
            continue
        in_path = os.path.join(src_dir, name)
        out_path = os.path.join(dest_dir, name)
        with Image.open(in_path) as img:
            masked = mask_corner_artifacts(
                img,
                auto_detect=True,
                threshold=threshold,
                dark_thresh=dark_thresh,
                max_auto_fraction=max_auto_fraction,
            )
            _save_image(masked, out_path)
        crop_image_padding(
            out_path,
            out_path,
            padding_color=padding_color,
            threshold=padding_threshold,
        )


def normalize_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize ``img`` using per-channel ``mean`` and ``std``.

    The image is expected to be in RGB format.

    :param img: ``PIL.Image`` to normalize.
    :param mean: Tuple of channel means.
    :param std: Tuple of channel standard deviations.
    :return: ``numpy.ndarray`` containing the normalized image data.
    """
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)
    return (arr - mean_arr) / std_arr


def adjust_bounding_box(original_box, original_size, resized_size, padding):
    """
    Adjusts bounding box coordinates based on the resized image dimensions and padding.
    :param original_box: Tuple (x1, y1, x2, y2) of the original bounding box coordinates.
    :param original_size: Tuple (width, height) of the original image size.
    :param resized_size: Tuple (width, height) of the resized image dimensions.
    :param padding: Tuple (left_padding, top_padding) indicating the amount of padding added.
    :return: Tuple (x1, y1, x2, y2) of the adjusted bounding box coordinates.
    """

    scale_x = resized_size[0] / original_size[0]
    scale_y = resized_size[1] / original_size[1]

    # Scale the original bounding box coordinates
    x1, y1, x2, y2 = original_box
    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    # Adjust for padding
    x1_adjusted = x1_scaled + padding[0]
    y1_adjusted = y1_scaled + padding[1]
    x2_adjusted = x2_scaled + padding[0]
    y2_adjusted = y2_scaled + padding[1]

    return x1_adjusted, y1_adjusted, x2_adjusted, y2_adjusted


def draw_bounding_boxes(bounding_boxes, image_path, save_path):
    """Overlay ``bounding_boxes`` on ``image_path`` and save to ``save_path``.

    Parameters
    ----------
    bounding_boxes : str
        Comma-separated bounding box coordinates in ``x1,y1,x2,y2`` format.
    image_path : str
        Path to the input image.
    save_path : str
        Location where the annotated image will be written.
    """
    # Open the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        coordinates = [int(coord.strip())
                       for coord in bounding_boxes.split(',') if coord.strip()]
        draw.rectangle(coordinates, outline="red", width=2)

        # Save the image with bounding boxes
        img.save(save_path)


def move_images(source_dir, dest_dir, df, filename_col='FRAME'):
    """
    Move images from a source directory to a destination directory based on filenames in a DataFrame.

    Parameters:
    - source_dir: The source directory where images are currently stored.
    - dest_dir: The destination directory where images should be moved.
    - df: The DataFrame containing the filenames of the images to be moved.
    - filename_col: The name of the DataFrame column containing the image filenames. Defaults to 'FRAME'.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        frame_filename = row[filename_col]
        source_path = os.path.join(source_dir, frame_filename)
        destination_path = os.path.join(dest_dir, frame_filename)

        # Attempt to move the file
        try:
            shutil.move(source_path, destination_path)
        except IOError as e:
            logger.error("Could not move %s. Error: %s", frame_filename, e)


def move_images_parallel(
    source_dir,
    dest_dir,
    df,
    filename_col="FRAME",
    source_path_col=None,
    copy=False,
):
    """Move images concurrently using :class:`ThreadPoolExecutor`.

    Parameters
    ----------
    source_dir:
        Base directory where images are stored when ``source_path_col`` is
        ``None``.
    dest_dir:
        Directory where images will be placed.
    df:
        DataFrame containing image file information.
    filename_col:
        Column in ``df`` with the destination image filename.
    source_path_col:
        Optional column in ``df`` with full source paths. When provided,
        ``source_dir`` is ignored for that row.
    copy:
        If ``True``, copy images instead of moving them.
    """

    os.makedirs(dest_dir, exist_ok=True)

    cols = [filename_col]
    if source_path_col:
        cols.append(source_path_col)

    rows = df[cols].itertuples(index=False, name=None)

    total = len(df)
    action = "copy" if copy else "move"
    logger.info("%s %d images to %s", action.capitalize(), total, dest_dir)

    def _transfer(row):
        if source_path_col:
            frame_filename, source_path = row
        else:
            (frame_filename,) = row
            source_path = os.path.join(source_dir, frame_filename)
        destination_path = os.path.join(dest_dir, frame_filename)
        try:
            if copy:
                shutil.copy(source_path, destination_path)
            else:
                shutil.move(source_path, destination_path)
        except IOError as e:
            logger.error("Could not %s %s. Error: %s",
                         action, frame_filename, e)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(_transfer, rows), total=total,
             desc=f"{action.title()}ing images"))
    logger.info("Finished %s images", action)


def validate_conversion(dest_dir, df, filename_col='FRAME'):
    """Validate that moved images match entries in ``df``.

    The function counts the rows in ``df`` and compares them against the
    number of image files located in ``dest_dir/img``. If the counts do not
    match, detailed diagnostics about missing, unexpected or duplicated file
    names are logged and a :class:`ValueError` is raised.

    Parameters
    ----------
    dest_dir:
        Root directory of the converted dataset. An ``img`` sub-directory is
        expected to contain the image files.
    df:
        DataFrame describing the dataset.
    filename_col:
        Name of the column in ``df`` containing the image filenames.
    """

    img_dir = os.path.join(dest_dir, 'img')
    try:
        files = [f for f in os.listdir(
            img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    except FileNotFoundError:
        logger.error("Image directory %s does not exist", img_dir)
        raise

    df_count = len(df)
    file_count = len(files)

    duplicates = df[df[filename_col].duplicated()][filename_col].tolist()
    if duplicates:
        logger.error("Duplicated filenames in metadata: %s",
                     ", ".join(sorted(duplicates)))

    if df_count != file_count:
        df_filenames = set(df[filename_col].astype(str))
        file_set = set(files)
        missing = sorted(df_filenames - file_set)
        extra = sorted(file_set - df_filenames)
        if missing:
            logger.error("Missing images: %s", ", ".join(missing))
        if extra:
            logger.error("Unexpected images: %s", ", ".join(extra))
        error_msg = (
            f"{df_count} rows in annotations but found {file_count} images in {img_dir}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        "Validated conversion for %s: %d rows match %d images", dest_dir, df_count, file_count
    )


def merge_rects(rects, gap=18, y_overlap=0.45, pad=2, H=None, W=None):
    """Merge nearby rectangles into larger boxes.

    Parameters
    ----------
    rects:
        Iterable of ``(x0, y0, x1, y1)`` tuples.
    gap:
        Maximum horizontal gap in pixels to consider boxes adjacent.
    y_overlap:
        Minimum vertical overlap ratio for merging.
    pad:
        Padding added around merged boxes.
    H, W:
        Optional image height and width to clamp coordinates.
    """

    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    merged: t.List[t.Tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in rects:
        placed = False
        for i, (X0, Y0, X1, Y1) in enumerate(merged):
            ov = max(0, min(y1, Y1) - max(y0, Y0))
            if ov / max(1, min(y1 - y0, Y1 - Y0)) >= y_overlap and max(
                0, max(X0, x0) - min(X1, x1)
            ) <= gap:
                merged[i] = (min(X0, x0), min(Y0, y0),
                             max(X1, x1), max(Y1, y1))
                placed = True
                break
        if not placed:
            merged.append((x0, y0, x1, y1))

    out = []
    for x0, y0, x1, y1 in merged:
        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad
        if H is not None and W is not None:
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(W - 1, x1)
            y1 = min(H - 1, y1)
        out.append((x0, y0, x1, y1))
    return out


def preprocess_for_text(img: np.ndarray) -> np.ndarray:
    """Enhance ``img`` prior to text detection.

    When OpenCV is available the image is converted to grayscale, contrast is
    normalised and the result is inverted.  A binary image is produced via
    Otsu's thresholding and converted back to BGR so it can be fed to OCR
    routines expecting three channels.  If OpenCV is not available the input is
    returned unchanged.
    """

    if cv2 is None:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.equalizeHist(gray)
    norm = cv2.bitwise_not(norm)
    _, thresh = cv2.threshold(
        norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def detect_text_boxes(
    img,
    ocr,
    scales=(1.0, 1.5, 2.0),
    restrict_to_edges=False,
    suppress_glare=False,
    preprocess=False,
):
    """Detect and merge rectangular text regions for ``img`` using ``ocr``.

    ``img`` should be a BGR ``numpy.ndarray``.  Detection is optionally run at
    multiple scales and glare is suppressed in the detector input.  The result
    consists of merged rectangles and a union bounding box covering all text.
    """

    H0, W0 = img.shape[:2]
    best_area, best_rects = -1, []

    for s in scales:
        if s != 1.0:
            if cv2 is None:
                continue
            resized = cv2.resize(img, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_CUBIC)
        else:
            resized = img

        det_img = resized.copy()
        if cv2 is not None and suppress_glare:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            glare = ((hsv[..., 2] >= 242) & (
                hsv[..., 1] <= 60)).astype(np.uint8) * 255
            glare = cv2.dilate(
                glare, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1
            )
            det_img[glare > 0] = 0

        if preprocess:
            det_img = preprocess_for_text(det_img)

        res = ocr.ocr(det_img, cls=False)
        rects = []
        pages = (
            res
            if res and isinstance(res[0], list) and isinstance(res[0][0], (list, tuple))
            else [res]
        )
        for page in pages:
            if not page:
                continue
            for det in page:
                if not det or det[0] is None:
                    continue
                P = np.asarray(det[0], dtype=np.float32)
                x0, y0 = P.min(axis=0)
                x1, y1 = P.max(axis=0)
                w, h = x1 - x0, y1 - y0
                if w * h < 100 or h > 0.5 * resized.shape[0]:
                    continue
                if restrict_to_edges:
                    if not (
                        x0 < 0.45 * resized.shape[1]
                        or x1 > 0.55 * resized.shape[1]
                        or y0 < 0.25 * resized.shape[0]
                    ):
                        continue
                rects.append(
                    (int(x0 / s), int(y0 / s), int(x1 / s), int(y1 / s))
                )

        rects = merge_rects(rects, H=H0, W=W0)
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
        if area > best_area:
            best_area, best_rects = area, rects

    if best_rects:
        xmin = min(r[0] for r in best_rects)
        ymin = min(r[1] for r in best_rects)
        xmax = max(r[2] for r in best_rects)
        ymax = max(r[3] for r in best_rects)
        union = (xmin, ymin, xmax, ymax)
    else:
        union = None

    return best_rects, union


def detect_ui_square_row(
    bgr: np.ndarray,
    expected_count_range: t.Tuple[int, int] = (5, 9),
    search_band_ratio: float = 0.35,
    bright_pct: float = 96.0,
    min_area_ratio: float = 2e-5,
    max_area_ratio: float = 2e-3,
    fill_ratio_min: float = 0.5,
    square_ar_tol: float = 0.6,
    mean_int_min: int = 170,
    dilate_frac_of_box: float = 0.15,
) -> t.Optional[t.Dict[str, t.Any]]:
    """Detect the horizontal row of small UI squares typically on the left side.

    Returns
    -------
    dict | None
        When a row is detected a dictionary is returned with keys:

        ``mask``
            ``np.uint8`` mask where detected squares are dilated.
        ``row_boxes``
            List of ``(x, y, w, h)`` for each square.
        ``bbox``
            ``(x, y, w, h)`` bounding box enclosing the row.

        ``None`` is returned when no suitable row is found.
    """

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    thr_val = max(200, int(np.percentile(g, bright_pct)))
    _, bin_hi = cv2.threshold(g, thr_val, 255, cv2.THRESH_BINARY)

    left_w = int(search_band_ratio * w)
    left_band = bin_hi[:, :left_w]

    min_area = int((w * h) * min_area_ratio)
    max_area = int((w * h) * max_area_ratio)

    contours, _ = cv2.findContours(
        left_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares: t.List[t.Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < min_area or area > max_area:
            continue
        ar = ww / float(hh) if hh else 0.0
        if ar < square_ar_tol or ar > (1.0 / square_ar_tol):
            continue
        cnt_area = cv2.contourArea(c)
        if cnt_area / max(area, 1) < fill_ratio_min:
            continue
        mask = np.zeros_like(left_band, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_int = cv2.mean(g[:, :left_w], mask=mask)[0]
        if mean_int < mean_int_min:
            continue
        squares.append((x, y, ww, hh))

    if not squares:
        return None

    squares = sorted(squares, key=lambda b: (b[1], b[0]))

    rows: t.List[t.List[t.Tuple[int, int, int, int]]] = []
    for s in squares:
        placed = False
        for row in rows:
            yh = np.mean([r[3] for r in row])
            if abs(s[1] - np.mean([r[1] for r in row])) <= max(6, 0.8 * yh):
                row.append(s)
                placed = True
                break
        if not placed:
            rows.append([s])

    candidates = []
    max_gap_px = int(0.015 * w)
    for row in rows:
        row = sorted(row, key=lambda b: b[0])
        n = len(row)
        if expected_count_range[0] <= n <= expected_count_range[1]:
            xs = [b[0] for b in row]
            gaps = np.diff(xs)
            if len(gaps) == 0:
                continue
            mean_w = np.mean([b[2] for b in row])
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            if (
                std_gap <= max(5.0, 0.7 * mean_w)
                and mean_gap < 3 * mean_w
                and mean_gap < max_gap_px
            ):
                candidates.append(row)

    if candidates:
        row = sorted(
            min(candidates, key=lambda r: np.mean([b[0] for b in r])),
            key=lambda b: b[0],
        )
    else:
        row = sorted(
            sorted(rows, key=lambda r: (-len(r),
                   np.mean([b[0] for b in r])))[0],
            key=lambda b: b[0],
        )

    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, ww, hh) in row:
        cv2.rectangle(mask, (x, y), (x + ww, y + hh), 255, -1)

    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bbox: t.Tuple[int, int, int, int] = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    k = max(1, int(round(dilate_frac_of_box * np.mean([b[2] for b in row]))))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    return {"mask": mask, "row_boxes": row, "bbox": bbox}


# Backwards compatibility
_detect_ui_square_row = detect_ui_square_row


def detect_ui_squares(img: np.ndarray) -> t.Optional[np.ndarray]:
    """Detect a row of small white UI squares and return a mask.

    The implementation searches for a horizontal sequence of bright,
    square-ish ticks typically found on the left side of endoscopic frames.  A
    solid bounding-box mask covering the detected row is returned.  When
    OpenCV is unavailable or no squares are found ``None`` is returned.

    Parameters
    ----------
    img:
        BGR image array in which to search.

    Returns
    -------
    ``numpy.ndarray`` mask with ``255`` marking the UI square row or ``None``
    if not detected.
    """

    if cv2 is None:
        return None

    det = detect_ui_square_row(img)
    h, w = img.shape[:2]
    if det is None:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = det["bbox"]  # type: ignore
    cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
    return mask


def draw_mask_contours(
    img: np.ndarray,
    mask: np.ndarray,
    color: t.Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Return a copy of ``img`` with ``mask`` contours overlaid.

    Parameters
    ----------
    img:
        BGR image on which to draw.
    mask:
        Single channel mask whose non-zero regions will be outlined.
    color:
        BGR colour for the contour lines. Defaults to green.
    thickness:
        Contour line thickness in pixels. Defaults to ``1``.

    Returns
    -------
    ``numpy.ndarray``
        Copy of ``img`` with mask contours drawn for debugging.
    """
    if cv2 is None:  # pragma: no cover - handled gracefully
        raise ImportError("OpenCV is required for drawing mask contours")

    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def mask_text_regions(
    img: np.ndarray,
    rects: t.Sequence[t.Tuple[int, int, int, int]],
    *,
    expand: int = 5,
    left_clamp: int = 25,
    inpaint: bool = False,
    inpaint_model=None,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Return ``img`` with text rectangles removed and the binary mask.

    Parameters
    ----------
    img:
        BGR image array.
    rects:
        Sequence of ``(x0, y0, x1, y1)`` boxes to be masked.
    expand:
        Pixels by which each rectangle is grown in all directions.
    left_clamp:
        Maximum distance from the left edge in pixels at which boxes are
        stretched to the border.
    inpaint:
        When ``True`` detected regions are inpainted instead of filled with
        black.
    inpaint_model:
        Optional inpainting model.  When ``None`` and ``inpaint`` is ``True``
        ``simple-lama-inpainting`` is initialised on demand.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The processed image and a ``uint8`` mask where ``255`` denotes text
        pixels.
    """

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in rects:
        left = max(0, x0 - expand)
        top = max(0, y0 - expand)
        right = min(w, x1 + expand)
        bottom = min(h, y1 + expand)
        if x0 <= left_clamp:
            left = 0
        mask[top:bottom, left:right] = 255

    if inpaint and mask.any():
        pil_img = Image.fromarray(img[:, :, ::-1])
        pil_mask = Image.fromarray(mask)
        if inpaint_model is None:
            try:
                from simple_lama_inpainting import SimpleLama  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "simple-lama-inpainting is unavailable. Install with `pip install simple-lama-inpainting`"
                ) from exc
            inpaint_model = SimpleLama()
        inpainted = inpaint_model(pil_img, pil_mask)
        # ``np.array(...)[..., ::-1]`` creates a view with a negative stride
        # which OpenCV drawing functions (e.g. ``cv2.rectangle``) cannot handle
        # as they require C-contiguous arrays.  ``copy()`` ensures we return a
        # contiguous array in BGR order.
        result = np.array(inpainted)[:, :, ::-1].copy()
    else:
        result = img.copy()
        result[mask > 0] = 0

    return result, mask


def create_text_mask(
    img_path,
    mask_path,
    ocr=None,
    preprocess=True,
    expand=5,
    left_clamp=25,
    inpaint_path=None,
    inpaint_model=None,
    debug_path=None,
):
    """Create an expanded rectangular text mask for ``img_path`` using PaddleOCR.

    The DB detector from :mod:`paddleocr` is employed to localise text
    regions.  For each detected text polygon a minimal axis-aligned
    rectangle is drawn onto the mask.  When present, the left-side row of
    bright UI squares is also masked out.

    Parameters
    ----------
    img_path:
        Location of the input image.
    mask_path:
        Output path where the mask image will be saved.  The mask uses a
        single ``L`` channel with ``255`` indicating detected text.
    ocr:
        Optional pre-initialised :class:`paddleocr.PaddleOCR` instance.
        When ``None`` an instance configured for English detection only
        is created on the fly.
    preprocess:
        When ``True`` detection is run at multiple scales and a glare
        suppression step is applied before running the detector.  Disable for
        faster but potentially less robust detection.
    expand:
        Number of pixels to grow each detected text box in all directions.
    left_clamp:
        Maximum distance from the left edge in pixels at which boxes are
        stretched to the image border.
    debug_path:
        Optional path where a debug image with mask contours is saved.
    """

    if ocr is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PaddleOCR is unavailable. Install paddlepaddle with `pip install \"paddlepaddle<3\"`"  # noqa: E501
            ) from exc
        ocr = PaddleOCR(
            det=True,
            rec=False,
            use_angle_cls=False,
            show_log=False,
            lang="en",
            det_db_thresh=0.25,
            det_db_box_thresh=0.38,
            det_db_unclip_ratio=2.2,
        )

    if cv2 is not None:
        bgr = cv2.imread(os.fspath(img_path))
    else:
        with Image.open(img_path) as _img:
            bgr = np.array(_img.convert("RGB"))[:, :, ::-1]

    scales = (1.0, 1.5, 2.0) if preprocess else (1.0,)
    rects, _ = detect_text_boxes(
        bgr,
        ocr,
        scales=scales,
        restrict_to_edges=True,
        suppress_glare=preprocess,
        preprocess=preprocess,
    )

    ui_mask = detect_ui_squares(bgr)

    inpaint = inpaint_path is not None
    processed, text_arr = mask_text_regions(
        bgr,
        rects,
        expand=expand,
        left_clamp=left_clamp,
        inpaint=inpaint,
        inpaint_model=inpaint_model,
    )

    combined = np.maximum(
        text_arr, ui_mask) if ui_mask is not None else text_arr
    _save_image(Image.fromarray(combined), mask_path)

    if debug_path is not None:
        dbg = bgr.copy()
        if ui_mask is not None:
            dbg = draw_mask_contours(dbg, ui_mask, color=(0, 255, 0))
        dbg = draw_mask_contours(dbg, text_arr, color=(0, 0, 255))
        _save_image(Image.fromarray(dbg[:, :, ::-1]), debug_path)

    if inpaint:
        _save_image(Image.fromarray(processed[:, :, ::-1]), inpaint_path)


def create_text_masks(
    src_dir,
    dest_dir,
    ocr=None,
    preprocess=True,
    expand=5,
    left_clamp=25,
    inpaint_dir=None,
    inpaint_model=None,
    debug_dir=None,
):
    """Generate text masks for all images in ``src_dir``.

    Files with extensions commonly associated with images are processed and
    a corresponding ``*_mask.png`` file is written to ``dest_dir``.
    ``ocr`` is passed to :func:`create_text_mask` to avoid repeatedly
    initialising the detector when processing many images.
    ``preprocess`` controls whether the OpenCV pipeline is applied before
    detection for all images.
    expand:
        Number of pixels to grow each detected text box.
    left_clamp:
        Maximum distance from the left edge in pixels at which boxes are
        stretched to the border.
    debug_dir:
        Optional directory where debug images with drawn mask contours are saved.
    """

    src_dir = os.fspath(src_dir)
    dest_dir = os.fspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    if inpaint_dir is not None:
        inpaint_dir = os.fspath(inpaint_dir)
        os.makedirs(inpaint_dir, exist_ok=True)
        if inpaint_model is None:
            try:
                from simple_lama_inpainting import SimpleLama  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "simple-lama-inpainting is unavailable. Install with `pip install simple-lama-inpainting`"
                ) from exc
            inpaint_model = SimpleLama()
    if debug_dir is not None:
        debug_dir = os.fspath(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)

    if ocr is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PaddleOCR is unavailable. Install paddlepaddle with `pip install \"paddlepaddle<3\"`"  # noqa: E501
            ) from exc

        ocr = PaddleOCR(use_angle_cls=False, lang="en")

    for name in os.listdir(src_dir):
        if not name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        ):
            continue

        src_path = os.path.join(src_dir, name)
        base = os.path.splitext(name)[0]
        dest_path = os.path.join(dest_dir, f"{base}_mask.png")
        inpaint_path = (
            os.path.join(
                inpaint_dir, f"{base}.png") if inpaint_dir is not None else None
        )
        debug_path = (
            os.path.join(debug_dir, name) if debug_dir is not None else None
        )

        if cv2 is not None:
            bgr = cv2.imread(src_path)
        else:
            with Image.open(src_path) as _img:
                bgr = np.array(_img.convert("RGB"))[:, :, ::-1]

        scales = (1.0, 1.5, 2.0) if preprocess else (1.0,)
        rects, _ = detect_text_boxes(
            bgr,
            ocr,
            scales=scales,
            restrict_to_edges=True,
            suppress_glare=preprocess,
            preprocess=preprocess,
        )
        ui_mask = detect_ui_squares(bgr)
        inpaint = inpaint_path is not None
        processed, text_arr = mask_text_regions(
            bgr,
            rects,
            expand=expand,
            left_clamp=left_clamp,
            inpaint=inpaint,
            inpaint_model=inpaint_model,
        )
        combined = np.maximum(
            text_arr, ui_mask) if ui_mask is not None else text_arr
        _save_image(Image.fromarray(combined), dest_path)

        if debug_path is not None:
            dbg = bgr.copy()
            if ui_mask is not None:
                dbg = draw_mask_contours(dbg, ui_mask, color=(0, 255, 0))
            dbg = draw_mask_contours(dbg, text_arr, color=(0, 0, 255))
            _save_image(Image.fromarray(dbg[:, :, ::-1]), debug_path)

        if inpaint:
            _save_image(Image.fromarray(processed[:, :, ::-1]), inpaint_path)


def estimate_background_color(image: np.ndarray, region_frac: float = 0.25) -> Tuple[int, int, int]:
    """Estimate median colour of the bottom-left region of ``image``.

    Parameters
    ----------
    image:
        BGR image array.
    region_frac:
        Fraction of width and height defining the region extending from the
        bottom-left corner. Default is ``0.25`` which corresponds to the
        lower-left quadrant.

    Returns
    -------
    Tuple[int, int, int]
        Median BGR colour of the sampled region.
    """
    h, w = image.shape[:2]
    region = image[int(h * (1 - region_frac)):, : int(w * region_frac)]
    pixels = region.reshape(-1, 3)
    median_color = np.median(pixels, axis=0).astype(np.uint8)
    return tuple(int(c) for c in median_color)


def is_color_similar(color: t.Sequence[int], ref=(135, 162, 18), tol: int = 35) -> bool:
    """Return ``True`` if ``color`` is within ``tol`` of ``ref`` for all channels."""
    return all(abs(int(c) - int(r)) <= tol for c, r in zip(color, ref))


def mask_full_viewport_by_expanded_box(
    image: np.ndarray,
    color_ref: t.Sequence[int],
    color_tol: int = 30,
    min_area: int = 1000,
    margin: int = 5,
) -> np.ndarray:
    """Create mask covering a bottom-left viewport matching ``color_ref``.

    The mask is computed by thresholding ``image`` for colours close to
    ``color_ref`` and restricting detection to the bottom-left quadrant. Each
    detected region above ``min_area`` is expanded by ``margin`` pixels on all
    sides and filled in the returned mask.
    """
    if cv2 is None:  # pragma: no cover - handled gracefully
        raise ImportError("OpenCV is required for viewport masking")

    lower = np.array([max(0, c - color_tol)
                     for c in color_ref], dtype=np.uint8)
    upper = np.array([min(255, c + color_tol)
                     for c in color_ref], dtype=np.uint8)
    color_mask = cv2.inRange(image, lower, upper)

    h, w = image.shape[:2]
    quadrant_mask = np.zeros((h, w), dtype=np.uint8)
    quadrant_mask[int(h / 2):, : int(w / 2)] = 255
    combined = cv2.bitwise_and(color_mask, quadrant_mask)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(combined)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x = max(0, x - margin)
            y = max(0, y - margin)
            bw = min(w - x, bw + 2 * margin)
            bh = min(h - y, bh + 2 * margin)
            cv2.rectangle(final_mask, (x, y), (x + bw, y + bh),
                          255, thickness=cv2.FILLED)

    return final_mask


def mask_viewport_auto_color_safe(
    image: np.ndarray,
    color_tol: int = 30,
    min_area: int = 1000,
    margin: int = 5,
    ref_color: t.Sequence[int] = (135, 162, 18),
    ref_tol: int = 35,
) -> Tuple[np.ndarray, Tuple[int, int, int], bool]:
    """Mask bottom-left viewport when its background colour looks like ``ref_color``.

    Returns a tuple ``(masked_img, estimated_color, applied)`` where ``applied``
    indicates whether masking was performed.
    """
    color_ref = estimate_background_color(image)
    if not is_color_similar(color_ref, ref_color, tol=ref_tol):
        return image.copy(), color_ref, False

    mask = mask_full_viewport_by_expanded_box(
        image, color_ref, color_tol, min_area, margin)
    masked_img = image.copy()
    masked_img[mask == 255] = 0
    return masked_img, tuple(int(c) for c in color_ref), True
