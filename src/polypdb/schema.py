from __future__ import annotations

"""Data model for unified polyp metadata.

This module defines the :class:`LesionMetadata` dataclass used throughout the
project to represent a single frame of a polyp along with harmonised metadata
fields.  The schema is intended to be the common denominator across all source
datasets.  Dataset specific attributes can be stored in the ``extras``
attribute.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class LesionMetadata:
    """Metadata associated with a single polyp frame.

    Parameters
    ----------
    global_lesion_id:
        Collision proof identifier derived from ``dataset`` and ``lesion_id``.
    dataset:
        Name of the originating dataset (e.g. ``sun`` or ``polar``).
    lesion_id:
        Identifier as used in the original dataset.
    n_frames:
        Number of frames available for the lesion.
    frame:
        New unique filename for the image frame.
    original_frame:
        Original filename from the source dataset.
    image_paths:
        List of absolute or relative paths pointing to the frame on disk.
    bounding_boxes:
        List of bounding boxes for lesions present in the frame. Each box is
        represented as ``[x_min, y_min, x_max, y_max]`` in pixel coordinates.
    morphology_shape:
        Normalised Paris classification or descriptive shape.
    morphology_size:
        Lesion size in millimetres or categorical descriptor.
    image_type:
        Imaging modality used to capture the frame (e.g. ``wl`` or ``nbi``).
    morphology_location:
        Anatomical location of the lesion if available.
    extras:
        Dictionary containing any dataset specific attributes that do not fit
        into the core schema.
    """

    global_lesion_id: str
    dataset: str
    lesion_id: str
    n_frames: int
    frame: str
    original_frame: str
    image_paths: List[str]
    bounding_boxes: List[List[int]] = field(default_factory=list)
    morphology_shape: Optional[str] = None
    morphology_size: Optional[Union[float, str]] = None
    image_type: Optional[str] = None
    morphology_location: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
