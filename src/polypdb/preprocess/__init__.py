"""Dataset-specific preprocessing utilities."""

from .piba import convert_to_standard_layout as convert_piba
from .polar import convert_to_standard_layout as convert_polar
from .sun import convert_to_standard_layout as convert_sun
from .piccolo import convert_to_standard_layout as convert_piccolo
from .polypgen import convert_to_standard_layout as convert_polypgen

CONVERTERS = {
    "piba": convert_piba,
    "polar": convert_polar,
    "sun": convert_sun,
    "piccolo": convert_piccolo,
    "polypgen": convert_polypgen,
}
"""Mapping of dataset identifiers to their conversion functions."""
