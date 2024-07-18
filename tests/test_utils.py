"""Testing utilities."""

from collections.abc import Iterable

import numpy as np


def compare_dicts(d1: dict, d2: dict):
    """Compare the values of two dictionaries. Dictionaries are not nested.
    They can contain lists, numpy arrays, and scalars.

    Args:
        d1 (dict): The first dictionary.
        d2 (dict): The second dictionary.
    """
    for key, value in d1.items():
        if isinstance(value, dict):
            compare_dicts(value, d2[key])
        elif isinstance(value, str):  # test for string
            assert value == d2[key]
        elif isinstance(value, Iterable):  # test for float (int) list or numpy array
            assert np.allclose(value, d2[key])
        elif value is None:
            assert d2[key] is None
        else:
            assert value == d2[key]

    return True
