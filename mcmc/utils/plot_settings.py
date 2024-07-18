"""Helper functions with colors for plotting figures."""
# functions taken from Kerry Halupka
# https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72

from pathlib import Path

import matplotlib as mpl
import numpy as np

current_dir = Path(__file__).parent


def hex_to_rgb(value: str) -> list[float]:
    """Converts hex to rgb colors.

    Args:
        value (str): string of 6 characters representing a hex colour.

    Returns:
        list: length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value: list[float]) -> list[float]:
    """Converts rgb to decimal colors (i.e. divides each value by 256).

    Args:
        value (list[float]): string of 6 characters representing a hex colour.

    Returns:
        list: length 3 of RGB values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(
    hex_list: list[str], float_list: list[float] | None = None
) -> mpl.colors.LinearSegmentedColormap:
    """Creates a color map that can be used in heat map figures. If float_list is not provided,
    color map graduates linearly between each color in hex_list. If float_list is provided,
    each color in hex_list is mapped to the respective location in float_list.

    Args:
        hex_list (list[str]): list of hex code strings
        float_list (list[float]): list of floats between 0 and 1, same length as hex_list. Must
            start with 0 and end with 1.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: continuous
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))
        ]
        cdict[col] = col_list
    return mpl.colors.LinearSegmentedColormap("j_cmap", segmentdata=cdict, N=256)


# Colors taken from Johannes Dietschreit's script and interpolated with correct lightness and Bezier
# http://www.vis4.net/palettes/#/100|s|fce1a4,fabf7b,f08f6e,d12959,6e005f|ffffe0,ff005e,93003a|1|1
with open(current_dir / "data/colors.txt", "r", encoding="utf-8") as f:
    std_hex_list = f.read().strip().split("\n")
cmap = get_continuous_cmap(std_hex_list)
colors = list(reversed(["#fce1a4", "#fabf7b", "#f08f6e", "#d12959", "#6e005f"]))
