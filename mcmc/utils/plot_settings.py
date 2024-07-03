"""Helper functions with colors for plotting figures."""
# functions taken from Kerry Halupka
# https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72

import matplotlib as mpl
import numpy as np


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


# colors taken from Johannes Dietschreit's script and interpolated with correct lightness and Bezier
# http://www.vis4.net/palettes/#/100|s|fce1a4,fabf7b,f08f6e,d12959,6e005f|ffffe0,ff005e,93003a|1|1
std_hex_list = [
    "#fce1a4",
    "#fcdea1",
    "#fcdc9e",
    "#fcda9b",
    "#fcd799",
    "#fcd496",
    "#fbd294",
    "#fbcf91",
    "#fbcd8f",
    "#fbca8d",
    "#fbc88b",
    "#fac589",
    "#fac387",
    "#fac085",
    "#f9be83",
    "#f9bb82",
    "#f8b980",
    "#f8b67e",
    "#f8b47d",
    "#f7b17b",
    "#f7ae7a",
    "#f6ac79",
    "#f6a977",
    "#f5a776",
    "#f5a475",
    "#f4a274",
    "#f49f73",
    "#f39c72",
    "#f29a71",
    "#f29770",
    "#f19470",
    "#f1926f",
    "#f08f6e",
    "#ef8d6d",
    "#ee8a6d",
    "#ed876c",
    "#ec856c",
    "#eb826b",
    "#ea806b",
    "#e97d6a",
    "#e87b6a",
    "#e77869",
    "#e67669",
    "#e57368",
    "#e37168",
    "#e26f67",
    "#e16c67",
    "#df6a66",
    "#de6766",
    "#dd6566",
    "#db6365",
    "#da6065",
    "#d85e64",
    "#d65c64",
    "#d55964",
    "#d35763",
    "#d25563",
    "#d05263",
    "#ce5063",
    "#cc4e62",
    "#cb4c62",
    "#c94962",
    "#c74761",
    "#c54561",
    "#c34361",
    "#c14061",
    "#bf3e61",
    "#bd3c60",
    "#bb3a60",
    "#b93860",
    "#b73660",
    "#b53360",
    "#b33160",
    "#b12f5f",
    "#af2d5f",
    "#ac2b5f",
    "#aa295f",
    "#a8275f",
    "#a5255f",
    "#a3235f",
    "#a1215f",
    "#9e1f5f",
    "#9c1d5f",
    "#9a1b5f",
    "#97195e",
    "#95175e",
    "#92155e",
    "#8f135e",
    "#8d115e",
    "#8a0f5e",
    "#880d5e",
    "#850b5e",
    "#82095e",
    "#7f075f",
    "#7d055f",
    "#7a045f",
    "#77035f",
    "#74025f",
    "#71015f",
    "#6e005f",
]
cmap = get_continuous_cmap(std_hex_list)
colors = list(reversed(["#fce1a4", "#fabf7b", "#f08f6e", "#d12959", "#6e005f"]))
