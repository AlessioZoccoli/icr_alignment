import numpy as np
from copy import deepcopy
from numpy.core.multiarray import ndarray
from cv2 import imshow, waitKey, destroyAllWindows


def show_image(image: np.ndarray, name: str = "image"):
    """
    handy wrapper for the image display pipeline
    :param image: image to show
    :param name: name of the window
    :return: None
    """
    imshow(name, image)
    waitKey(0)
    destroyAllWindows()


def highlight_spaces_line(image: np.ndarray, spaces: list, color: str = "red", random_lightness: bool = False) -> np.ndarray:
    """
    Displays <image> spaces (<spaces>) in <color>.
    White background and black text.

    Note: passing a 3d array as an image to cv2 results in a YMC encoded image by default.

    :param image: 1-D image
    :param spaces: list of spaces intervals at the center of the image
            eg. [(0, 7), (15, 15), (23, 25), ...]
            Means that in     image[center, 0:7], image[center, 15:15] there is space
    :param color: spaces are highlighted in this color
    :param random_lightness: randomize hue
    :return: image with highlighted spaces in red
    """
    # 1D
    assert len(image.shape) == 2
    # no side effects
    _image = deepcopy(image)

    # highlight the spaces
    for sp in spaces:
        selected = _image[:, sp[0]:sp[1]]
        selected[selected > 0] = 64 if not random_lightness else np.random.randint(20, 240)

    if color == "red":
        img_line_colors = np.stack([_image, _image, image], axis=2)
    elif color == "blue":
        img_line_colors = np.stack([image, _image, _image], axis=2)
    elif color == "green":  # green
        img_line_colors = np.stack([_image, image, _image], axis=2)
    elif color == "yellow":
        img_line_colors = np.stack([_image, image, image], axis=2)
    elif color == "magenta":
        img_line_colors = np.stack([image, _image, image], axis=2)
    else:  # "cyan":
        img_line_colors = np.stack([image, image, _image], axis=2)

    return img_line_colors
