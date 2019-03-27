import numpy as np
from copy import deepcopy
from numpy.core.multiarray import ndarray
from cv2 import imshow, waitKey, destroyAllWindows, INTER_AREA, resize


def show_image(image: np.ndarray, name: str = "image", minimize=False):
    """
    Handy wrapper for the image display pipeline
    :param image: image to show
    :param name: name of the window
    :return: None
    """
    destroyAllWindows()
    if minimize:
        minimized = resize(image, None, fx=0.45, fy=0.45, interpolation=INTER_AREA)
        imshow(name, minimized)
    else:
        imshow(name, image)
    waitKey(0)
    destroyAllWindows()
    # waitKey(1)


def show_image_ask_spaces(image: np.ndarray, tot_spaces:int, name: str = "specify spaces",
                          message: str = "which spaces are correct? ('0, ..., n', ':' for all, '' none of them )    ") -> list:
    """
    Shows the image and ask the user to insert the correct spaces in <spaces>
    :param tot_spaces: total number of spaces
    :param image: image to show
    :param name: name of the window
    :param message: print this message while asking for the spaces
    :return: selected spaces
    """
    selection = []

    imshow(name, image)
    waitKey(0)

    error_str = "\nValid inputs:\n• 0,1,2,...,j,...,n" \
                "\n• :"\
                "\n• '' (empty string)\n"

    stop = False
    print('\n')

    while not stop:
        try:
            stop = True
            selection = input(message).strip()
            if selection == ':':
                selection = list(range(tot_spaces))
            elif selection.isnumeric() or len(selection) >= 1:
                selection = eval("[" + selection + "]")
                if any([s >= tot_spaces for s in selection]):
                    stop = False
                    print("Out of range!\nSelected = {}, allowed = {} ".format(selection, list(range(tot_spaces))))

        except SyntaxError:
            print(error_str)
    destroyAllWindows()

    return selection


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
