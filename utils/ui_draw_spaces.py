from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np

# drawing = False  # true if mouse is pressed
ix, iy = -1, -1
spaces = []


# def draw_missing_spaces(img: np.ndarray):
#     # mouse callback function
#     def draw_circle(event, x, y, flags, param):
#         global ix, iy, drawing, spaces
#         if event == cv2.EVENT_LBUTTONDOWN:
#             drawing = True
#             ix, iy = x, y
#         elif event == cv2.EVENT_MOUSEMOVE:
#             if drawing:
#                 cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
#         elif event == cv2.EVENT_LBUTTONUP:
#             drawing = False
#             cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
#             start_x, end_x = sorted((ix, x))
#             spaces.append((start_x, end_x))
#
#     cv2.namedWindow('insert')
#     cv2.setMouseCallback('insert', draw_circle)
#     while (1):
#         cv2.imshow('insert', img)
#         if cv2.waitKey(20) & 0xFF in (27, ord('q')):
#             cv2.destroyAllWindows()
#             break
#     print("exited GUI\n")


def draw_missing_spaces(img: np.ndarray):
    image = deepcopy(img)
    c = 0

    # mouse callback function
    def draw_line(event, x, y, flags, param):
        global ix, iy, spaces

        # if c % 200 == 0:
        #     print("spaces ", spaces)

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), -1)
            start_x, end_x = sorted((ix, x))
            spaces.append((start_x, end_x))

    # clears the screen
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    cv2.namedWindow('insert')
    cv2.setMouseCallback('insert', draw_line)

    while (1):
        cv2.imshow('insert', image)
        if cv2.waitKey(20) & 0xFF in (27, ord('q')):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break

    print("exited GUI\n")


def get_drawn_spaces(take_in_range=10) -> list:
    """
    returns the global variable
    :return take_in_range: group spaces where distance is <= this threshold
    :return: list of spaces with the format [(fromX_1, toX_1)
    """
    global spaces
    out_spaces = spaces
    spaces = []  # flush global var
    return out_spaces

# from utils.user_interface import draw_missing_spaces, get_drawn_spaces
# import numpy as np
# a = np.ones((600, 800))

# import cv2
# for i in range(3):
#     draw_missing_spaces(a)
#     print(i, get_drawn_spaces(), '\n')
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)