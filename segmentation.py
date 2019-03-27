from copy import deepcopy
from pprint import pprint

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.visualization import show_image


def count_zero(ar: np.ndarray, axis=-1) -> int:
    """
    inverse of np.count_nonzero
    :param axis: axis or tuple of axes along which to count zeros.
        Default is -1, meaning that zeros will be counted
        along a flattened version of ar
    :param ar: the array for which to count zeros.
    :return: number of zeros if axis=None else array of scalars representing the number of zeros on the axis
    """
    if axis < 0:
        numzeros = ar.size - np.count_nonzero(ar)
    else:
        numzeros = ar.shape[axis] - np.count_nonzero(ar, axis=axis)
    return numzeros


def count_white_in_interval(image: np.ndarray, interval_x: tuple=None, interval_y: tuple=None, axis=None) -> int:
    """
    counts number of pixel with value >= white_thrs in image in the slices interval_x (horizontal)
    and interval_y (vertical)
    :param image: input image
    :param interval_x: horizontal interval = from interval_x[0] to interval_x[1]
    :param interval_y: vertical interval = from interval_y[0] to interval_y[1]
    :param axis: axis or tuple of axes along which to count white_thrs pixels.
    :return: number of pixels of white_thrs value
    """
    dimensions = len(image.shape)
    y, dy = interval_y if interval_y else (0, image.shape[0])
    x, dx = interval_x if interval_x else (0, image.shape[1])
    _image = image[y:dy, x:dx, :] if dimensions == 3 else image[y:dy, x:dx]

    return np.count_nonzero(cv2.inRange(_image, 20, 255), axis=axis)


def count_white_ratio(image: np.ndarray, interval_x: tuple=None, interval_y: tuple=None, axis=None) -> int:
    start_base, end_base = interval_x
    start_h, end_h = interval_y if interval_y else (0, image.shape[0])

    assert start_base < end_base
    assert start_h < end_h

    area = (end_base - start_base) * (end_h - end_base)
    ratio = count_white_in_interval(image, interval_x, interval_y, axis) / area
    # print(-(end_base - start_base), -ratio)
    return ratio


def count_black_ratio(image: np.ndarray, interval_x: tuple=None, interval_y: tuple=None, axis=None) -> int:
    start_base, end_base = interval_x if interval_x else (0, image.shape[1])
    start_h, end_h = interval_y if interval_y else (0, image.shape[0])

    assert start_base < end_base
    assert start_h < end_h

    area = (end_base - start_base) * (end_h - start_h)
    ratio = np.count_nonzero(image[start_h: end_h, start_base: end_base] <= 10) / area  # (end_base - start_base)
    return ratio


def space_in_line(line_image: np.ndarray, strip=True, erode_dilate=False) -> (list, int):
    if erode_dilate:
        return spaces_in_line_manipulated(line_image, strip)
    return spaces_in_line_simple(line_image, strip)


# def insert_subspace(image: np.ndarray, bigger: tuple, smaller: tuple) -> tuple:
#     """
#     smaller is partially or totally inside bigger
#     :param image: input image
#     :param bigger: bigger space
#     :param smaller: smaller space
#     :return: bigger and smaller updated
#     """
#     bigger_new, smaller_new = bigger, smaller
#     if smaller[0] <= bigger[0]:
#         bigger_new = (np.argmax(np.count_nonzero(image[:, smaller[1]:]), axis=0))
#     return None

# def split_overlapping_spaces(image:np.ndarray, left: tuple, right: tuple) -> tuple:
#     """
#     shorter space is inside bigger
#     :param left: leftmost spaces (based on the starting x)
#     :param right: rightmost space
#     :return: updated spaces (a_new, b_new)
#     """
#     left_new, right_new = left, right
#
#     # right is inside left or they overlap on the right
#     if left[1] >= right[1]:
#         left_new = (left[0], np.argmax(np.count_nonzero(image[:, left[0]:right[0]]), axis=0)+left[0])
#     elif left[1] < right[1]:
#         left_new =


def spaces_in_line_manipulated(line_image: np.ndarray, strip=True, discount=10) -> (list, int):
    """
    Returns the coordinates of the spaces inside line_image as intervals (startIntrv, endIntrv) and the center
    row of line_image. Groups the nearest columns having similar values of white pixels.
    if strip the starting and ending space are omitted in the intervals but will be returned separately

    line_image is manipulated using dilatation and erosion.

    :param line_image: image of a single line.
    :param strip: starting and ending spaces are not taken
    :return: x-axis coordinates of the spaces and the center of the line
    """

    # np.mean(height of every bounding box)
    # mean_height = 21.681010469695515

    # dilatation - erosion to clean impurity
    # kernel = np.ones((2, 2), np.uint8)
    # dilation = cv2.dilate(line_image, kernel, iterations=1)
    # erosion = cv2.erode(dilation, kernel, iterations=4)

    # show_image(line_image, name="before")
    kernel_rect = np.ones((5, 1), np.uint8)
    kernel_sq = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(line_image, kernel_rect, iterations=1)
    erosion = cv2.erode(dilation, kernel_sq, iterations=5)
    # show_image(erosion, name="after")

    # return
    invert_erosion = cv2.bitwise_not(erosion)

    # center of the line
    midline = np.argmax(np.count_nonzero(invert_erosion, axis=1))
    del invert_erosion

    # erosion = line_image
    count_white_col = np.argwhere(np.count_nonzero(erosion[midline-10:midline+10, :], axis=0) >= discount).flatten().tolist()
    # count_white_col = np.argwhere(np.count_nonzero(
    #     erosion[midline-round(line_image.shape[0]*1/3):midline+round(line_image.shape[0]*1/3), :], axis=0)
    #                               >= line_image.shape[0]*2/3-6).flatten().tolist()

    spaces = group_consecutive_values(count_white_col, threshold=1)

    # first and last column indexes
    spaces_intervals = [(sp[0], sp[-1]) for ind, sp in enumerate(spaces) if sp[-1] - sp[0] >= 5]
    starts_at = spaces_intervals[0][1]
    ends_at = spaces_intervals[-1][0]

    # if strip the starting and ending space are omitted in the intervals but will be returned separately
    if strip:
        #
        output_spaces = (spaces_intervals[1:-1], (starts_at, ends_at))
    else:
        output_spaces = (spaces_intervals, None)

    return output_spaces, midline


def spaces_in_line_simple(line_image: np.ndarray, strip=True, discount=10) -> (list, int):
    """
    Returns the coordinates of the spaces inside line_image as intervals (startIntrv, endIntrv) and the center
    row of line_image. Groups the nearest columns having similar values of white pixels.
    if strip the starting and ending space are omitted in the intervals but will be returned separately.

    :param line_image: image of a single line.
    :param strip: starting and ending spaces are not taken
    :param discount: relaxes the constraint on the number of white pixels
    :return: x-axis coordinates of the spaces and the center of the line
    """

    invert_line = cv2.bitwise_not(deepcopy(line_image))
    # center of the line
    midline = np.argmax(np.count_nonzero(invert_line, axis=1))
    del invert_line

    # erosion = line_image
    count_white_col = np.argwhere(np.count_nonzero(line_image, axis=0) >= line_image.shape[0]-discount).flatten().tolist()
    assert count_white_col
    spaces = group_consecutive_values(count_white_col, threshold=1)

    # first and last column indexes
    spaces_intervals = [(sp[0], sp[-1]) for ind, sp in enumerate(spaces) if sp[-1] - sp[0] >= 5]
    starts_at = spaces_intervals[0][1]
    ends_at = spaces_intervals[-1][0]

    # show_image(line_image[:, spaces_intervals[-1][0]:spaces_intervals[-1][1]])

    # not much ink on the end but we don't want them as spaces
    if starts_at+1 == spaces_intervals[0][0]:
        spaces_intervals.pop(0)
    if ends_at-1 == spaces_intervals[-1][1]:
        spaces_intervals.pop(-1)

    # if strip the starting and ending space are omitted in the intervals but will be returned separately
    if strip:
        #
        output_spaces = (spaces_intervals[1:-1], (starts_at, ends_at))
    else:
        output_spaces = (spaces_intervals, None)

    return output_spaces, midline


def center_space(space_interval: tuple) -> float:
    """
    In: center_space( (20, 30) )
    >> 25.0

    :param space_interval: a tuple containing the start and the end of the space interval
    :return: the center of space_interval
    """
    start, end = space_interval
    assert end > start
    return (end - start) / 2 + start


def group_consecutive_values(values, threshold: int = 2) -> list:
    """
        raggruppa valori interi consecutivi crescenti in una lista (a distanza threshold).
        ad es.: [1,2,4,6] -> [[1,2],[4],[6]] con threshold = 1
                [1,2,4,6] -> [[1,2,4,6]] con threshold = 2
    """

    run = []
    result = [run]
    last = values[0]

    for v in values:
        if v - last <= threshold:
            run.append(v)
        else:
            run = [v]
            result.append(run)
        last = v
    return result


def group_nearest_spaces_2(space_intrv: list, threshold: int=7) -> list:
    """
    list of the spaces found in the image as [(start, end),...]
    :param space_intrv: list of spaces
    :param threshold: if distance between two spaces is less than this, merge
    :return: updated list of spaces
    """
    grouped = []
    for i in range(0, len(space_intrv) - 1):
        if abs(space_intrv[i + 1][0] - space_intrv[i][1]) <= threshold:
            grouped.append((space_intrv[i][0], space_intrv[i + 1][1]))
        else:
            grouped.append(space_intrv[i])
    grouped.append(space_intrv[i+1])
    return grouped


def group_nearest_spaces(space_intrv: list, threshold: int=7) -> list:
    """
    list of the spaces found in the image as [(start, end),...]
    :param space_intrv: list of spaces
    :param threshold: if distance between two spaces is less than this, merge
    :return: updated list of spaces
    """
    assert space_intrv

    grouped = []
    omit = []
    for i in range(0, len(space_intrv) - 1):
        if abs(space_intrv[i + 1][0] - space_intrv[i][1]) <= threshold and i not in omit:
            grouped.append((space_intrv[i][0], space_intrv[i + 1][1]))
            omit.append(i+1)
        elif abs(space_intrv[i + 1][0] - space_intrv[i][1]) <= threshold and i in omit and i > 0:
            # in this case space[i] is grouped with the previous/es one/s
            prev_index = i-1
            # multiple spaces grouped? Then iterate
            while prev_index in omit:
                prev_index -= 1
            grouped.append((space_intrv[prev_index][0], space_intrv[i + 1][1]))
            omit.append(i + 1)
        else:
            grouped.append(space_intrv[i])
    grouped.append(space_intrv[i+1])
    grouped = [grouped[sp_index] for sp_index in range(len(grouped)) if sp_index not in omit]

    return grouped


def intervals_to_ranges(intervals: list) -> list:
    """
    intervals_to_range([(0, 3), (7, 10)]) = [0, 1, 2, 3, 7, 8, 9, 10]
    :param intervals: list of every space represented by its boundaries
    :return: list of each column of the spaces
    """
    return [list(range(start, end+1)) for start, end in intervals]


def remove_trailing_spaces(spaces: list, from_x: int, to_x: int) -> list:
    """
    removes unwanted spaces (eg. caused by long and thin serifs)
    :param spaces: spaces in the image as list of tuples (start, end)
    :param from_x: remove spaces from here
    :param to_x: sto removing spaces here
    :return: updated spaces
    """
    return [s for s in spaces if from_x <= s[0] and s[1] <= to_x]


def remove_left_margin(page_img):
    black_px_per_column = np.count_nonzero(
        cv2.bitwise_not(page_img), axis=0
    )[:page_img.shape[1] // 2]

    left_margins = np.argwhere(
        black_px_per_column < np.average(black_px_per_column) * 0.5  # 0.5 updated for 40r-44v_130
    )

    # print(left_margins)
    # pprint(black_px_per_column < np.average(black_px_per_column) * 0.45)

    left_margin = int(
        max([min(g) for g in group_consecutive_values(left_margins)])
    )

    return page_img[:, left_margin:], left_margin


def remove_left_margin_2(page_img):
    first_quarter = page_img.shape[1] // 4
    third_quarter = 3 * first_quarter

    white_px_per_column = np.count_nonzero(
        page_img[:, first_quarter: third_quarter], axis=0
    )
    left_margins = np.argwhere(
        (np.count_nonzero(page_img[:, :first_quarter], axis=0) - np.mean(white_px_per_column)) > 150.0
    )
    consecutive_values = group_consecutive_values(left_margins)
    left_margin = int(
        max([min(g) for g in consecutive_values])
    )

    return page_img[:, left_margin:], left_margin


def remove_left_margin_fourier_manipulated(page_img: np.ndarray) -> tuple:
    """
    removes left column of space + big capital letters
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    :param page_img: input image
    :return: cropped image and left margin
    """

    first_quarter_X = page_img.shape[1] // 4
    third_quarter_X = 3 * first_quarter_X
    # first_half_Y = page_img.shape[0] // 2

    color_threshold = 24 # 230

    # FOURIER ANALYSIS
    f = np.fft.fft2(page_img)
    fshift = np.fft.fftshift(f)
    rows, cols = page_img.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    np.putmask(img_back, img_back < color_threshold, 0)

    kernel_dil = np.ones((4, 1), np.uint8)
    kernel_erosion = np.ones((1, 4), np.uint8)
    dilation = cv2.dilate(deepcopy(img_back), kernel_dil, iterations=5)
    erosion = cv2.erode(dilation, kernel_erosion, iterations=5)

    # CROPPING
    # we have text lines in range [:, first_quarter_X: third_quarter_X]
    white_px_per_column = erosion[:, :third_quarter_X] >= color_threshold
    mean_white_px = np.mean(np.count_nonzero(white_px_per_column[:, first_quarter_X:], axis=0))
    left_margins = np.argwhere(
        np.count_nonzero(white_px_per_column[:, :first_quarter_X], axis=0) <= mean_white_px * 0.59
    )
    left_margin = max(group_consecutive_values(left_margins), key=lambda x: len(x))[-1][0]

    return page_img[:, left_margin-17:], left_margin


def remove_left_margin_fourier(page_img: np.ndarray, r_or_v=None) -> tuple:
    """
    removes left column of space + big capital letters
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    :param page_img: input image
    :param r_or_v: part of the filename image, r = page tend to hang to the left, v = other way around
    :return: cropped image and left margin
    """
    # kernel = np.ones((2, 2), np.uint8)
    # dilation = cv2.dilate(deepcopy(page_img), kernel, iterations=1)
    # erosion = cv2.erode(dilation, kernel, iterations=4)

    first_quarter_X = page_img.shape[1] // 4
    third_quarter_X = 3 * first_quarter_X
    # first_half_Y = page_img.shape[0] // 2

    color_threshold = 210

    # FOURIER ANALYSIS
    f = np.fft.fft2(page_img)
    fshift = np.fft.fftshift(f)
    rows, cols = page_img.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    np.putmask(img_back, img_back < color_threshold, 0)

    # CROPPING
    # we have text lines in range [:, first_quarter_X: third_quarter_X]
    white_px_per_column = img_back[:, :third_quarter_X] >= color_threshold
    mean_white_px = np.mean(np.count_nonzero(white_px_per_column[:, first_quarter_X:], axis=0))
    left_margins = np.argwhere(
        np.count_nonzero(white_px_per_column[:, :first_quarter_X], axis=0) <= mean_white_px * 0.6
    )
    left_margin = max(group_consecutive_values(left_margins), key=lambda x: len(x))[-1][0]

    return page_img[:, left_margin-15:], left_margin


def page_to_lines(page_img: np.ndarray) -> np.ndarray:
    """
    IMPORTANT: use page_to_lines_2 for robustness against short text lines

    Splits each page into an array of rectangular images, each rectangle is a line of text

    :param page_img: input image in greyscale
    :return: list of lines of text as (image, y-start of text)
    """
    black_px_per_row = np.count_nonzero(cv2.bitwise_not(page_img), axis=1)
    threshold = np.average(black_px_per_row) * 1.1
    ixs = np.argwhere(black_px_per_row < threshold)
    ixs_grouped = group_consecutive_values(ixs, threshold=7)

    min_ixs = [int(np.average(ixs)) for ixs in ixs_grouped]
    lines = []
    # page_img[min_ixs,:] = 127
    # plt.imshow(page_img)
    # plt.show()

    for i in range(0, len(min_ixs) - 1):
        top_y = min_ixs[i]
        bottom_y = min_ixs[i + 1]
        line = page_img[top_y:bottom_y]
        lines.append((line, top_y))

    return np.array(lines)


def page_to_lines_2(page_img: np.ndarray) -> np.ndarray:
    """
    Splits each page into an array of rectangular images, each rectangle is a line of text.

    :param page_img: input image in greyscale
    :return: list of lines of text as (image, y-start of text)
    """
    black_px_per_row = np.count_nonzero(cv2.bitwise_not(page_img[:, :page_img.shape[1] // 2]), axis=1)
    threshold = np.average(black_px_per_row) * 1.1
    ixs = np.argwhere(black_px_per_row < threshold)
    ixs_grouped = group_consecutive_values(ixs, threshold=7)

    min_ixs = [int(np.average(ixs)) for ixs in ixs_grouped]
    lines = []

    for i in range(0, len(min_ixs) - 1):
        top_y = min_ixs[i]
        bottom_y = min_ixs[i + 1]
        line = page_img[top_y:bottom_y]
        lines.append((line, top_y))

    return np.array(lines)


def line_to_words(line, top_y):
    """
        suddivide una linea di testo in parole.
        input: la linea e la sua posizione y
        output: una lista di triple (parola, top_y, left_x)
    """
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(line, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=4)

    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(erosion))

    midline = np.argmax(np.count_nonzero(cv2.bitwise_not(erosion), axis=1))

    words = sorted(
        [(line[:, x:x+w], x, top_y)
            for x, y, w, h, _ in stats[1:] if (y <= midline <= y+h)],
        key=lambda x: x[1]
    )

    return words


def find_local_minima(a):
    """
        ritorna gli indici corrispondenti ai minimi locali in un'immagine
    """
    local_minima = []

    # un punto e' minimo locale se e' minore dei suoi punti adiacenti;
    # nello specifico, diciamo che deve essere minore strettamente almeno di uno dei due
    # (mentre puo' essere minore o uguale all'altro)

    local_minima.append(0)

    for i in range(1, len(a) - 1):

        is_local_minimum = (
                (a[i] <= a[i - 1] and a[i] < a[i + 1]) or
                (a[i] < a[i - 1] and a[i] <= a[i + 1])
        )

        if is_local_minimum:
            local_minima.append(i)

    local_minima.append(len(a) - 1)

    return np.array(local_minima)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(int(window_len / 2) - 1):-(int(window_len / 2))]


import os

# if __name__ == '__main__':
#     src = 'page_images_nicolas_crop'
#     dst = 'lines'
#     pages = os.listdir(src)
#
#     for page in pages:
#         page_img = cv2.imread(os.path.join(src, page), cv2.IMREAD_GRAYSCALE)
#
#         lines = page_to_lines(page_img)
#
#         for line, y in lines:
#             fnm = '{}_{}_{}_{}.png'.format(page.split('.')[0], y, line.shape[1], line.shape[0])
#             cv2.imwrite(os.path.join(dst, fnm), line)
