from json import load
from sys import maxsize
from os.path import exists
from pprint import pprint
from numpy import full, hstack, zeros
from utils.visualization import show_image
import cv2


def view_aligned_bbxes(mapping_path: str, image_path: str):
    """
    Shows mapping (bounding box: trascription) on image
    :param mapping_path: mapping between bounding boxes and associated textual content

            "lineN": {
                "xN_yN_wN_hN": "salvete",
                ...
            }
            ...

    :param image_path: unix file path
    :return: None
    """
    with open(mapping_path, 'r') as f:
        mapping = load(f)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    p_x, p_y, p_w, _ = map(int, mapping_path.split('/')[-1].split('.')[0].split('_')[1:-1])

    # coordinates of the bounding boxes to extract from image + separators
    base_image = zeros((300, p_w), dtype='uint8')

    sorted_mapping = sorted(mapping.items(), key=lambda e: int(e[0]))

    # testing specific lines
    try:
        lines_to_test = list(map(int, input("choose lines to test ").strip().split(',')))
        print(lines_to_test)
    except ValueError as v:
        print(v)
        lines_to_test = []

    for line, bbx2tr in sorted_mapping:
        if int(line) in lines_to_test or not lines_to_test:
            bbxes2tr_sorted = sorted(bbx2tr.items(), key=lambda b: int(b[0].split('_')[0]))

            print('##### ', line)
            pprint(bbxes2tr_sorted)
            print('\n\n')

            for bbx, transcript in bbxes2tr_sorted:
                x, y, w, h = map(int, bbx.split('_'))
                from_Y, to_Y, from_X, to_X = p_y+y, p_y+y+h+1, p_x+x, p_x+x+w+1
                cv2.destroyAllWindows()
                show_image(image[from_Y:to_Y, from_X:to_X], name=transcript)


def view_rows_by_bbxes_coordinates(mapping_path, image_path):
    """
    shows each (selected) row cropped by extreme values of the bounding boxes of the same line
    :param mapping_path: mapping between bounding boxes and associated textual content

            "rowN": {
                "xN_yN_wN_hN": "salvete",
                ...
            }
            ...

    :param image_path: unix file path
    :return: None
    """
    with open(mapping_path, 'r') as f:
        mapping = load(f)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    p_x, p_y, p_w, _ = map(int, mapping_path.split('/')[-1].split('.')[0].split('_')[1:-1])

    # coordinates of the bounding boxes to extract from image + separators
    sorted_mapping = sorted(mapping.items(), key=lambda e: int(e[0]))

    # testing specific lines
    try:
        lines_to_test = list(map(int, input("choose lines to test ").strip().split(',')))
    except ValueError as v:
        # print(v)
        lines_to_test = []

    for line, bbx2tr in sorted_mapping:
        if int(line) in lines_to_test or not lines_to_test:
            bbxes2tr_int = list(map(
                lambda k: (list((map(int, k[0].split('_')))), k[1]), bbx2tr.items()
            ))

            transcription = " ".join([tr for coords, tr in sorted(bbxes2tr_int, key=lambda v: v[0][0])])

            min_X, max_X = maxsize, -1
            min_Y, max_Y = maxsize, -1

            for coords, _ in list(bbxes2tr_int):
                x, y, w, h = coords
                if x < min_X:
                    min_X = x
                if x+w > max_X:
                    max_X = x+w
                if y < min_Y:
                    min_Y = y
                if y+h > max_Y:
                    max_Y = y+h

            from_X = min_X + p_x
            to_X = max_X + p_x
            from_Y = min_Y + p_y
            to_Y = max_Y + p_y

            print('\n', transcription)
            cv2.destroyAllWindows()
            show_image(image[from_Y:to_Y, from_X:to_X], name="line {}".format(line))


if __name__ == '__main__':
    path_mapping = 'aligned/043v_590_215_1365_1843/mapping/043v_590_215_1365_1843_bbxs2transcription.json'
    path_image = 'page_images/40r-44v_160/043v.png'

    assert exists(path_mapping)
    assert exists(path_image)

    view_aligned_bbxes(path_mapping, path_image)
    # view_rows_by_bbxes_coordinates(path_mapping, path_image)
