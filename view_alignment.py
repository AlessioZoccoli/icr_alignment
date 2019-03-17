from json import load
from os.path import exists
from pprint import pprint

from numpy import full, hstack, zeros

from utils.visualization import show_image
from copy import deepcopy
import cv2


def view_alignment(mapping_path, image_path):
    with open(mapping_path, 'r') as f:
        mapping = load(f)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    p_x, p_y, p_w, _ = map(int, mapping_path.split('/')[-1].split('.')[0].split('_')[1:-1])

    # coordinates of the bounding boxes to extract from image + separators
    base_image = zeros((300, p_w), dtype='uint8')

    sorted_mapping = sorted(mapping.items(), key=lambda e: int(e[0]))

    for row, bbx2tr in sorted_mapping:
        bbxes2tr_sorted = sorted(bbx2tr.items(), key=lambda b: int(b[0].split('_')[0]))

        print('##### ', row)
        pprint(bbxes2tr_sorted)
        print('\n\n')

        for bbx, transcript in bbxes2tr_sorted:
            x, y, w, h = map(int, bbx.split('_'))
            from_Y, to_Y, from_X, to_X = p_y+y, p_y+y+h+1, p_x+x, p_x+x+w+1

            show_image(image[from_Y:to_Y, from_X:to_X], name=transcript)


if __name__ == '__main__':
    path_mapping = 'aligned/040v_599_271_1374_1804/mapping/040v_599_271_1374_1804_bbxs2transcription.json'
    path_image = 'page_images/40r-44v_160/040v.png'

    assert exists(path_mapping)
    assert exists(path_image)

    view_alignment(path_mapping, path_image)
