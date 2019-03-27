import re
import json
import shutil
from pprint import pprint
from math import log
from segmentation import *
from alignment_errors import AlignmentException
from utils.ui_draw_spaces import draw_missing_spaces, get_drawn_spaces
from utils.visualization import highlight_spaces_line, show_image, show_image_ask_spaces
from numpy import vstack, union1d, intersect1d, setxor1d, argmin
from collections import defaultdict
from bisect import insort

TSC_DIR = '../gt_transcriptions'
IMG_DIR = '../page_images_crop/40r-44v_160/'
TSC_EXT = '.txt'
IMG_EXT = '.png'


def test_start_alignment():
    dst_dir = '../aligned'
    fnms = set(f.split('.')[0] for f in os.listdir(IMG_DIR))
    fnms = sorted(fnms - set(os.listdir(dst_dir)))

    for fnm in fnms:
        print('\n###### {} ######'.format(fnm))

        stop = input("STOP alignment? [y/n] ")
        if stop == "y":
            break
        else:
            # load page image
            page_img_og = cv2.imread(os.path.join(IMG_DIR, fnm + IMG_EXT), cv2.IMREAD_GRAYSCALE)
            # cut capital letters at margin
            page_img, left_margin = remove_left_margin(page_img_og)
            # segment page image into lines
            img_lines = page_to_lines_2(page_img)

            for index, (line, _) in enumerate(img_lines):
                show_image(line, name=str(index))


if __name__ == "__main__":
    test_start_alignment()