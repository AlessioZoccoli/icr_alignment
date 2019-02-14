import os
import cv2
import numpy as np
import json
import shutil
from segmentation import page_to_lines, line_to_words, remove_left_margin
from alignment_errors import AlignmentException
from collections import deque, OrderedDict

from text_processing import estimate_word_width_simple

TSC_DIR = 'gt_transcriptions'
IMG_DIR = 'page_images_crop'
TSC_EXT = '.txt'
IMG_EXT = '.png'

AVG_CHARSIZES = json.load(open('average_charsizes.json'))


def max_bbx(bbxs):
    x1 = min([x for x, _, _, _ in bbxs])
    y1 = min([y for _, y, _, _ in bbxs])
    x2 = max([x + w for x, _, w, _ in bbxs])
    y2 = max([y + h for _, y, _, h in bbxs])

    return x1, y1, x2 - x1, y2 - y1


if __name__ == '__main__':
    dst_dir = 'aligned'
    tsc_dir_name = 'transcriptions'
    img_dir_name = 'images'
    fnms = [f.split('.')[0] for f in os.listdir(IMG_DIR)]

    alignment = dict()

    for fnm in fnms:
        print(fnm)
        if os.path.isdir(os.path.join(dst_dir, fnm)):
            shutil.rmtree(os.path.join(dst_dir, fnm))

        tsc_dst_dir = os.path.join(dst_dir, fnm, tsc_dir_name)
        img_dst_dir = os.path.join(dst_dir, fnm, img_dir_name)
        os.makedirs(tsc_dst_dir)
        os.makedirs(img_dst_dir)

        # load page image
        page_img_og = cv2.imread(os.path.join(IMG_DIR, fnm + IMG_EXT), cv2.IMREAD_GRAYSCALE)
        # cut capital letters at margin
        page_img, left_margin = remove_left_margin(page_img_og)
        # segment page image into lines
        img_lines = page_to_lines(page_img)
        # load GT transcription
        with open(os.path.join(TSC_DIR, fnm.split('_')[0] + TSC_EXT), 'r') as tsc_file:
            tsc_lines = tsc_file.readlines()

        if len(img_lines) != len(tsc_lines):
            raise AlignmentException(
                "Line mismatch: {} lines segmented, but transcription has {} lines"
                    .format(len(img_lines), len(tsc_lines))
            )

        #  alignment begins
        split_th = 0.25  # the higher, the more we are conservative about splitting
        merge_th = 0.3
        alignment[fnm] = dict()

        c = 0

        print("STOP AT 5")

        for tsc_line, (img_line, top_y) in zip(tsc_lines[:], img_lines[:]):
            # word_imgs is a double-ended queue containing computed word segmentations.
            word_imgs = deque(line_to_words(img_line, top_y))

            # if c > 5:
            #     break
            # print(c)
            # c += 1

            for tsc_word in tsc_line.split():
                estimated_width = estimate_word_width_simple(tsc_word)
                try:
                    word_img, word_img_x, word_img_y = word_imgs.popleft()
                except IndexError:
                    print('Alignment for', tsc_word, 'failed: no word images left')
                    break

                if word_img.shape[1] >= estimated_width * (1 + split_th):
                    # word image is wider than expected:
                    # we split it into two sub-images, and align tsc to the leftmost.
                    black_px_per_column = np.count_nonzero(cv2.bitwise_not(word_img), axis=0)
                    interval = black_px_per_column[
                               int(estimated_width * (1 - split_th)):int(estimated_width * (1 + split_th))
                               ]
                    split_point = np.argmin(interval) + int(estimated_width * (1 - split_th))

                    word_img_cur = word_img[:, :split_point]
                    print('case split:', tsc_word)
                    alignment[fnm][(word_img_x + left_margin - 2,
                                    word_img_y,
                                    word_img_cur.shape[1],
                                    word_img_cur.shape[0])] = tsc_word

                    word_img_next = word_img[:, split_point:]
                    word_imgs.appendleft((word_img_next, word_img_x + split_point, word_img_y))

                elif word_img.shape[1] <= estimated_width * (1 - merge_th):
                    # word image is shorter than expected:
                    # we merge it with following word images.
                    to_combine = [(word_img_x, word_img_y, word_img.shape[1], word_img.shape[0])]
                    while max_bbx(to_combine)[2] <= estimated_width * (1 - merge_th):
                        try:
                            word_img, word_img_x, word_img_y = word_imgs.popleft()
                        except IndexError:
                            print('Alignment for', tsc_word, 'failed: no word images left')
                            break

                        to_combine.append(
                            (word_img_x, word_img_y, word_img.shape[1], word_img.shape[0])
                        )

                    w_x, w_y, w_w, w_h = max_bbx(to_combine)
                    print('case merge:', tsc_word)
                    alignment[fnm][(w_x + left_margin - 2, w_y, w_w, w_h)] = tsc_word

                else:
                    # align
                    print('case aligned:', tsc_word)
                    alignment[fnm][
                        (word_img_x + left_margin - 2, word_img_y, word_img.shape[1], word_img.shape[0])] = tsc_word

            print()

        for ((bb_x, bb_y, bb_w, bb_h), tsc) in alignment[fnm].items():
            cv2.imwrite(os.path.join(
                img_dst_dir,
                '{}_{}_{}_{}.png'.format(bb_x, bb_y, bb_w, bb_h)
            ),
                page_img_og[bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
            )
            with open(os.path.join(
                    tsc_dst_dir,
                    '{}_{}_{}_{}.txt'.format(bb_x, bb_y, bb_w, bb_h)), 'w') as f:
                f.write(tsc)
        for ((bb_x, bb_y, bb_w, bb_h), tsc) in alignment[fnm].items():
            cv2.rectangle(
                page_img_og,
                (bb_x, bb_y),
                (bb_x + bb_w, bb_y + bb_h),
                192,
                1
            )
            cv2.putText(
                page_img_og,
                tsc,
                (bb_x, bb_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                64,
                1,
                cv2.LINE_AA
            )
        cv2.imwrite(fnm + '_TEXT.png', page_img_og)

    bounding_boxes = OrderedDict(
        [(page_k,
          OrderedDict([('_'.join(map(str, bbx_k)), tsc_v)
                       for bbx_k, tsc_v in sorted(bbxs.items(), key=lambda x: (x[0][1], x[0][0]))]))
         for page_k, bbxs in alignment.items()]
    )

    with open('bounding_boxes_elena.json', 'w') as bb_f:
        bb_f.write(json.dumps(bounding_boxes, indent=2))