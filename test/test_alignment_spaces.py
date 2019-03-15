import os
import cv2
import re
import json
import shutil
from pprint import pprint
from math import log
from segmentation import *
from alignment_errors import AlignmentException
from utils.visualization import highlight_spaces_line, show_image
from numpy import vstack, union1d, intersect1d, setxor1d, argmin
from bisect import insort

TSC_DIR = '../gt_transcriptions'
IMG_DIR = '../page_images_crop/40r-44v_130/'
TSC_EXT = '.txt'
IMG_EXT = '.png'
AVG_CHARSIZES = json.load(open('../average_charsizes_extended.json'))


def estimate_word_width(tsc_word: str, default_char_width=20, all_lowercase=False, previous_word="",
                        show_clean_word=False, charsizes=AVG_CHARSIZES):
    """
    Returns an estimate for the length of tsc_word by summing average size of each glyph in the same string
    :param tsc_word: str. The string for which to estimate the width
    :param all_lowercase: Case insensitive, convert all to lower case.
    :param default_char_width: if not in dictionary
    :param show_clean_word: print the word after the substitutions
    :param previous_word: the word before tsc_word
    :return:
    """

    tsc_wrd = tsc_word.lower() if all_lowercase else tsc_word

    # letters may be transcribed lower case but written upper case if preceded by a full stop
    mult_first = 1
    try:
        if previous_word[-1] == '.':
            mult_first = 1.3
    except IndexError:
        pass

    word = re.sub(r"\(.*?\)|\'.*?\'", "", tsc_wrd
                  .replace('(us)', '4')
                  .replace('(ue)', ';')
                  .replace('(et)', '7')
                  .replace('(rum)', '2')) \
        .replace('-', '') \
        .replace(',', '') \
        .replace('v', 'u')

    if show_clean_word:
        print("{} -> {}".format(tsc_word, word))

    char_width = lambda ch_row_ind: AVG_CHARSIZES[ch_row_ind[0]][0] * (1 if not ch_row_ind[1] == 0 else mult_first)

    est_w = sum([
        char_width((c, i)) if c in AVG_CHARSIZES.keys() else default_char_width
        for i, c in enumerate(list(word))
    ])
    return int(est_w)


if __name__ == '__main__':
    dst_dir = '../aligned'
    tsc_dir_name = '../transcriptions'
    img_dir_name = '../images'
    fnms = [f.split('.')[0] for f in os.listdir(IMG_DIR)]

    #
    #   alignment = {
    #                   filename1: {
    #                                   startX, startY, width, height
    #                              }
    #                   ...
    #               }
    #
    alignment = dict()

    for fnm in fnms:
        print(fnm)

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
                "Line mismatch: {} lines segmented, but transcription has {} lines".format(len(img_lines),
                                                                                           len(tsc_lines))
            )

        #  alignment begins
        split_th = 0.25  # the higher, the more we are conservative about splitting
        merge_th = 0.3
        alignment[fnm] = dict()

        # for each line: transcription of line L, image of line L
        transcr2img = zip(tsc_lines[:], img_lines[:])

        for ind, (tsc_line, (img_line, top_y)) in enumerate(transcr2img):
                #
                #    original: no dilatation or erosion
                #
                # spaces, center = spaces_in_line(img_line)
                (spaces, (first, last)), center = spaces_in_line_simple(img_line)
                num_spaces = len(tsc_line.split()) - 1
                print(tsc_line.split(), '\n')

                other_spaces = min((len(spaces) - num_spaces) // 2, 4)
                biggest_spaces = group_nearest_spaces(sorted(spaces, key=lambda e: -e[1] + e[0]))[
                                 :num_spaces + other_spaces]
                image_spaces = highlight_spaces_line(img_line, biggest_spaces, "red")
                #
                #   manipulated: dilatation and erosion
                #
                (spaces_m, (first_m, last_m)), center_m = spaces_in_line_manipulated(img_line, discount=10)
                other_spaces_m = min((len(spaces_m) - num_spaces) // 2, 3)
                biggest_spaces_m = group_nearest_spaces(sorted(spaces_m, key=lambda e: -e[1] + e[0]))[
                                   :num_spaces + other_spaces_m]
                image_spaces_m = highlight_spaces_line(img_line, biggest_spaces_m, "green")
                spaces_m_flatten = [el for rng in intervals_to_ranges(biggest_spaces_m) for el in rng]
                spaces_orig_flatten = [el for rng in intervals_to_ranges(biggest_spaces) for el in rng]

                #
                # start and end of the line
                #
                start_text = (0, max(first_m, first))
                end_text = (min(last_m, last), img_line.shape[1])

                #
                #   intersection
                #
                spaces_intersection = intersect1d(spaces_m_flatten, spaces_orig_flatten)
                spaces_intersect_intervals = [(sp[0], sp[-1]) for sp in
                                              group_consecutive_values(spaces_intersection, threshold=1)
                                              if sp[-1] - sp[0] > 5]
                # spaces_intersect_intervals = group_nearest_spaces(sorted(spaces_intersect_intervals))
                spaces_intersect_intervals_sorted = sorted(spaces_intersect_intervals,
                                                           key=lambda e: -log(e[1] - e[0]) - count_white_ratio(
                                                               image=img_line, interval_x=e))
                spaces_intersect_intervals_sorted = remove_trailing_spaces(spaces_intersect_intervals_sorted,
                                                                           from_x=start_text[1] + 25,
                                                                           to_x=end_text[0] - 25)

                othersp = min((len(spaces_intersect_intervals_sorted) - num_spaces) // 2, 3)
                spaces_intersect_intervals = sorted(spaces_intersect_intervals_sorted[:num_spaces + othersp])

                show_image(vstack((
                    image_spaces_m,
                    image_spaces,
                    highlight_spaces_line(img_line, spaces_intersect_intervals, "blue")
                )))

                #
                #   difference
                #   considering the interval as a hole not just set of consecutive pixels, so if part of an interval
                #   is in difference then it is discarded
                #
                spaces_difference = setxor1d(spaces_m_flatten, spaces_orig_flatten)
                spaces_diff_intervals = [(sp[0], sp[-1]) for ind, sp in enumerate(
                    group_consecutive_values(spaces_difference, threshold=1))
                                         if
                                         sp[0] - 1 not in spaces_orig_flatten and sp[0] - 1 not in spaces_m_flatten and
                                         sp[-1] + 1 not in spaces_orig_flatten and sp[-1] + 1 not in spaces_m_flatten
                                         ]
                spaces_diff_intervals.extend(spaces_intersect_intervals_sorted[num_spaces+othersp:])
                spaces_diff_intervals = sorted(spaces_diff_intervals)

                image_spaces_diff = highlight_spaces_line(img_line, spaces_diff_intervals, "yellow")

                # show_image(vstack((
                #      highlight_spaces_line(img_line, spaces_intersect_intervals_sorted, "green"),
                #      highlight_spaces_line(img_line, spaces_intersect_intervals, "red", random_lightness=True),
                #      image_spaces_diff
                # )))

                #
                #    transcriptions
                #
                # the biggest ones + starting and ending
                spaces_list = sorted([start_text] + spaces_intersect_intervals + [end_text])

                # each word of this line found in the transcription
                words = tsc_line.split()
                words_widths_estimate = []
                for ix, w in enumerate(words):
                    previous_w = "" if ix == 0 else words[ix - 1]
                    words_widths_estimate.append(estimate_word_width(w, previous_word=previous_w,
                                                                     charsizes=AVG_CHARSIZES))

                #
                # checking that intersection picked the right spaces
                #
                c = 0
                show_image(highlight_spaces_line(img_line, spaces_intersect_intervals, "red", random_lightness=True),
                           name="before while loop")
                while True:

                    word_diffs = []
                    has_anomalous_width = False
                    c += 1

                    # print(len(spaces_list)-1 == len(words))
                    for si in range(1, len(spaces_list)):
                        try:
                            left_word_estimate = words_widths_estimate[si - 1]
                        except IndexError:
                            # last word reached
                            pass

                        left_word_width = spaces_list[si][0] - spaces_list[si - 1][1]
                        left_word_diff = (left_word_estimate - left_word_width) / left_word_estimate
                        word_diffs.append(left_word_diff)
                        if abs(left_word_diff) >= 0.5:
                            has_anomalous_width = True
                        # print(si, "/", num_spaces, " •• ", left_word_diff, words[si-1])

                    # assert not has_anomalous_width
                    if not has_anomalous_width:
                        break

                    # each space in space_list is associated with the word at its RIGHT (except the last one)
                    words_widths_diffs = sorted(range(len(word_diffs)), key=lambda ind_w: word_diffs[ind_w])
                    widest = words_widths_diffs[0]
                    narrowest = words_widths_diffs[-1]

                    # any space to insert?
                    eval_insertion = word_diffs[widest]
                    # best_candidate_insertion_diff = 1000
                    best_candidate_insertion = None

                    if eval_insertion < -0.5:
                        sp_left = spaces_list[widest]  # spaces_list[widest - 1]
                        sp_right = spaces_list[widest + 1]  # spaces_list[widest]

                        # between sp_left/right
                        candidate_spaces = [(s_start, s_end) for s_start, s_end in spaces_diff_intervals
                                            if sp_left[1] <= s_start and s_end <= sp_right[0]]

                        if candidate_spaces:
                            # choosing the space that minimizes the difference between calculated and expected width
                            for cand in candidate_spaces:
                                diff_cand = abs((cand[0] - sp_left[1]) - words_widths_estimate[widest-1])
                                diff_no_cand = abs((sp_right[0] - sp_left[1]) - words_widths_estimate[widest-1])
                                if diff_cand < diff_no_cand:
                                    best_candidate_insertion = cand

                    # any space to remove?
                    eval_removal = word_diffs[narrowest]
                    eval_space_index = narrowest + 1
                    # show_image(name="eval_removal", image=highlight_spaces_line(img_line, spaces_intersect_intervals, "blue"))

                    if eval_removal > 0.5 and eval_space_index + 1 < len(spaces_list):
                        # narrowest/cand is between sp_left/right
                        sp_left = spaces_list[eval_space_index - 1]
                        sp_right = spaces_list[eval_space_index + 1]

                        cand = spaces_list[eval_space_index]
                        try:
                            left_word_estimate = words_widths_estimate[eval_space_index]
                        except IndexError:
                            pass
                        # if minimizes the error (difference)
                        if eval_removal > abs((sp_right[0] - sp_left[1] - left_word_estimate)) / left_word_estimate:
                            spaces_list.remove(cand)
                            spaces_intersect_intervals.remove(cand)

                    # elif eval_space_index >= len(words_widths_estimate) and eval_removal > 0.5:
                    #     assert spaces_list[-2] == spaces_intersect_intervals[-2]
                    #     spaces_list.pop(-2)
                    #     spaces_intersect_intervals.pop(-2)

                    if best_candidate_insertion:
                        insort(spaces_intersect_intervals, best_candidate_insertion)
                        insort(spaces_list, best_candidate_insertion)
                        spaces_diff_intervals.remove(best_candidate_insertion)

                    # ensure no infinite loop
                    if c > 5:
                        break
                    has_anomalous_width = False
                # END WHILE
                show_image(highlight_spaces_line(img_line, spaces_intersect_intervals, "cyan", random_lightness=True), name="after while loop")
                # break

                spaces_intersect_intervals = sorted(spaces_intersect_intervals)
                spaces_diff_intervals = sorted(spaces_diff_intervals)

                image_spaces_intersect = highlight_spaces_line(img_line, spaces_intersect_intervals, "blue")
                image_spaces_diff = highlight_spaces_line(img_line, spaces_diff_intervals, "cyan")
                # break

                #
                #
                #
                take_spaces = spaces_intersect_intervals
                missings = num_spaces - len(take_spaces)

                if len(spaces_intersect_intervals) != num_spaces:
                    # indexes of the nearest spaces in space_intersect to the ones in spaces_diff
                    # if new spaces must be taken, these are placed in nearest_spaces
                    nearest_spaces = [argmin(
                        [abs(center_space(its) - center_space(d)) for its in sorted(spaces_intersect_intervals)])
                        for d in spaces_diff_intervals]

                    candidate_spaces = []
                    print("nearest ", nearest_spaces)

                    for candidate, near in zip(spaces_diff_intervals, nearest_spaces):
                        left, right = None, None
                        # candidate = tuple(map(np.int, candidate))
                        near_space = spaces_intersect_intervals[near]

                        # candidate space is at the left of his nearest space (already taken)
                        if candidate[1] <= near_space[0]:
                            right = near_space[0]
                            if near > 0:
                                left = spaces_intersect_intervals[near - 1][1]
                            else:
                                left = start_text[1]

                            # width of the next word if candidate space is taken
                            candidate_next_word_width = near_space[0] - candidate[1]
                            estimated_width = words_widths_estimate[near]  # +1?
                            # print("word ", words[near])

                        else:  # candidate space is at the right of his nearest space (already taken)
                            left = near_space[1]
                            try:
                                right = spaces_intersect_intervals[near + 1][0]
                            except IndexError:
                                right = end_text[0]

                            assert right > left

                            candidate_next_word_width = candidate[0] - near_space[1]
                            estimated_width = words_widths_estimate[near + 1]
                            # print("word ", words[near+1])
                        try:
                            estimated_next_word_width = words_widths_estimate[near + 2]
                        except IndexError:
                            estimated_next_word_width = 0

                        candidate_word_width = candidate[0] - left
                        current_word_width = right - left
                        difference = abs(candidate_word_width - estimated_width) / estimated_width
                        # print("estimated_next ", estimated_next_word_width)
                        candidate_area = (candidate[1] - candidate[0]) * img_line.shape[0] / 2
                        white = count_white_in_interval(img_line, interval_x=candidate)
                        candidate_spaces.append((difference, white / candidate_area))

                        #
                        # pprint(OrderedDict({
                        #     "difference": difference,
                        #     "white": white,
                        #     "white_ratio": white / candidate_area
                        # }))
                        # print("\n\n")

                    print(candidate_spaces, '\n')

                    # END FOR

                    candidate_spaces = sorted([(idx, (csp, white)) for idx, (csp, white) in enumerate(candidate_spaces)
                                               if csp <= 0.5 and white >= 1.8], key=lambda e: e[1][0])[:missings]
                    take_spaces.extend([spaces_diff_intervals[c_index] for c_index, _ in candidate_spaces])

                image_spaces_taken = highlight_spaces_line(img_line, take_spaces, "magenta")

                print(ind, ")   words: {}, spaces found/expected: {}/{}\n"
                      .format(len(words), len(take_spaces), num_spaces))

                # print("taken spaces\n{}".format(sorted(take_spaces)))

                show_image(vstack(
                    [image_spaces, image_spaces_m, image_spaces_intersect, image_spaces_diff, image_spaces_taken]),
                    name="orig  man  int  diff  taken")

                print("••••••••••••••••••••••••••••••••••••••••••••••••••••••\n\n")
