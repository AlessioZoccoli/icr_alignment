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
                  .replace('(rum)', '2')
                  .replace('r(um)', '2')
                  .replace('(Christ)', 'x')
                  .replace('(christ)', 'x')) \
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


def rec_defaultdict():
    # arbitrarily nested defaultdict
    return defaultdict(rec_defaultdict)


def test_start_alignment():
    dst_dir = '../aligned'
    tsc_dir_name = 'transcriptions'
    img_dir_name = 'images'
    map_dir_name = 'mapping'

    fnms = set(f.split('.')[0] for f in os.listdir(IMG_DIR))
    fnms = sorted(fnms - set(os.listdir(dst_dir)))

    print("aligned so far: ", os.listdir(dst_dir))

    #
    #   this is the output data structure
    #      {
    #          filename: {
    #                      row_row_index: {
    #                                      bbx: 'transcription'
    #                                      ...
    #                                  }
    #                    }
    #      }
    #
    #

    for fnm in fnms:
        print('\n###### {} ######'.format(fnm))

        stop = input("STOP alignment? [y/n] ")
        if stop == "y":
            break
        else:
            # output data structure
            bboxes2transcript = rec_defaultdict()

            images_path = os.path.join(dst_dir, fnm, img_dir_name)
            transcriptions_path = os.path.join(dst_dir, fnm, tsc_dir_name)
            mapping_path = os.path.join(dst_dir, fnm, map_dir_name)

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

            #
            #  alignment begins
            #
            # for each line: transcription of line L, image of line L
            transcr2img = zip(tsc_lines[:], img_lines[:])

            # testing specific lines
            try:
                lines_to_test = list(map(int, input("choose lines to test ").strip().split(',')))
                print(lines_to_test)
            except ValueError as v:
                print(v)
                lines_to_test = []

        for row_ind, (tsc_line, (img_line, top_y)) in enumerate(transcr2img):
            if row_ind in lines_to_test or not lines_to_test:
                #
                #    original: no dilatation or erosion
                #
                # spaces, center = spaces_in_line(img_line)
                (spaces, (first, last)), center = spaces_in_line_simple(img_line)
                num_spaces = len(tsc_line.split()) - 1

                print('\n', "  •  ".join(tsc_line.split()), '\n')

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
                # image_spaces_m = highlight_spaces_line(img_line, biggest_spaces_m, "green")
                spaces_m_flatten = [el for rng in intervals_to_ranges(biggest_spaces_m) for el in rng]
                spaces_orig_flatten = [el for rng in intervals_to_ranges(biggest_spaces) for el in rng]

                # start of the line
                whitest_pnt_start = min([
                    (np.argmax(np.count_nonzero(img_line[:, 0:first] == 0, axis=0), axis=0), 0),
                    (np.argmax(np.count_nonzero(img_line[:, 0:first_m] == 0, axis=0), axis=0), 1)
                ], key=lambda e: e[0])
                start_text = (0, whitest_pnt_start[0])
                # end of the line
                end_left_pt = min(last_m, last)
                white_count_rev = np.count_nonzero(img_line[:, end_left_pt:], axis=0)[::-1]
                whitest_pnt_end = len(white_count_rev) - 1 - np.argmax(white_count_rev) + end_left_pt
                end_text = (whitest_pnt_end, img_line.shape[1])


                ####
                # show_image(img_line[:, 0:first], name=str(whitest_pnt_start[1] == 0)[0] + ' ' + str(row_ind))
                # show_image(img_line[:, 0:first_m], name=str(whitest_pnt_start[1] == 1)[0] + ' manip ' + str(row_ind))
                # show_image(img_line[:, 0:max(start_text[1], 1)])

                # show_image(img_line[:, end_text[0]:])

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

                #
                #   difference
                #   considering the interval as a hole not just set of consecutive pixels, so if part of an interval
                #   is in difference then it is discarded
                #
                spaces_difference = setxor1d(spaces_m_flatten, spaces_orig_flatten)
                spaces_diff_intervals = [(sp[0], sp[-1]) for row_ind, sp in enumerate(
                    group_consecutive_values(spaces_difference, threshold=1))
                                         if
                                         sp[0] - 1 not in spaces_orig_flatten and sp[0] - 1 not in spaces_m_flatten and
                                         sp[-1] + 1 not in spaces_orig_flatten and sp[-1] + 1 not in spaces_m_flatten
                                         ]
                spaces_diff_intervals.extend(spaces_intersect_intervals_sorted[num_spaces + othersp:])
                spaces_diff_intervals = sorted(spaces_diff_intervals)

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
                    words_widths_estimate.append(estimate_word_width(w, previous_word=previous_w))

                #
                # checking that intersection picked the right spaces
                #
                c = 0

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
                    words_widths_diffs = sorted(range(len(word_diffs)), key=lambda row_ind_w: word_diffs[row_ind_w])
                    widest = words_widths_diffs[0]
                    narrowest = words_widths_diffs[-1]

                    # any space to insert?
                    eval_insertion = word_diffs[widest]
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
                                diff_cand = abs((cand[0] - sp_left[1]) - words_widths_estimate[widest - 1])
                                diff_no_cand = abs((sp_right[0] - sp_left[1]) - words_widths_estimate[widest - 1])
                                if diff_cand < diff_no_cand:
                                    best_candidate_insertion = cand

                    # any space to remove?
                    eval_removal = word_diffs[narrowest]
                    eval_space_row_index = narrowest + 1

                    if eval_removal > 0.5 and eval_space_row_index + 1 < len(spaces_list):
                        # narrowest/cand is between sp_left/right
                        sp_left = spaces_list[eval_space_row_index - 1]
                        sp_right = spaces_list[eval_space_row_index + 1]

                        cand = spaces_list[eval_space_row_index]
                        try:
                            left_word_estimate = words_widths_estimate[eval_space_row_index]
                        except IndexError:
                            pass
                        # if minimizes the error (difference)
                        if eval_removal > abs((sp_right[0] - sp_left[1] - left_word_estimate)) / left_word_estimate:
                            spaces_list.remove(cand)
                            spaces_intersect_intervals.remove(cand)

                    if best_candidate_insertion:
                        insort(spaces_intersect_intervals, best_candidate_insertion)
                        insort(spaces_list, best_candidate_insertion)
                        spaces_diff_intervals.remove(best_candidate_insertion)

                    # ensure no infinite loop
                    if c > 5:
                        break
                    has_anomalous_width = False
                # END WHILE

                spaces_intersect_intervals = sorted(spaces_intersect_intervals)
                spaces_diff_intervals = sorted(spaces_diff_intervals)

                #
                #   Second processing step evaluates weather a space in diff_intervals should be taken or not
                #
                take_spaces = spaces_intersect_intervals
                missings = num_spaces - len(take_spaces)

                if len(spaces_intersect_intervals) != num_spaces:
                    # row_indexes of the nearest spaces in space_intersect to the ones in spaces_diff
                    # if new spaces must be taken, these are placed in nearest_spaces
                    nearest_spaces = [argmin(
                        [abs(center_space(its) - center_space(d)) for its in sorted(spaces_intersect_intervals)])
                        for d in spaces_diff_intervals]

                    candidate_spaces = []

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
                            # candidate_next_word_width = near_space[0] - candidate[1]
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
                            try:
                                estimated_width = words_widths_estimate[near + 1]
                            except IndexError:
                                print("## no near +1 ", near + 1)
                                pass  # keeps the last assignment
                        # try:
                        #     estimated_next_word_width = words_widths_estimate[near + 2]
                        # except IndexError:
                        #     estimated_next_word_width = 0

                        candidate_word_width = candidate[0] - left
                        # current_word_width = right - left
                        difference = abs(candidate_word_width - estimated_width) / estimated_width
                        # print("estimated_next ", estimated_next_word_width)
                        candidate_area = (candidate[1] - candidate[0]) * img_line.shape[0] / 2
                        white = count_white_in_interval(img_line, interval_x=candidate)
                        candidate_spaces.append((difference, white / candidate_area))
                    # END FOR

                    candidate_spaces = sorted([(idx, (csp, white)) for idx, (csp, white) in enumerate(candidate_spaces)
                                               if csp <= 0.5 and white >= 1.8], key=lambda e: e[1][0])[:missings]
                    take_spaces.extend([spaces_diff_intervals[c_row_index] for c_row_index, _ in candidate_spaces])

                image_spaces_taken = highlight_spaces_line(img_line, take_spaces, "magenta")

                take_spaces = sorted(take_spaces)

                # SOME INFO
                print(row_ind, ")   words: {}, spaces found/expected: {}/{}\n"
                      .format(len(words), len(take_spaces), num_spaces))

                # row_indexes of the spaces selected by the user
                selected_row_indexes = show_image_ask_spaces(image_spaces_taken, tot_spaces=len(take_spaces))
                image_selected_spaces = highlight_spaces_line(img_line,
                                                              [take_spaces[i] for i in selected_row_indexes], "blue")

                # draw missing spaces if any
                has_missing_spaces = input("draw spaces? (empty = NO else int):  ")
                selected_spaces = []
                if has_missing_spaces:
                    # valid input
                    while not isinstance(eval(has_missing_spaces), int):
                        has_missing_spaces = input("insert number of spaces to insert:  ")

                    has_drawn = False
                    # ask until the number of input spaces is correct
                    while not has_drawn:
                        print("draw and press 'q' to exit\n")
                        try:
                            num_missings = eval(has_missing_spaces)  # to be inserted
                            draw_missing_spaces(image_selected_spaces)
                            drawn = get_drawn_spaces()  # drawn spaces

                            assert len(drawn) == num_missings

                            for d_lft, d_rt in drawn:
                                # inserted spaces overlaps start_text or end_text?
                                if start_text[0] < d_lft <= start_text[1]:
                                    print("### inserted starting space")
                                    start_text = (start_text[0], np.argmax(
                                        np.count_nonzero(img_line[:, :d_lft], axis=0), axis=0))
                                elif end_text[0] <= d_rt < end_text[1]:
                                    print("### inserted ending space ", end_text)
                                    white_count_rev = np.count_nonzero(img_line[:, d_rt:], axis=0)[::-1]
                                    left_end = len(white_count_rev) - 1 - np.argmax(white_count_rev, axis=0)
                                    end_text = (left_end, end_text[1])

                                    """
                                    # the correct cutting columns is likely to be at the end of the line (and of the
                                    # selected sub-image)
                                    guard = min(end_text[1] - d_rt, 10)
                                    # argmax take the "first" max, so start scanning from the right
                                    white_count_rev = np.count_nonzero(img_line[:, d_rt + guard:], axis=0)[::-1]
                                    left_end = len(white_count_rev) - 1 - np.argmax(white_count_rev, axis=0)\
                                        + d_rt + guard
                                    end_text = (left_end, end_text[1])
                                    """

                            selected_spaces = sorted(
                                [start_text] + [take_spaces[i] for i in selected_row_indexes] + drawn + [end_text]
                            )
                            # exit
                            has_drawn = True
                        except AssertionError:
                            print("\nERROR: number of drawn spaces != number of spaces requested, {} {}".format(
                                len(drawn), num_missings
                            ))
                            cv2.destroyAllWindows()
                else:
                    selected_spaces = sorted([start_text] + [take_spaces[i] for i in selected_row_indexes] + [end_text])

                image_select_draw_spaces = highlight_spaces_line(img_line, selected_spaces, "red")
                show_image(image_select_draw_spaces, name="selected + drawn")

                ####
                # show_image(img_line[:, start_text[0]:start_text[1]], name="space before text")

                for sp in range(1, len(selected_spaces)):
                    space_left = selected_spaces[sp - 1]
                    space_right = selected_spaces[sp]
                    # whitest column left
                    try:
                        left = np.argmax(np.count_nonzero(img_line[:, space_left[0]:space_left[1]], axis=0))
                    except ValueError:
                        print(space_left[0], space_left[1], " attempt to get argmin of an empty sequence")
                        left = 0
                    left += space_left[0]

                    # whitest column right
                    """
                    
                                    white_count_rev = np.count_nonzero(img_line[:, d_rt + guard:], axis=0)[::-1]
                                    left_end = len(white_count_rev) - 1 - np.argmax(white_count_rev, axis=0)\
                                        + d_rt + guard
                                    end_text = (left_end, end_text[1])
                    
                    """
                    try:
                        if sp == len(selected_spaces)-1:  # last space?
                            print("###### cutting the last space")
                            white_count_rev = np.count_nonzero(img_line[:, space_right[0]:], axis=0)[::-1]
                            right = len(white_count_rev) - 1 - np.argmax(white_count_rev)
                        else:
                            right = np.argmax(np.count_nonzero(img_line[:, space_right[0]:space_right[1]], axis=0))
                    except ValueError:  # attempt to get argmin of an empty sequence
                        print(space_right[0], space_right[1], " attempt to get argmax of an empty sequence")
                        right = space_right[1]
                    right += space_right[0]

                    # left (start bbx x), top_y (start bbx y), width, height
                    bbx_name = "{}_{}_{}_{}".format(left + left_margin, top_y, right - left, img_line.shape[0])
                    try:
                        bboxes2transcript[row_ind][bbx_name] = words[sp - 1]
                        show_image(img_line[:, left: right], name=words[sp - 1])
                        # print(words[sp - 1], '\n')

                    except IndexError:
                        print("words and index = ", sp - 1, "  len(words) ", len(words))

                print("\n ••••••••••••••••••••••••••••••••••••••••••••••••••••••\n\n")


if __name__ == '__main__':
    test_start_alignment()
