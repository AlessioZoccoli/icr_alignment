import cv2
import numpy as np
import os
from segmentation import group_consecutive_values, find_local_minima, smooth


def crop_pages(src_folder, dst_fldr):
    filenames = os.listdir(src_folder)

    for filename in sorted(filenames):
        print(filename)
        cropped_img, cropped_name = crop_page(os.path.join(src_folder, filename))
        write_status = cv2.imwrite(os.path.join(dst_fldr, filename[:-4] + cropped_name), cropped_img)
        assert write_status


def crop_page(img_filename):
        page = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)

        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(page, kernel, iterations=1)

        rows_count = np.count_nonzero(cv2.bitwise_not(dilation), axis=1)
        cols_count = np.count_nonzero(cv2.bitwise_not(dilation), axis=0)

        rows_count_smooth = smooth(rows_count, window_len=270)
        cols_count_smooth = smooth(cols_count, window_len=210)

        row_thresh = 60
        col_thresh = 60

        row_peaks = group_consecutive_values([ix[0] for ix in np.argwhere(rows_count_smooth >= row_thresh)], threshold=70)
        col_peaks = group_consecutive_values([ix[0] for ix in np.argwhere(cols_count_smooth >= col_thresh)])

        row_peaks_start_end = np.array([(peak[0], peak[-1]) for peak in row_peaks])
        col_peaks_start_end = np.array([(peak[0], peak[-1]) for peak in col_peaks])

        # print(row_peaks)
        # print(col_peaks)
        # print("\n\n")
        # print(row_peaks_start_end,'\n')
        # print(col_peaks_start_end)

        rows_maxpeak_ix = np.argmax([end-start for start, end in row_peaks_start_end])
        cols_maxpeak_ix = np.argmax([end-start for start, end in col_peaks_start_end])

        rows_maxpeak_start, rows_maxpeak_end = row_peaks_start_end[rows_maxpeak_ix]
        cols_maxpeak_start, cols_maxpeak_end = col_peaks_start_end[cols_maxpeak_ix]

        rows_top_cut = np.max(find_local_minima(rows_count[:rows_maxpeak_start]))
        rows_bottom_cut = np.min(find_local_minima(rows_count[rows_maxpeak_end:]))+len(rows_count[:rows_maxpeak_end])

        cols_left_cut = np.max(find_local_minima(cols_count[:cols_maxpeak_start]))
        cols_right_cut = np.min(find_local_minima(cols_count[cols_maxpeak_end:]))+len(cols_count[:cols_maxpeak_end])

        border = 70

        x, y = (cols_left_cut-border), (rows_top_cut-border)
        w, h = (cols_right_cut+border-x), (rows_bottom_cut+border-y)
        crop = page[y:y+h, x:x+w]

        crop_filename = img_filename.split('.')[0]+('_%d_%d_%d_%d.png' % (x, y, w, h))
        # print("crop_filename ", os.path.join(dst_folder, crop_filename))

        return crop, crop_filename

        # write_status = cv2.imwrite(os.path.join(dst_folder, crop_filename), crop)
        # assert write_status

        # print(row_peaks_start_end)
        # plt.plot(rows_count_smooth)
        # plt.plot(np.ones(len(rows_count_smooth))*row_thresh)
        # plt.show()
        #
        # print(col_peaks_start_end)
        # plt.plot(cols_count_smooth)
        # plt.plot(np.ones(len(cols_count_smooth))*col_thresh)
        # plt.show()


if __name__ == '__main__':
    src_fldr = './page_images/40r-44v_165/'
    dst_fldr = './page_images_crop/40r-44v_165/'

    if not os.path.exists(dst_fldr):
        os.makedirs(dst_fldr)

    crop_pages(src_fldr, dst_fldr)
