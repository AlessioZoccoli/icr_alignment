import os
import json
import cv2
import shutil
import matplotlib.pyplot as plt


if __name__ == '__main__':
    src_folder = 'page_images_nicolas_best'
    dst_folder = 'aligned_gt_nicolas_best'
    img_folder = 'images'
    tsc_folder = 'transcriptions'
    alignment = json.load(open('bounding_boxes_elena.json', 'r'))

    for pagename in alignment:
        # create folder per page
        if os.path.isdir(os.path.join(dst_folder, pagename)):
            shutil.rmtree(os.path.join(dst_folder, pagename))
        os.makedirs(os.path.join(dst_folder, pagename, img_folder))
        os.makedirs(os.path.join(dst_folder, pagename, tsc_folder))

        page_n = pagename.split('_')[0]

        page_img = cv2.imread(
            os.path.join(src_folder, page_n+'.png'),
            cv2.IMREAD_GRAYSCALE
        )

        page_x, page_y, page_w, page_h = map(int, pagename.split('_')[1:])

        page_img_crop = page_img[page_y:page_y+page_h,page_x:page_x+page_w]

        for bbx in alignment[pagename]:
            word_x, word_y, word_w, word_h = map(int, bbx.split('_'))
            # cv2.rectangle(page_img_crop, (word_x,word_y), (word_x+word_w, word_y+word_h), 127, 1)
            cv2.imwrite(
                os.path.join(dst_folder, pagename, img_folder, bbx+'.png'),
                page_img_crop[word_y:word_y+word_h, word_x:word_x+word_w]
            )
            with open(os.path.join(
                        dst_folder,
                        pagename,
                        tsc_folder,
                        bbx+'.txt'),
                      'w') as tsc_file:
                tsc_file.write(alignment[pagename][bbx])
        # print(pagename)
        # plt.imshow(page_img_crop)
        # plt.show()

