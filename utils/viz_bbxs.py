import json
import  cv2


if __name__ == '__main__':
    gt = '_gt'
    alignment = json.load(open('bounding_boxes_elena.json', 'r'))
    page_folder = 'page_images_crop/'

    for page_fnm in alignment:
        page_img = cv2.imread(page_folder+page_fnm+'.png', cv2.IMREAD_GRAYSCALE)

        for bbx in alignment[page_fnm]:
            x, y, w, h = tuple(map(int, bbx.split('_')))

            cv2.rectangle(
                page_img,
                (x, y),
                (x+w, y+h),
                192,
                1
            )
            cv2.putText(
                page_img,
                alignment[page_fnm][bbx],
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                64,
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(page_fnm+gt+'.png', page_img)
