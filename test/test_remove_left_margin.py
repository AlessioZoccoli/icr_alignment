from numpy import unique

from segmentation import remove_left_margin, remove_left_margin_2, remove_left_margin_fourier_manipulated, remove_left_margin_fourier
from cv2 import imread, IMREAD_GRAYSCALE, resize, INTER_AREA
from utils.visualization import show_image
from os import path, listdir

dir = '../page_images_crop/40r-44v_160/'
img_paths = [dir + o for o in listdir(dir) if o[-4:] == '.png'] # and o[:4] == '042v']


def test_remove_left_margin():
    for img_path in img_paths:
        assert path.exists(img_path)

        this_path = img_path.split('/')[-1]
        print('\n', this_path)

        img = imread(img_path, IMREAD_GRAYSCALE)

        removed_lf_img, _ = remove_left_margin_2(img)
        removed_lf_img_fourier_man, _ = remove_left_margin_fourier_manipulated(img)

        removed_lf_img_fourier, _ = remove_left_margin_fourier(img)

        show_image(resize(removed_lf_img,
                          None, fx=0.45, fy=0.45, interpolation=INTER_AREA),
                   name='vanilla_'+this_path)

        show_image(resize(removed_lf_img_fourier_man,
                          None, fx=0.45, fy=0.45, interpolation=INTER_AREA),
                   name='fourier_manipulated_'+this_path)

        show_image(resize(removed_lf_img_fourier,
                          None, fx=0.45, fy=0.45, interpolation=INTER_AREA),
                   name='fourier_'+this_path)


if __name__ == '__main__':
    test_remove_left_margin()

