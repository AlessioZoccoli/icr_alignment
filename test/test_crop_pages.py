from os import listdir

from crop_pages import crop_page
from cv2 import imread, IMREAD_GRAYSCALE, resize, INTER_AREA
from utils.visualization import show_image


src_dir = '../page_images/40r-44v_130/'
img_filenames = [src_dir + o for o in listdir(src_dir) if o[-4:] == '.png']


def test_crop_page():
    for img_filename in img_filenames:
        cropped, new_name = crop_page(img_filename)
        new_name = img_filename.split('/')[-1][:-4] + new_name
        show_image(resize(cropped, None, fx=0.45, fy=0.45, interpolation=INTER_AREA), name=new_name)


if __name__ == '__main__':
    test_crop_page()
