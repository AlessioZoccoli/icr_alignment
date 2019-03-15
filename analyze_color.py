import cv2
from os import listdir, path
from numpy import argmax, zeros, array, full
from utils.visualization import show_image

dir = './page_images/40r-44v_130/'
images = [path.join(dir, i) for i in listdir(dir) if i[-4:] in {'.png', '.jpg'}]


def calculate_histogram(image: array) -> array:
    img = cv2.imread(image, 0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist


def mask_by_colors(word_img, colors) -> array:
    """
        freturns a mask on the image word_img based on colors
        :param word_img: np.ndarray dtype = uint8. shape = (height,width,channels)
        :param colors: np.ndarray dtype = uint8. Each element in the form (blue,green,red)
        :return: mask of shape (h,w) on the original one, which takes the value of 0 if colors are not present,
                255 otherwise.
    """
    mask = zeros((word_img.shape[0], word_img.shape[1]), dtype='uint8')

    for c in colors:
        # if len(c) == 1:
        #    print("\n\n\nC ", c)

        colormask = cv2.inRange(word_img, lowerb=c, upperb=c)
        mask = cv2.bitwise_or(mask, colormask)
    return mask


def show_histograms():
    for img in images:
        print(img)

        hist = calculate_histogram(img)
        most_freq = argmax(hist[:len(hist)//2])
        print("most frequent color {}\n".format(most_freq))

        image = cv2.imread(img)
        # show_image(image, minimize=True)
        mask_most_freq = mask_by_colors(word_img=image, colors=[array([most_freq], dtype="uint8")])  # array([most_freq]*3, dtype="uint8"))
        show_image(mask_most_freq, name=str(img.split('/')[-1])+' most_freq', minimize=True)


if __name__ == "__main__":
    show_histograms()
