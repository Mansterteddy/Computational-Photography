import numpy as np
from scipy import  misc
import matplotlib.pyplot as plt

def rescale_image(index, scale_size):
    image_filename = str(index) + ".JPG"
    raw_image = misc.imread(image_filename)
    (height, width, channel) = raw_image.shape
    print "width: ", width
    print  "height: ", height

    resize_image = misc.imresize(raw_image, (height / scale_size, width / scale_size, channel))
    output_filename = str(index) + "_resize.jpg"
    misc.imsave(output_filename, resize_image)
    #plt.imshow(resize_image)
    #plt.show()

def alpha_blend(index, scale_size, alpha=0.6):
    image_filename = str(index) + ".JPG"
    raw_image = misc.imread(image_filename).astype(float)
    (height, width, channel) = raw_image.shape

    final_image = np.zeros(raw_image.shape)

    for i in range(scale_size):
        for j in range(scale_size):
            image_index = i * scale_size + j + 1
            image_name = str(image_index) + '_resize.jpg'
            alpha_image = misc.imread(image_name).astype(float)
            start_x = i * height / scale_size
            end_x = (i + 1) * height / scale_size
            start_y = j * width / scale_size
            end_y = (j + 1) * width /scale_size
            print start_x, end_x, start_y, end_y
            print alpha_image.shape
            final_image[start_x: end_x, start_y: end_y] = alpha_image * (1 - alpha) + raw_image[start_x: end_x, start_y: end_y] * alpha

    plt.imshow(final_image.astype(np.uint8))
    plt.show()
    misc.imsave("final.png", final_image)


def test():
    raw_image_1 = misc.imread("5.JPG")
    raw_image_2 = misc.imread("4.JPG")
    alpha = 0.3
    raw_image_1 = raw_image_1.astype(float) * alpha + raw_image_2.astype(float) * (1 - alpha)
    plt.imshow(raw_image_1.astype(np.uint8))
    plt.show()

if __name__ == "__main__":
    #test()
    scale_size = 4
    for i in range(scale_size ** 2):
        rescale_image(i+1, scale_size)
    alpha_blend(17, scale_size)
