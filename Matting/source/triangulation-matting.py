#coding = utf-8

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

#b1, b2 as example, two images, for the same foreground, fg and alpha channel are all same
#fg + (1 - alpha) * b_1 = c_1
#fg + (1 - alpha) * b_2 = c_2
#There are three color channel, so there is 6 equations, take it into matrix form, we want to compute fg, alpha.
#Then the composite image is fg + (1 - alpha) * new_backgroud

def matting(b1, b2, c1, c2):
        """
        Compute the triangulation matting equation

        Param:
        b1: background image 1
        b2: background image 2
        c1: composite image 1
        c2: composite image 2

        Returns:
        fg: foreground Image
        alpha: alpha image
        """
        b1_r, b1_g, b1_b = b1[:, :, 0], b1[:, :, 1], b1[:, :, 2]
        b2_r, b2_g, b2_b = b2[:, :, 0], b2[:, :, 1], b2[:, :, 2]
        c1_r, c1_g, c1_b = c1[:, :, 0], c1[:, :, 1], c1[:, :, 2]
        c2_r, c2_g, c2_b = c2[:, :, 0], c2[:, :, 1], c2[:, :, 2]

        img_shape = b1.shape 
        fg = np.zeros(img_shape)
        #img_shape includes width * height * 3, alpha only need width and height, so range from 0 to 1
        alpha = np.zeros(img_shape[: 2])

        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                a = np.array([
                [b1_r[i, j]], 
                [b1_g[i, j]],
                [b1_b[i, j]],
                [b2_r[i, j]],
                [b2_g[i, j]],
                [b2_b[i, j]]])
                b = np.array([
                [c1_r[i, j] - b1_r[i, j]],
                [c1_g[i, j] - b1_g[i, j]],
                [c1_b[i, j] - b1_b[i, j]],
                [c2_r[i, j] - b2_r[i, j]],
                [c2_g[i, j] - b2_g[i, j]],
                [c2_b[i, j] - c2_b[i, j]]])

                #hstack: Take a sequence of arrays and stack them horizontally to make a single array.
                A = np.hstack((matrix, -1 * a))
                #pinv: calculate the pseudo-inverse of a matrix, assume A is singular, AXA = A X is A's pseudo-inverse
                #clip: Limit the values in an array
                x = np.clip(np.dot(np.linalg.pinv(A), b), 0.0, 1.0)
                fg[i, j] = np.array([x[0][0], x[1][0], x[2][0]])
                #Matrix dim is 6 * 4, 4 * 1, 6 * 1, alpha is the 4th result in array.
                alpha[i, j] = x[3][0]
        return fg, alpha


def multiply_alpha(alpha, b):
    """
    Multiplies (1-alpha) and the backgoround Image
    
    Param:
    alpha: alpha matte Image
    b: new background Image

    Returns:
    c: (1 - alpha) * background
    """

    img_shape = b.shape
    c = np.zeros(img_shape)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            #alpha image has one value for each pixel
            #unlike the background image which has three values - r, g, b
            c[i][j] = b[i][j] * (1.0 - alpha[i][j])
        
    return c;


if __name__ == "__main__":
    window = np.array(Image.open('window.jpg')) / 255.0
    
    b1 = np.array(Image.open('flowers-backA.jpg')) / 255.0
    b2 = np.array(Image.open('flowers-backB.jpg')) / 255.0
    c1 = np.array(Image.open('flowers-compA.jpg')) / 255.0
    c2 = np.array(Image.open('flowers-compB.jpg')) / 255.0
    
    fg, alpha = matting(b1, b2, c1, c2)
    
    imsave('flowers-alpha.jpg', alpha, cmap = cm.gray)
    imsave('flower-foreground.jpg', fg)
    
    b = multiply_alpha(alpha, window)
    composite = fg + b
    plt.show(imshow(composite))
    imsave('flower-composite.jpg', composite)