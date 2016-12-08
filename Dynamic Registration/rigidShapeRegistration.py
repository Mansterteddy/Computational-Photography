# encoding: utf-8

from PIL import ImageFilter
from PIL import Image
from PIL import ImageDraw
import numpy as np
import copy as cp
import math
from pylab import *
from matplotlib import animation

def RadiansFromDegrees(theta):
    return theta * (math.pi / 180.0)

# python传值和传引用是根据参数对象来的，如果函数收到的是一个可变对象的引用，那么在函数里就能够修改对象的原始值
# 如果函数收到的是一个不可变对象的引用（比如数字、字符或者元组），那么就不能直接修改原始对象
# 尚未完成的部分：增加noise
def trans(X, theta, offset, noise):

    Y = cp.deepcopy(X)

    theta_radians = RadiansFromDegrees(theta)
    rotation_mat = np.array(
        [[math.cos(theta_radians), - math.sin(theta_radians)], [math.sin(theta_radians), math.cos(theta_radians)]])

    offset_array = np.array([[offset[0], offset[1]]]).T

    for i in xrange(0, len(X)):

        Y[i] = np.dot(rotation_mat, np.array([X[i]]).T) + offset_array

    return Y


def TupleFromList(X):
    Y = []
    for i in xrange(0, len(X)):
        Y.append(tuple(X[i]))
    return Y

def XYListFromList(X, size):
    X_List = []
    Y_List = []
    for i in xrange(0, len(X)):
        X_List.append(X[i][0])
        Y_List.append(size[1] - X[i][1])

    return X_List, Y_List

def compute_ZRT():
    pass


def show_main():
    # 'L' for greyscale images, 'RGB' for true color images and 'CMYK' for pre-press image
    image = Image.open('./data/man5.bmp').convert('L')
    im_contour = image.filter(ImageFilter.CONTOUR)
    print "image size: ", image.size
    image_size = image.size

    image_array = np.array(im_contour)
    contour_array = []
    for i in xrange(0, len(image_array)):
        for j in xrange(0, len(image_array[0])):
            if image_array[i][j] != 255:
                # 图像坐标系的特殊之处，坐标系的原点在左上角，x向下，y向右
                image_array_item = [j, i]
                contour_array.append(image_array_item)

    X_List, Y_List = XYListFromList(contour_array, image_size)

    fig = figure(figsize = (8, 6), dpi = 80)
    subplot(1, 1, 1)

    ax = plt.axes(xlim = (0, image_size[0]), ylim = (0, image_size[1]))

    line1, = plt.plot([], [], 'ro')
    line2, = plt.plot([], [], 'ro')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    #这里就是一个很好的closure函数的例子，需要回调函数时，但是回调函数需要应用其他变量，此时就传参给它的父函数，然后父函数返回这个闭包函数。
    def animate_closure(contour_array):
        def animate(i):
            if(i < 5):
                im_contour_rotation = TupleFromList(trans(contour_array, 5 + 1 * i, (0, 0), False))
                X_List_1, Y_List_1 = XYListFromList(im_contour_rotation, image_size)
                line1.set_data(X_List, Y_List)
                line2.set_data(X_List_1, Y_List_1)
                return line1, line2
            else:
                return line1, line2

        return animate
    '''
    def animate(i):
        im_contour_rotation = TupleFromList(trans(contour_array, 5 + 0.01 * i, (0, 0), False))
        X_List_1, Y_List_1 = XYListFromList(im_contour_rotation, image_size)
        line1.set_data(X_List, Y_List)
        line2.set_data(X_List_1, Y_List_1)
        return line1, line2
'''
    animator = animation.FuncAnimation(fig, animate_closure(contour_array), init_func = init, frames = 10, interval = 10, blit = True)

    show()

    '''
    X_List, Y_List = XYListFromList(contour_array, image_size)

    plot(X_List, Y_List, 'ro')

    im_contour_rotation = TupleFromList(trans(contour_array, 5, (5, 0), False))
    X_List_1, Y_List_1 = XYListFromList(im_contour_rotation, image_size)

    plot(X_List_1, Y_List_1, 'ro')

    xlim(0, image.size[0])
    ylim(0, image.size[1])

    show()
    '''

    '''
    im_contour_rotation = TupleFromList(trans(contour_array, 5, (5, 0), False))
    draw = ImageDraw.Draw(im_contour)
    draw.point(im_contour_rotation, fill=(255, 0, 0))
    del draw
    im_contour.show()
    '''


    # xrange 和 range ：range返回的是列表 xrange是迭代器
    # 深拷贝和浅拷贝 浅拷贝：copy.copy 深拷贝：copy.deepcopy
    '''
    im_contour_i = copy.deepcopy(im_contour)
    draw = ImageDraw.Draw(im_contour_i)
    draw.line((0, 0, im_contour_i.size[0], im_contour_i.size[1]), fill = 128)
    del draw
    im_contour.show()
    '''


if __name__ == "__main__":
    show_main()