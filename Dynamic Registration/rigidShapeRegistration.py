# encoding: utf-8

import numpy as np
import copy as cp
import math
import time
import mkl

from PIL import ImageFilter
from PIL import Image
from PIL import ImageDraw
from pylab import *
from matplotlib import animation
from scipy import spatial
from scipy import linalg


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

def matrixLeastSquare(A, b):
    #return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    return np.linalg.lstsq(A, b)[0]

#X1, Y1：待配准 X2, Y2：待变形
def compute_ZRT(X1, Y1, X2, Y2, length, tree):
    tick1 = time.time()

    A = np.zeros((4 * length, 2 * length + 3))
    w1 = 2**2
    w2 = 5**2
    b = np.zeros((4 * length, 1))

    print "******Time consumed: ", time.time() - tick1

    A1 = w1 * np.eye(2 * length)
    A[0: 2 * length, 0: 2 * length] = A1
    A2 = w2 * np.eye(2 * length)
    A[2 * length: 4 * length, 0: 2 * length] = A2

    for i in xrange(2 * length, len(A)):
        for j in xrange(2 * length, len(A[0])):
            if (i % 2 == 0):
                if (j - 2 * length == 0):
                    A[i][j] = Y2[(i - 2 * length) / 2] * w2
                elif (j - 2 * length == 1):
                    A[i][j] = -1 * w2
                else:
                    A[i][j] = 0
            else:
                if (j - 2 * length == 0):
                    A[i][j] = -X2[(i - 2 * length) / 2] * w2
                elif (j - 2 * length == 1):
                    A[i][j] = 0
                else:
                    A[i][j] = -1 * w2

    '''
    for i in xrange(0, len(A)):
        for j in xrange(0, len(A[0])):
            if (( i >= 0 ) and ( i < 2 * length) and (j >= 0) and (j < 2 * length)):
                if(i == j):
                    A[i][j] = w1
                else:
                    pass

            elif((i >= 2 * length) and (i < 4 * length) and (j >= 0 and (j < 2 * length))):
                if(i - j == 2 * length):
                    A[i][j] = w2
                else:
                    pass

            elif((i >= 2 * length) and (j >= 2 * length)):
                if(i % 2 == 0):
                    if(j - 2 * length == 0):
                        A[i][j] = Y2[(i - 2 * length) / 2] * w2
                    elif(j - 2 * length == 1):
                        A[i][j] = -1 * w2
                    else:
                        A[i][j] = 0
                else:
                    if(j - 2 * length == 0):
                        A[i][j] = -X2[(i - 2 * length) / 2] * w2
                    elif(j - 2 * length == 1):
                        A[i][j] = 0
                    else:
                        A[i][j] = -1 * w2

            else:
                pass
            '''

    print "******Time consumed: ", time.time() - tick1

    for i in xrange(0, 4 * length, 2):
        if i >= 0 and i < 2 * length:
            index = i / 2
            node = np.array([X2[index][0], Y2[index][0]])
            tree_dis, tree_pos = tree.query(node)
            corr_point = tree.data[tree_pos]
            corr_point_x = corr_point[0]
            corr_point_y = corr_point[1]
            b[i][0] = corr_point_x * w1
            b[i + 1][0] = corr_point_y * w1

        else:
            index = (i - 2 * length) / 2
            b[i][0] = X2[index] * w2
            b[i + 1][0] = Y2[index] * w2

    print "******Time consumed: ", time.time() - tick1

    mkl.set_num_threads(mkl.get_max_threads())

    x = linalg.lstsq(A, b)[0]
    #x = matrixLeastSquare(A, b)
    print "******Time consumed: ", time.time() - tick1

    return x

#Try to avoid too many copy and paste
def show_main_2():
    tick1 = time.time()

    image = Image.open('./data/man5.bmp').convert('L')
    im_contour = image.filter(ImageFilter.CONTOUR)
    image_size = image.size
    image_array = np.array(im_contour)

    contour_x = np.array([])
    contour_y = np.array([])
    contour_array = []

    for i in xrange(0, len(image_array), 2):
        for j in xrange(0, len(image_array[0]), 2):
            if image_array[i][j] != 255:
                image_array_item = [j, i]
                contour_array.append(image_array_item)
                contour_x = np.append(contour_x, j)
                contour_y = np.append(contour_y, i)


    print "Time consumes: ", (time.time() - tick1)

    im_contour_rotation = TupleFromList(trans(contour_array, 5, (2, 2), False))
    X_List_1, Y_List_1 = XYListFromList(im_contour_rotation, image_size)


def show_main_1():
    tick1 = time.time()

    image = Image.open('./data/man5.bmp').convert('L')
    im_contour = image.filter(ImageFilter.CONTOUR)
    image_size = image.size

    image_array = np.array(im_contour)
    contour_array = []

    for i in xrange(0, len(image_array), 2):
        for j in xrange(0, len(image_array[0]), 2):
            if image_array[i][j] != 255:
                    image_array_item = [j, i]
                    contour_array.append(image_array_item)

    print "Time consumes 1: ", time.time() -  tick1

    X_List, Y_List = XYListFromList(contour_array, image_size)

    im_contour_rotation = TupleFromList(trans(contour_array, 5 , (2, 2), False))
    X_List_1, Y_List_1 = XYListFromList(im_contour_rotation, image_size)

    len_array = len(X_List)

    print "len_array: ", len_array
    print "Time consumes 2: ", time.time() - tick1

    count = 0
    X1 = cp.deepcopy(X_List)
    Y1 = cp.deepcopy(Y_List)
    X2 = cp.deepcopy(X_List_1)
    Y2 = cp.deepcopy(Y_List_1)

    figure()
    subplot(1, 2, 1)

    plot(X1, Y1, 'ro')
    plot(X2, Y2, 'yo')

    first_X = []
    first_Y = []
    final_X = []
    final_Y = []

    #生成KDTree
    tree = spatial.KDTree(zip(X1, Y1))

    print  "Time consumes 3: ", time.time() - tick1

    while(count < 10):
        res = compute_ZRT(X1, Y1, X2, Y2, len_array, tree)
        print "Time consumes ", count + 4, ": ", time.time() - tick1
        X2 = res[0: len_array * 2: 2]
        Y2 = res[1: len_array * 2 + 1: 2]
        if count == 0:
            first_X = X2
            first_Y = Y2
        count += 1
        final_X = X2
        final_Y = Y2

    subplot(1, 2, 2)
    plot(first_X, first_Y, 'yo')
    plot(final_X, final_Y, 'bo')

    show()


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
    show_main_1()