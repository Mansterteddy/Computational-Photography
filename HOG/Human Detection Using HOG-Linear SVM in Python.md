# Human Detection Using HOG-Linear SVM in Python

1、训练数据来自INRIA Person Dataset，其中正样本是64 128的人体图像，负样本是64 128的非人体图像

2、过程：

提取HOG特征（可以使用skimage库中的hog函数）

HOG特征：

1、灰度化 2、采用Gamma校正法对输入图像进行颜色空间的标准化（归一化），目标是调节图像的对比度，降低图像局部的阴影和光照变化所造成的影响，同时可以抑制噪音的干扰 3、计算图像中每个像素的梯度（包括大小和方向），主要是为了捕获轮廓信息，同时进一步弱化光照的干扰 4、将图像分为小的cells（6x6 pixels/cell） 5、统计每个cell的梯度直方图（不同梯度的个数），即可形成每个cell的descriptor 6、将每几个cell组成一个block（例如3x3 cell /block），一个block中所有cell的特征descriptor串联起来便可以得到该block的HOG特征descriptor 7、将图像image内的所有block的HOG特征descriptor串联起来就可以得到该image的HOG特征descriptor

训练SVM（每张图片提取出来的HoG的特征有6480维，使用线性SVM）

进行人体检测（对图片进行滑动窗口检测，可能在不同尺度上都检测到了目标，这样会造成标记的混乱，可以使用非极大值抑制 NMS 对重复标记的目标进行剔除，这里从imutils包中导入非极大值抑制函数）



































