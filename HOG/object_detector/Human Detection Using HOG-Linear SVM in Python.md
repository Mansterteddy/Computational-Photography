# Human Detection Using HOG-Linear SVM in Python

1、训练数据来自INRIA Person Dataset，其中正样本是64 128的人体图像，负样本是64 128的非人体图像

2、过程：

提取HOG特征（可以使用skimage库中的hog函数）

训练SVM（每张图片提取出来的HoG的特征有6480维，使用线性SVM）

进行人体检测（对图片进行滑动窗口检测，可能在不同尺度上都检测到了目标，这样会造成标记的混乱，可以使用非极大值抑制 NMS 对重复标记的目标进行剔除，这里从imutils包中导入非极大值抑制函数）