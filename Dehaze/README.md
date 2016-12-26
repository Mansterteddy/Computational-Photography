Dehaze实现了去雾和引导滤波器，分别来自于He Kaiming的两篇文章。
其中去雾算法为Single Image Haze Removal using Dark Channel Prior的实现。
引导滤波器为Guided Image Filtering的实现。

使用说明：
1、img文件夹中放置的是待处理的图像
2、result文件夹中放置的是处理过的图像
3、src文件夹中放置的是源程序

util.py文件用于返回待处理的文件路径
main.py文件接收待处理的文件路径，处理图片，同时输出去雾后的图片结果


dehaze.py文件实现Single Image Haze Removal using Dark Channel Prior

核心思想是：
$$
I =A*(1-t)+R*t
$$
在只知道I的情况下，求解A、R和t。

其中I是图像像素点值，A是atmosphere，t是transmission，R是radiance。

radiance就是去雾后的图像信息。

He Kaiming发现了图像中的一种特殊机制，在没有雾的图像中，除了sky region，每个像素的RGB通道中，总有一个通道值会很低，这个性质就被称之为Dark Channel Prior，唯一不满足这个性质的是sky region，但是在雾图片中，sky region往往和A很接近，因此sky region的t接近于0，相当于sky region的每个通道值不再重要，因为$R*t$为0，$I=A*(1-t)$仍然成立，可以无缝容纳到当前模型中。

所以先对整张图像求解drak_channel，得到一个和图像大小一致的矩阵。此时这个矩阵的值应当只是：$I= A(1-t)$，那么假如知道A，我们就可以求解得到t。

在这里，我们假设整张图像的A都是一致的，而且雾引起的A接近于白色，而R更丰富一些，所以A要大于R，因此越大的I，表明t越接近0，也就是约等于A。在本文中，为了更进一步，我们选取在Dark Channel里找最大的A，这样的A更真实，因为前一种方法找到的A有可能是场景中的一个白色物体。

因此我们可以计算得到整张图的t。

但是由于此时计算得到的t是针对每一个像素计算得到的，因此在空间上不具备连续性，此时需要滤波，在这里我们使用的是He Kaiming的另一个工作：引导滤波器。

滤波后得到了A、t和I，那么可以很容易求解得到R。



guidedfilter.py文件用于实现Guided Image Filtering，接受一个输入图像和一个用于引导的图像，输出一个滤波后的图像，使得滤波图像在滤波后，和引导图像的“结构”保持接近。

这篇文章的重点就是引入了一个ridge regression。
$$
E(a_k,b_k)=\sum((a_kI_i+b_k-p_i)^2+\sigma a_k^2)
$$
I是引导图像，经过a和b的变化后，应当尽可能和输入图像p接近，后面的平方项是惩罚因子。在这里，采用滑动窗口法，对每一个窗口求解a和b。

因为是基于滑动窗口的方法，因此对于每个像素的a和b都可能求解了很多次，因此需要求平均。

Algorithm:

1、	

mean_I = f_mean(I)
mean_p = f_mean(p)
corr_I = f_mean(I.* I)
corr_IP = f_mean(I.* p)

2、

VarI = corrI - mean_I.* mean_I
cov_Ip = corr_Ip - mean_I.* mean_p

3、

a = cov_Ip ./ (var_I + sigma)
b = mean_p - a.* meanI 

4、

mean_a = fmean(a)
mean_b = fmean(b)

5、

q = mean_a.* I + mean_b

由此计算出滤波后的图像q。

在实现滑动窗口法的过程中，最好不要一个一个窗口遍历，而是聪明地采用矩阵操作，
利用numpy中的cumsum函数，使得矩阵中每一个位置的元素，都和对应窗口中的元素进行运算。
构造一个通用的窗口滤波函数，使得处理任何矩阵成为可能，包括全1矩阵，这类矩阵可以提供加权平均时，
对应的系数，最终直接使用两个矩阵相除，就可以得到结果。