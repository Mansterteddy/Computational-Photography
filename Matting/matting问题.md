# matting问题

以前follow邱师兄的工作，读了不少matting的论文。前几天和GraphiCon的人聊天时，王士玮师兄说，在我Github下看到了Chen Qifeng的名字。最近又看到Sun Jian的homepage上写，Wang Jue加入了Face++。

随后想了想，matting是个很基本的问题，里面有不少人，也出现不少好的工作，这些工作反映了机器学习乃至计算机科学中很本质的一些东西，于是乎就写成了这篇博客。

## Blue Screen Matting

这篇文章的作者是Alvy Ray Smith和James F. Blinn，如果按修真的设定，这就是开宗立派的祖师爷了。

James F. Blinn原来在JPL工作，对，就是《火星救援》里，把马特达蒙从火星救回来的的那个JPL。

Alvy Ray Smith是图形学的先驱，他的职业生涯十分传奇，最开始在施乐做绘图程序，当然后面的故事我们都知道了，艾伦凯邀请乔布斯访问施乐，拥有图形界面的mac正式问世，从此改变了世界。后来老爷子给自己所在的动画公司起了名字，叫Pixar，再后来，老爷子就到了微软，在微软期间，他完成了这篇论文，当然他还是HSV颜色空间的作者。

老爷子有个很酷的中文名字，值得讲一下。老爷子的英文是Alvy Ray Smith，Ray是光线的意思，Smith在英文中有铁匠之意，所以老爷子的英文名应该翻译为：操控光线的匠人，这既是他的名字，又暗指了他的工作，图形学不就是操作光线么？有位中国人为他起了这个十分精彩的中文名：匠白光。（此处我想起了高德纳这个名字，也是精彩之极。）

返回这篇文章，这篇文章第一次定义了matting问题，即：

$I=\alpha F+(1-\alpha)B$

一张图片I，应该由前景F和前景B合成，合成的方式，就是通过透明度$\alpha$来操作。使用透明度、前景和背景，我们可以合成这样的照片。 ![final](C:\Users\manster\Documents\GitHub\Computational-Photography\Image_Editor\final.png)

matting问题研究的是，如何通过左边的I，推测出右边的三个变量$\alpha$、F和B，难度可想而知。

在这篇文章中，作者给出了Triangulation Matting的方法，整体思想也很简单，假设我知道了B和I，那么有没有可能得到$\alpha$和F，于是乎，作者提出了，针对同一张前景，切换背景，来计算得到前景和透明度的方法。如下图所示：

假设知道了两张前景背景图和对应的背景图：

 ![flowers-backA](C:\Users\manster\Documents\code\matting\triangulation-matting\flowers-backA.jpg)

 ![flowers-compA](C:\Users\manster\Documents\code\matting\triangulation-matting\flowers-compA.jpg)

 ![flowers-backB](C:\Users\manster\Documents\code\matting\triangulation-matting\flowers-backB.jpg)

 ![flowers-compB](C:\Users\manster\Documents\code\matting\triangulation-matting\flowers-compB.jpg)

应用最小二乘法，计算得到对应的前景和透明度（这里有个骚操作，使用SVD来加速计算）：

 ![flower-foreground](C:\Users\manster\Documents\code\matting\triangulation-matting\flower-foreground.jpg)

 ![flowers-alpha](C:\Users\manster\Documents\code\matting\triangulation-matting\flowers-alpha.jpg)

随后结合新的背景，就可以生成极富真实性的合成图片了：

 ![window](C:\Users\manster\Documents\code\matting\triangulation-matting\window.jpg)

 ![flower-composite](C:\Users\manster\Documents\code\matting\triangulation-matting\flower-composite.jpg)

当然那个时候，matting问题还主要应用于电影拍摄等场景，因此可以看出，这篇文章的操作很复杂，应用范围也很有限。

这篇文章还介绍了一些工业界的方法，里面有一些Magic Number，本文让我深切地感受到，真正的技术壁垒，并不是方法或者理论，而是数据库，和经过无数次实验，找到的Magic Number。

## Poisson Matting

这篇文章的作者列表上，有Sun Jian，Jia Jiaya，Tang Chi-Keung，Shum Heung-Yeung。前段时间MSRA周年庆，不少大牛都来了，读这些大牛的paper，就仿佛阅读《三国志》一样，你可以很清晰地看到，这些人的名字是如何一个一个串联在一起的。我打算以后写代码混不下去的话，就去写《各大公司首席科学家列传》了。

回到这篇文章，随着matting技术的发展，学术界也形成了一套解决matting的数据库和标准流程，其中最特殊的一点是，使用tri-map作为辅助工具。trai-map分为三种颜色，黑色代表完全背景（此处$\alpha$为0），白色代表完全前景（此处$\alpha$为1），灰色代表不确定区域（$\alpha$待定）。

 ![trollTrimap](C:\Users\manster\Documents\code\matting\poisson-matting\trollTrimap.png)

这篇文章，对matting等式进行观察后，两边求导，得到如下的式子：
$$
\nabla I=(F-B)\nabla \alpha+\alpha\nabla F+(1-\alpha)\nabla B
$$
在这里我们假设$\nabla F$和$\nabla B$很小，所以式子就简化成如下的形式：
$$
\nabla \alpha\approx \frac{1}{F-B}\nabla I
$$
于是乎，我们就能写出一个这样的能量方程：
$$
\alpha^*=argmin_\alpha\int\int_{p\in \Omega}||\nabla\alpha_p-\frac{1}{F_p-B_p}\nabla I_p||^2dp
$$
其中$\Omega$指的就是不确定区域。上述问题可以很自然地转成微分方程去做，具体地，转为泊松方程求解，这也是这个方法叫Poisson Equation的原因。当然，这篇文章发表的时间段，也正是研究人员将泊松方程系统地引入图像处理的时间段，在计算机历史上，我们经常可以看到，将数学和物理的研究成果，借鉴到计算机科学的故事。

泊松方程有一个非常美妙的定理，即如果指定了边界上的Dirichlet条件或者Neumann条件，那么泊松方程在区域内的解是唯一可确定的，因此当我们知道不确定区域内的密度函数，和边界上的$\alpha$值，我们就可以计算出这个区域内的$\alpha$值。于是乎，我们定义边界条件：
$$
\alpha_{\partial \Omega}=1 \quad p\in\Omega_F\\\alpha_{\partial \Omega}=0\quad p\in\Omega_B
$$
而密度函数是：
$$
\nabla\alpha=div(\frac{\nabla I}{F-B})
$$
随后我们就可以通过Gauss-Seidel算法来求解得到$\alpha$。

当然这里还有一个小问题是，F和B我们是不知道的，在这里采用了一种非常直观的方式，就是使用最近邻的方法，确定F和B的值，当然随后使用filter平滑F-B，来保证F-B的变化不会太突兀。

最后整个算法是一个迭代式的算法，通过多次迭代，来保证$\alpha$达到一个比较稳定的状态。

当然还有基于泊松方程的图像融合等算法，也是值得一看的，math is amazing!

最后的效果图是这样的：

 ![troll](C:\Users\manster\Documents\code\matting\poisson-matting\troll.png)

 ![trollAlpha](C:\Users\manster\Documents\code\matting\poisson-matting\trollAlpha.png)

## Bayes Matting

贝叶斯公式算是这个世界上应用最广泛的公式了，也是机器学习非常基础的一项技能。

matting问题可以直观地表示为如下形式：
$$
argmax_{F,B,\alpha}P(F,B,\alpha|C)=argmax_{F,B,\alpha}P(C|F,B,\alpha)P(F)P(B)P(\alpha)/P(C)
\\=argmax_{F,B,\alpha}L(C|F,B,\alpha)+L(F)+L(B)+L(
\alpha)
$$
L是取log之意，这个操作的意义在于：由于概率都是[0,1]，如果大量概率连乘，在计算机中的表示会变为0，同时由于P(C)是常量，因此可以忽略。

那么接下来的任务就是，对$L(C|F,B,\alpha),L(F),L(B),L(\alpha)$建立模型，建立的模型可以多种多样，只要reasonable即可。所有应用贝叶斯公式和最大后验概率（MAP）的方法，关键就在于对概率的定义。

比如下图的定义表示：对于预测的$\alpha$、F和B，它们融合的结果和真实的图像越接近，那么概率越高。
$$
L(C|F,B,\alpha)=-||C-\alpha F-(1-\alpha)B||^2/\sigma^2_C
$$
对于L(F)来讲，我们要建立的是前景颜色的概率分布，表示当前选定的F，概率有多大，在文章中，它使用一种聚类方法，来对前景的颜色分布建模。

对于L(B)来讲，文中说针对不同类型的matting问题，可以使用不同的定义，在这里，我们直接使用L(F)的定义。

对于$L(\alpha)$,这篇文章假设，$\alpha$的分布是平均的，因此我们可以在MAP中忽略这一项。

最后就是求解这个MAP问题了，对于凸优化问题，我们可以直接使用导数为0的方式求解。

当然由于$\alpha$会和F、B相乘，因此这个MAP问题比较难，所以文章采用两段法求解，先固定$\alpha$求解F、B，随后固定F、B，求解$\alpha$，迭代这个过程，直到结果比较稳定。

结果是这样的：

 ![gandalf](C:\Users\manster\Documents\code\matting\bayesian-matting\gandalf.png)

 ![gandalfTrimap](C:\Users\manster\Documents\code\matting\bayesian-matting\gandalfTrimap.png)

 ![gandalfAlpha](C:\Users\manster\Documents\code\matting\bayesian-matting\gandalfAlpha.png)

## Learning Based Digital Matting

研一的时候，上常虹老师的课，听她讲到semi-supervised learning时，我打了个激灵，卧槽，可以拿来做matting，跑回宿舍一查，就在想：“娘希匹，这帮孙子下手也忒快了！要是我早生几年blabla。”（著名键盘侠，张远同学——赵子承语）

半监督学习的核心在于：在给定标记数据的情况下，预测结果不止和标记数据有关，还和未标记数据有关，更精确地说，和数据的相对位置有关。

如果能建立一种$alpha$和像素点颜色的映射，那么给定一个像素点，通过它的颜色来预测$\alpha$，是一件多么美妙的事情啊！于是就有了本文。

本文有两个假设：

1、任何未知的像素的$\alpha$，是周围像素$\alpha$的线性组合，从而我们可以将这种线性组合关系表达为矩阵的形式，注意：在这里我们使用了半监督学习中相对位置的概念；

2、假设$\alpha$和该像素点的颜色向量呈线性相关关系；

于是乎，对于每一个点的$\alpha$，我们期望使用周围像素$\alpha$的线性组合来预测，这个线性组合的参数，又是通过学习得到的，这个学习的过程，就是建立$\alpha$和颜色特征向量之间相关关系的过程。

这个方法更美妙的一点在于，在建立$\alpha$和颜色特征向量的关系时，最后问题的解析解中出现了内积的形式，这时就可以引入Kernel Trick，将维度升高，去学更复杂的关系。

 ![troll](C:\Users\manster\Documents\code\matting\learning-based-matting\troll.png)

 ![trollTrimap](C:\Users\manster\Documents\code\matting\learning-based-matting\trollTrimap.png)

 ![trollAlpha](C:\Users\manster\Documents\code\matting\learning-based-matting\trollAlpha.png)

## KNN Matting

还记得大三时在人人上，看到冠一姐和黄神讨论SBTree时，知道了Chen Qifeng这个人，顿时膜拜了起来orz。后来在GraphiCon里听士玮兄回忆，Chen Qifeng带他们训练的往事，士玮兄讲了一句话，我很赞同：“发现以前的偶像居然跟我在同一领域，有点小激动！”很多时候，就是这些优秀的人，给了我勇闯新世界的勇气。

这篇文章虽然实现起来很简单，但是信息量却很大，作者在文中很细致地讨论了idea和motivation，以及operation，而且这篇文章的目标也不是一般的Natural Image Matting，还涉及到了multiple layer matting。

真要细细说来，不知何年何月，所以只是在此提及。

## 从Matting到Dehaze

上个月在知乎上看到有个问题，博士三年还没发paper是一种怎样的体验，底下有个回答讲了He Kaiming的故事。真伪不谈，但是从He Kaiming的文章中，我们明显可以看到matting问题的影子。

He Kaiming那篇著名的Single Image Haze Removal Using Dark Channel Prior，里面把包含雾霾的图像建模成这个样子：
$$
I(x)=J(x)t(x)+A(1-t(x))
$$
其中I是图像，J是场景辐射强度，A是global atmospheric light，t可以理解为相对权重，我们希望得到去雾后的图像，就是J。

可以看出，这个问题和matting一摸一样。He Kaiming这篇文章的贡献就在于，他发现了暗通道先验，而有了这个异常良好的性质，这个很像matting的问题就没那么病态了，通过一些假设，我们可以直接使用矩阵运算，得到J。

当然为了让最后的图像效果更好，He Kaiming又做了Guided Filter，这个工作的精髓就是，岭回归。

He Kaiming和Sun Jian的工作，从Poisson Matting，到Single Image Haze Removal Using Dark Channel Prior，到ResNet，给我的感觉就是，对问题的洞察极其深刻，使用的方法又极其简单，颇有一种重剑无锋，大巧不工的感觉。

## 结语

Matting对我影响很大，这个问题很基本也很难，于是前辈们使用了各种各样的假设去求解，这些工作让我大开眼界，极大地影响了我的研究思路。

当然，有关matting，还有很多漂亮的工作，我只很浅薄地介绍了我最喜欢的几个方法。至于matting的详细survey，请阅读Wang Jue的Homepage：http://www.juew.org/。