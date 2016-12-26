# Matting

Matting指的是从图像和视频中，分离出前景和背景。但是与以往的分割问题不同的是，Matting并不是硬分割，而是软分割。

Matting认为图像可以表达成：
$$
I=\alpha *back+(1-\alpha)*fore
$$
其中$\alpha$为不透明度。因此在只知道I的情况下，需要求解出其他三项，难度可想而知。

目前文件夹中实现的方法是tri-matting，作者是alpha通道的提出者匠白光。tri-matting在前景背景融合的图片外，还引入了背景图片，这样就可以使用最小二乘法，准确地计算得到前景和$\alpha$信息。

后面的研究，集中于单张图片的matting，不引入其他信息。

matting研究的领军人物有：Adobe的Wang Jue，Face++的Sun Jian，Stanford的Chen Qifeng等。

经典方法包括：Closed-form matting，KNN matting，Possion Matting等。

目前只实现了tri-matting，更多的方法待添加中...

