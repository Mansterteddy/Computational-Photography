# Dynamic Registratioon Implementation

这部分代码，来自于SIGGRAPH ASIA的一个tutorial，这个教程，依照我的理解，是ICP算法的一个补充，将ICP算法纳入了一个更大的框架，引入更多的先验因子，使得配准方法更接近真实情况，这个教程也有开源代码，我给出的是python实现，目前只实现了Rigid Registration，未来希望加入Non-Rigid Registration, Data-Based Registration。

主要思想就是：
$$
E_{reg}=E_{match}+E_{prior}
$$
其中$E_{match}$衡量的是目标模型和变换模型之间的差距，$E_{prior}$衡量的是变换模型和源模型之间，是否满足运动先验。