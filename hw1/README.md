hw1
===

- 实验目的：

熟悉RGB、HSV、CIELab颜色空间、直方图。

利用HSV（或RGB，CIELab）颜色直方图，进行基于颜色的图像检索

- 实验步骤：

1、取四张图片，画出三种模型下的颜色直方图
RGB：各通道256级
HSV：H通道灰度级180，S通道灰度级256
CIElab：忽略亮度L通道，并且将a及b通道量化为50个区间

2、采用RGB, HSV量化，累加直方图，CCV，Centering refinement，Color Coherence Distance Refinement进行图像检索。

- 注意

    - 相似性度量：采用Correlation、直方图相交Intersection、开方统计Chi-Square statistic、巴式距离Bhattacharyya四种方法进行相似性度量

    - HSV的区域2分为8个灰度区

    - 采用open CV 和Matlab或其它方法均可
    
    - 注意颜色空间的转换方式以及图像文件的读取