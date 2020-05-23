# Style-transfer_GAN
## 任务信息
  本项目基于Pix2Pix模型和CycleGAN网络分别实现了不同场景下对不同目标的视觉风格化转化。通过实验可以明显的观察到两种视觉风格化转化方法在不同场景下各自的优缺点，同时调用 GPU 提升模型训练速度。通过训练集图像对算法框架进行训练，并对训练的模型进行检测测试。
## 数据集
  本次实验中，CycleGAN网络基于vangogh2photo数据集进行训练，将梵高风格的油画与现实风景画的风格相互转换。Pix2Pix模型则基于facades图像进行训练和检测，在有监督的情况下进行抽象风格向现实风格的转换。
## 运行环境
1.硬件环境

名称|配置
----|----
处理器|Intel Core i7-6700HQ CPU@2.60GHz
显卡|NVIDIA GeForce GTX 960M
物理内存|8.00 GB
操作系统|Windows 10 家庭中文版，64位
编程语言|Python 3.6.5

2.框架

算法|学习框架
----|----
CycleGAN|tensorflow-gpu 1.10.0
Pix2Pix|tensorflow-gpu 1.10.0

3.其他的相关软件和工具库主要有 CUDA9.0、NVIDIA cuDNN 7.0、Anaconda4.8.3等


***本实验所用到的数据集、得到的训练结果及训练成熟的模型文件不包含在本项目中***

***链接为：https://pan.baidu.com/s/1JvMzehuzees_h1bEUaWIfA 提取码：kdcv***
