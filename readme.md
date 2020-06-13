# 简介
学习《三维点云分析》的课后习题练习代码

# Data
ModelNet40: 链接:https://pan.baidu.com/s/1Pblw4hlxy-mXfICesdYPbg  密码:bgth

# 编译
```
$ git clone https://github.com/QinZiwen/analyze_pointcloud.git
$ cd analyze_pointcloud
$ mkdir build && cd build
$ cmake ..
$ make
```

> 注意：编译的可执行程序在工程的bin下，library在工程lib目录下。

# 部分结果截图：
![](https://github.com/QinZiwen/analyze_pointcloud/blob/master/images/PCA/PCA.png)<br/>
![](https://github.com/QinZiwen/analyze_pointcloud/blob/master/images/PCA/downsample.png)<br/>
![点云中地面分割](https://github.com/QinZiwen/analyze_pointcloud/blob/master/images/Fitting/find_ground.png)<br/>
