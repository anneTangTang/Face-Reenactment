## 一键实现人物A驱动人物B (Full Head Driven)

该项目是我的毕设项目，主要基于Deep Video Portraits实现了全头驱动的任务。
按照下面的步骤可以运行代码，代码的输入是两个视频（一个source video，一个target video），输出是一个被驱动的视频。

- download `BFM.mat` to `face3d/utils/Data/`, link:[TODO]
- install `ffmpeg` first
- 根据自己电脑的`cuda`和`driver`版本安装相应版本的`pytorch`和`torchvision`
- 直接运行`sh pipeline.sh [source_video] [target_video]`，即可运行全流程的代码
- 代码运行结束后，生成的驱动视频保存在`drive/drive.mp4`中

> 由于该模型针对不同人物需要重新训练，所以代码运行可能需要较长时间，例如几小时到几天。
> 代码运行总时长随着视频长度的增长而增加、
> 为了达到较好的生成效果，请确保target视频至少两分钟。


## 该项目在持续优化中，欢迎提出您的批评建议，我会非常乐于接受！谢谢！

