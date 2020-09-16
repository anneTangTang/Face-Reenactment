## 一键实现人物A驱动人物B
- download BFM.mat to face3d/utils/Data/ , link:
- install `ffmpeg` first
- 根据自己电脑的`cuda`和`driver`版本安装相应版本的`pytorch`和`torchvision`
- 直接运行`sh pipeline.sh [source_video] [target_video]`，即可运行全流程的代码
- 代码运行结束后，生成的驱动视频保存在`drive/drive.mp4`中

> 由于该模型针对不同人物需要重新训练，所以代码运行可能需要较长时间，例如几小时到几天。
> 代码运行总时长随着视频长度的增长而增加、
> 为了达到较好的生成效果，请确保target视频至少两分钟。
