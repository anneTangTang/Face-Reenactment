## Face Reenactment

This project is mainly based on DeepVideoPortraits to implement face reenactment, using 3D methods.
Follow the steps below, and you can run the project. The input consists of
one target video and one source video, and the output is a driven video.
 
- Download [BFM.mat](https://drive.google.com/file/d/1YwTD_xbjFuXH_FJTsNFs9uf14RFdfEK6/view?usp=sharing) to `face3d/utils/Data/`
- Install `ffmpeg`, `pytorch` and `torchvision`
- Run `sh pipeline.sh [source_video] [target_video]` in your command line
- The output video is saved in `drive/drive.mp4`

> Since the model needs to be trained from scratch for different characters, 
> it may take a long time to run the code, such as several hours or even several days.
> Running time increases with the length of the video.
> In order to achieve acceptable results, please ensure that the target video is at least two minutes.

#### The project is being continuously optimized. Your criticism and suggestions are welcome, and I will be very happy to accept it! Thank you!

