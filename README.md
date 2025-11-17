# Detecting suspicious people in crowds
## Pipeline
### Images extraction
File extract_frames.py extracts 2 frames per 1 second for a video.
### Image upscale
In file extract_skeletons.py I am using model ESPCN_x4.pb. For quicker upscale is used cv2 with cuda compatibility.

## First attempt
I tried using yolo models yolo8x-pose.pt and yolo11x-pose.pt for person detection and skeleton extraction. So called 1 stage.
It did solid job but even with the best tunning it fell behind in more crowded scenes as shown below.
![test_76_1.png](test_76_1.png)
![test_76_1_upscaled_2.png](test_76_1_upscaled_2.png)

## Second attempt
From first attempt I learned that yolo model is good for detecting people but worse in skeleton extraction in crowded scenes.
So I decided to use 2 stage detection. First stage will be yolo person detection. Second stage will be skeleton extraction
using VitPose model.