# A simple python implementation of IOU tracker on video
In this code we aimed to track objects from a detection model. For the tracker part we use IOU tracker.
This code requires a detection model which gives you detected bounding boxes from a object that you want. So before applying the tracker, first load your detection model and feed the detected bboxes in each frame to the tracker (`Lines 74-78 in example.py`).

This tracker will compute the iou value between two detected bounding boxes from the past frame and the current frame. If the threshold of iou will be grater than 0.5 then the id of the current bounding box remain the same of the past bounding box, otherwise it will get a new id.

## Running the code
<br>for running the code, one can easily run the `python example.py` the container:
>python example.py


## Parameters in the config file:
1. input_video_path: path and name of input video
2. output_video_path: path and name of output video
3. model_path: path and name of your detection model
