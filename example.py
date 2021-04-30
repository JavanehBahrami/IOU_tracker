import cv2
import os

from detector import DetectNet
from tracker import IOUTracker


def read_frames(video_path):
    clip_list = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, num_frames):
        ret, frame = cap.read()
        if ret:
            frame_ch = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_list.append(frame_ch)

    return clip_list, width, height, fps


def draw_bboxes(image, id_list, bbox_list):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, confidences, class_ids, track_ids = [], [], [], []

    for track_id, bbox in zip(id_list, bbox_list):
        bboxes.append(bbox)
        track_ids.append(track_id)

    clr = (0, 0, 255)
    for bb, tid in zip(bboxes, track_ids):
        cv2.rectangle(image, (bb[0], bb[1]),
                      (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)

        label = '_id:'+str(tid)

        (label_width, label_height), baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        y_label = max(bb[1], label_height)

        cv2.rectangle(image, (bb[0], y_label - label_height),
                      (bb[0] + label_width, y_label + baseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label, (bb[0], y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

    return image


def write_video(out, frame):
    out.write(frame)

    return


def process_video(input_video_path, detection_model):
    frames_list, w, h, fps = read_frames(input_video_path)

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_draw_list = []
    out = cv2.VideoWriter('./output_video.avi', fourcc, fps,
                          (w, h), True)

    iou_tracker = IOUTracker()
    for image in frames_list:

        bbox_list, confidence_list, class_id_list = detection_model.predict(
            image)

        if len(bbox_list) > 0:
            id_list = iou_tracker.track_objects(bbox_list)
            print('id_list = ', id_list)
            image = draw_bboxes(image, id_list, bbox_list)

        write_video(out, image)

    return


if __name__ == '__main__':
    input_video_path = "./video_test.mp4"

    # load your custom detection model (for example for a trt detection model)
    model_path = './model.trt'
    detection_model = DetectNet(model_path)

    process_video(input_video_path, detection_model)
