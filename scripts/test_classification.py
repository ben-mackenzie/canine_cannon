import argparse
import os

import cv2
import numpy as np

import canine_cannon.yolo as ccyolo

YOLO_CFG_DIR = os.path.join(os.path.dirname(ccyolo.__file__), 'yolov3.cfg')
YOLO_LBS_DIR = os.path.join(os.path.dirname(ccyolo.__file__), 'yolov3.txt')
YOLO_WTS_DIR = os.path.join(os.path.dirname(ccyolo.__file__), 'yolov3.weights')

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
args = ap.parse_args()


# CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# CAP_PROP_POS_AVI_RATIO Relative position of the video file
# CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# CAP_PROP_FPS Frame rate.
# CAP_PROP_FOURCC 4-character code of codec.
# CAP_PROP_FRAME_COUNT Number of frames in the video file.
# CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# CAP_PROP_SATURATION Saturation of the image (only for cameras).
# CAP_PROP_HUE Hue of the image (only for cameras).
# CAP_PROP_GAIN Gain of the image (only for cameras).
# CAP_PROP_EXPOSURE Exposure (only for cameras).
# CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# CAP_PROP_WHITE_BALANCE Currently unsupported
# CAP_PROP_RECTIFICATION

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(YOLO_LBS_DIR, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(YOLO_WTS_DIR, YOLO_CFG_DIR)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.imshow("object detection", image)
cv2.waitKey()

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
