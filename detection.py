import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv3
from chainercv import utils
from chainercv.visualizations import vis_bbox

import cv2
import json


"""
voc_bbox_label_names = (
0    'aeroplane',
1    'bicycle',
2    'bird',
3    'boat',
4    'bottle',
5    'bus',
6    'car',
7    'cat',
8    'chair',
9    'cow',
10    'diningtable',
11    'dog',
12    'horse',
13    'motorbike',
14    'person',
15    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
"""

label_names = voc_bbox_label_names
model = YOLOv3(n_fg_class=len(label_names), pretrained_model='voc0712')

output = {}

frame_cnt = 0
frame_count = 0

cap = cv2.VideoCapture('2.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 5500)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    """
    cv2.imshow("a", frame)
    cv2.waitKey(30)
    print(frame_cnt)
    frame_count += 1
    continue
    """

    cv2.imwrite("1.jpg", frame)
    image_path = "1.jpg"
    img = utils.read_image(image_path, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(img, bbox, label, score, label_names=label_names)
    #plt.pause(.01)

    labels = label.tolist()
    print(labels)
    boxes = bbox.tolist()
    output[frame_cnt] = []
    idx = -1
    for label in labels:
        idx += 1
        # 5:bus, 6:car, 14:person
        if not label in [5, 6, 14]:
            continue
        
        output[frame_cnt].append(boxes[idx] + [label])

    #output[frame_cnt] = bbox.tolist()

    print(output[frame_cnt])
    frame_cnt += 1

    print(frame_cnt)
    if frame_cnt == 1000:
        break
    

with open("output_5500.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
