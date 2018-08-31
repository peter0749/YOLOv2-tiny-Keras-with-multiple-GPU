import os
import sys
import time
import config as conf
import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
if conf.USE_XLA:
    tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=tfconfig))
import numpy as np
import models
from pycocotools.coco import COCO
from utils import normalize
from utils import decode_netout, draw_bbox_and_masks
import cv2
import seaborn as sns

video_name = 0
frame_skip = 1

YOLO_LAST_CKPT_PATH = os.path.join(conf.YOLO_CKPT, 'last.hdf5')
UNET_LAST_CKPT_PATH = os.path.join(conf.U_NET_CKPT, 'last.hdf5')

print('Generating metadata...')
coco_valid = COCO(conf.VALID_ANNO)

if not os.path.exists(conf.YOLO_CKPT):
    os.makedirs(conf.YOLO_CKPT)

yolo_model = models.get_yolo_model(img_size=conf.YOLO_DIM, gpus=0, load_weights = YOLO_LAST_CKPT_PATH, verbose=True)[0]
unet_model = models.get_U_Net_model(img_size=conf.U_NET_DIM, gpus=0, load_weights = UNET_LAST_CKPT_PATH, verbose=True)[0]
print('YOLO model loaded!')

videoCapture = cv2.VideoCapture(video_name)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

labels = [ l['name'] for l in coco_valid.loadCats(conf.CLASS_IDS) ]
colors = np.clip(np.round(np.asarray(sns.color_palette("Set2", len(labels)))*255), 0, 255).astype(np.uint8)
dummy = np.empty((1, 1, 1, 1, conf.TRUE_BOX_BUFFER, 4))
obj_threshold = conf.OBJECT_THRESHOLD

success, frame = videoCapture.read()
frame_n = 0
while success:
    if frame_n % frame_skip == 0:

        s = time.time()
        resized_frame = cv2.resize(frame, (conf.YOLO_DIM, conf.YOLO_DIM), interpolation=cv2.INTER_AREA)
        resized_frame = cv2.resize(frame, (conf.YOLO_DIM, conf.YOLO_DIM), interpolation=cv2.INTER_AREA)
        YCrCb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
        YCrCb[...,0] = clahe.apply(YCrCb[...,0])
        resized_frame = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
        preprocessed_img = normalize(resized_frame[...,::-1].astype(np.float32))[np.newaxis,...]
        pred_netout = yolo_model.predict_on_batch([preprocessed_img, dummy])[0]
        boxes = decode_netout(pred_netout, conf.CLASSES, obj_threshold, conf.NMS_THRESHOLD, conf.ANCHORS)
        print('Detected objects: %d'%len(boxes))
        unet_inputs = []
        true_boxes = []
        for box in boxes:
            xmin  = np.clip(int((box.x - box.w/2) * size[0]), 0, size[0])
            xmax  = np.clip(int((box.x + box.w/2) * size[0]), 0, size[0])
            ymin  = np.clip(int((box.y - box.h/2) * size[1]), 0, size[1])
            ymax  = np.clip(int((box.y + box.h/2) * size[1]), 0, size[1])
            label = box.get_label()
            score = box.get_score()
            true_boxes.append((xmin,xmax,ymin,ymax,label,score))
            crop = frame[ymin:ymax,xmin:xmax]
            crop = np.zeros((conf.U_NET_DIM, conf.U_NET_DIM, 3), dtype=np.float32) if crop.shape[0]<=0 or crop.shape[1]<=0 else cv2.resize(crop, (conf.U_NET_DIM, conf.U_NET_DIM), interpolation=cv2.INTER_AREA)
            unet_inputs.append(crop)
        unet_inputs = normalize(np.asarray(unet_inputs, dtype=np.float32))
        pred_masks = unet_model.predict(unet_inputs, batch_size=conf.U_NET_BATCH_SIZE) if len(unet_inputs)>0 else []
        t = time.time()

        img  = draw_bbox_and_masks(frame, true_boxes, pred_masks, labels, colors=colors)
        tick = t-s
        c = (0, 255, 0) if tick*fps < 1 else (0, 0, 255)

        cv2.putText(img, '%.2fms'%(tick*1000), (img.shape[1]-img.shape[1]//6, img.shape[0]//12), cv2.FONT_HERSHEY_SIMPLEX, 2e-3 * img.shape[0], c, 2)
        cv2.imshow("detector", img)
        key = cv2.waitKey(1)
    if key==43: # 43: +; 45: -
        obj_threshold += 0.05
    elif key==45:
        obj_threshold -= 0.05
    elif key>0:
        break
    success, frame = videoCapture.read()
    frame_n += 1

