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
from utils import decode_netout, draw_boxes
import cv2
import seaborn as sns

video_name = str(sys.argv[1])
frame_skip = 1

LAST_CKPT_PATH = os.path.join(conf.YOLO_CKPT, 'last.hdf5')
CKPT_PATH = os.path.join(conf.YOLO_CKPT, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

print('Generating metadata...')
coco_valid = COCO(conf.VALID_ANNO)

if not os.path.exists(conf.YOLO_CKPT):
    os.makedirs(conf.YOLO_CKPT)

yolo_model, cpu_model = models.get_yolo_model(img_size=conf.YOLO_DIM, gpus=0, load_weights=LAST_CKPT_PATH, verbose=True)
GEN_CONF = conf.yolo_generator_config
GEN_CONF['BATCH_SIZE'] = 1
print('YOLO model loaded!')

videoCapture = cv2.VideoCapture(video_name)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('detect_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

labels = [ l['name'] for l in coco_valid.loadCats(conf.CLASS_IDS) ]
colors = np.clip(np.round(np.asarray(sns.color_palette("Set2", len(labels)))*255), 0, 255).astype(np.uint8)
dummy = np.empty((1, 1, 1, 1, conf.TRUE_BOX_BUFFER, 4))

success, frame = videoCapture.read()
frame_n = 0
while success:
    if frame_n % frame_skip == 0:

        s = time.time()
        resized_frame = cv2.resize(frame, (conf.YOLO_DIM, conf.YOLO_DIM), interpolation=cv2.INTER_AREA)
        YCrCb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
        YCrCb[...,0] = clahe.apply(YCrCb[...,0])
        resized_frame = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
        preprocessed_img = normalize(resized_frame[...,::-1].astype(np.float32))[np.newaxis,...]
        pred_netout = yolo_model.predict_on_batch([preprocessed_img, dummy])[0]
        boxes = decode_netout(pred_netout, conf.CLASSES, conf.OBJECT_THRESHOLD, conf.NMS_THRESHOLD, conf.ANCHORS)
        t = time.time()

        print('Detected objects: %d'%len(boxes))
        img  = draw_boxes(frame, boxes, labels, colors=colors)

        tick = t-s
        c = (0,255,0) if tick*fps<1 else (0,0,255)

        cv2.putText(img, '%.2fms'%(tick*1000), (img.shape[1]-img.shape[1]//6, img.shape[0]//12), cv2.FONT_HERSHEY_SIMPLEX, 2e-3 * img.shape[0], c, 2)
        cv2.imshow("detector", img)
        videoWriter.write(img)
    if cv2.waitKey(1)>=0:
        break
    success, frame = videoCapture.read()
    frame_n += 1

