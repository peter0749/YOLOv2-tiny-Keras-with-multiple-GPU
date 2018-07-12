import os
import sys
import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=tfconfig))
import numpy as np
import models
import reader
import config as conf
from utils import normalize
from utils import decode_netout, draw_boxes
import cv2

video_name = str(sys.argv[1])
frame_skip = 1

LAST_CKPT_PATH = os.path.join(conf.YOLO_CKPT, 'last.hdf5')
CKPT_PATH = os.path.join(conf.YOLO_CKPT, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

print('Generating metadata...')
coco_valid = reader.dataset_filepath(conf.VALID_IMG, conf.VALID_ANNO)[1]

if not os.path.exists(conf.YOLO_CKPT):
    os.makedirs(conf.YOLO_CKPT)

yolo_model, cpu_model = models.get_yolo_model(img_size=conf.YOLO_DIM, gpus=0, load_weights=LAST_CKPT_PATH, verbose=True)
GEN_CONF = conf.yolo_generator_config
GEN_CONF['BATCH_SIZE'] = 1
print('YOLO model loaded!')

videoCapture = cv2.VideoCapture(video_name)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (conf.YOLO_DIM, conf.YOLO_DIM)
videoWriter = cv2.VideoWriter('detect_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), float(fps/frame_skip), size)

labels = [ l['name'] for l in coco_valid.loadCats(conf.CLASS_IDS) ]
dummy = np.empty((1, 1, 1, 1, conf.TRUE_BOX_BUFFER, 4))

success, frame = videoCapture.read()
frame_n = 0
while success:
    if frame_n % frame_skip == 0:
        resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        preprocessed_img = normalize(resized_frame[...,::-1].astype(np.float32))[np.newaxis,...]
        pred_netout = yolo_model.predict_on_batch([preprocessed_img, dummy])[0]
        boxes = decode_netout(pred_netout, conf.CLASSES, conf.OBJECT_THRESHOLD, conf.NMS_THRESHOLD, conf.ANCHORS)
        print('Detected objects: %d'%len(boxes))
        img  = draw_boxes(resized_frame, boxes, labels)
        cv2.imshow("detector", img)
        videoWriter.write(img)
    if cv2.waitKey(1)>=0:
        break
    success, frame = videoCapture.read()
    frame_n += 1

