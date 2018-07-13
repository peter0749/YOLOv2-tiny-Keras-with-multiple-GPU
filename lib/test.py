import os
import json
import config as conf
import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
if conf.USE_XLA:
    tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=tfconfig))
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import models
from utils import normalize
from utils import decode_netout
import cv2
from tqdm import tqdm

YOLO_LAST_CKPT_PATH = os.path.join(conf.YOLO_CKPT, 'last.hdf5')
UNET_LAST_CKPT_PATH = os.path.join(conf.U_NET_CKPT, 'last.hdf5')

print('Generating metadata...')
coco_test = COCO(conf.TEST_ANNO)

if not os.path.exists(conf.YOLO_CKPT):
    os.makedirs(conf.YOLO_CKPT)

yolo_model = models.get_yolo_model(img_size=conf.YOLO_DIM, gpus=0, load_weights = YOLO_LAST_CKPT_PATH, verbose=True)[0]
unet_model = models.get_U_Net_model(img_size=conf.U_NET_DIM, gpus=0, load_weights = UNET_LAST_CKPT_PATH, verbose=True)[0]
print('YOLO model loaded!')

dummy = np.empty((1, 1, 1, 1, conf.TRUE_BOX_BUFFER, 4))
final_results = []

imgIds = coco_test.getImgIds()
annotation_cnt = 0
for imageId in tqdm(imgIds, total=len(imgIds)): # all images
    image = coco_test.loadImgs(imageId)[0]
    image_path = os.path.join(conf.TEST_IMG, image['file_name'])
    width, height = image['width'], image['height']
    img_norm = normalize(cv2.imread(image_path))
    img_rs   = cv2.resize(img_norm, (conf.YOLO_DIM, conf.YOLO_DIM), interpolation=cv2.INTER_AREA)
    yolo_out = yolo_model.predict_on_batch([img_rs[np.newaxis,...], dummy])[0]
    boxes    = decode_netout(yolo_out, conf.CLASSES, conf.OBJECT_THRESHOLD, conf.NMS_THRESHOLD, conf.ANCHORS)
    unet_inputs = []
    true_boxes = []
    for box in boxes:
        xmin  = np.clip(int((box.x - box.w/2) * width),  0, width)
        xmax  = np.clip(int((box.x + box.w/2) * width),  0, width)
        ymin  = np.clip(int((box.y - box.h/2) * height), 0, height)
        ymax  = np.clip(int((box.y + box.h/2) * height), 0, height)
        label = box.get_label()
        score = box.get_score()
        box_w, box_h = xmax-xmin, ymax-ymin
        crop = img_norm[ymin:ymax,xmin:xmax]
        crop = np.zeros((conf.U_NET_DIM, conf.U_NET_DIM, 3), dtype=np.float32) if crop.shape[0]<=0 or crop.shape[1]<=0 else cv2.resize(crop, (conf.U_NET_DIM, conf.U_NET_DIM), interpolation=cv2.INTER_AREA)
        unet_inputs.append(crop)
        true_boxes.append((xmin,ymin,box_w,box_h,label,score))
    unet_inputs = np.asarray(unet_inputs, dtype=np.float32)
    pred_masks = unet_model.predict(unet_inputs, batch_size=conf.U_NET_BATCH_SIZE) if len(unet_inputs)>0 else np.array([])
    for box, mask in zip(true_boxes, pred_masks):
        x, y, w, h, label, score = box
        binary_mask_crop = (cv2.resize(mask[...,0], (w,h), interpolation=cv2.INTER_LINEAR)>conf.U_NET_THRESHOLD).astype(np.uint8)
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = binary_mask_crop
        rle = maskUtils.encode(np.asfortranarray(full_mask))
        annotation = {
                'id': annotation_cnt,
                'image_id': image['id'],
                'category_id': conf.CLASS_IDS[label],
                'segmentation': rle,
                'area': int(np.sum(binary_mask_crop)),
                'bbox': [x, y, w, h],
                'iscrowd': 1,
                'score': float(score)
        }
        final_results.append(annotation)
        annotation_cnt += 1
print('Saving results...')
with open(conf.SUBMISSION, 'w') as result_fp:
    result_fp.write(json.dumps(final_results))
print('Done.')
