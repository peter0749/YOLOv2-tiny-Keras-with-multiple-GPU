import os
import gc
import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=tfconfig))
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import keras.backend as K
import models
import reader
import config as conf
from sklearn.model_selection import train_test_split
from utils import normalize, multi_gpu_ckpt
from generators import YOLO_BatchGenerator
import matplotlib.pyplot as plt
from utils import decode_netout, draw_boxes

LAST_CKPT_PATH = os.path.join(conf.YOLO_CKPT, 'last.hdf5')
CKPT_PATH = os.path.join(conf.YOLO_CKPT, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

print('Generating metadata...')
# train_imgs, coco_train = reader.dataset_filepath(conf.TRAIN_IMG, conf.TRAIN_ANNO)
val_imgs, coco_valid = reader.dataset_filepath(conf.VALID_IMG, conf.VALID_ANNO)

if not os.path.exists(conf.YOLO_CKPT):
    os.makedirs(conf.YOLO_CKPT)

yolo_model, cpu_model = models.get_yolo_model(img_size=conf.YOLO_DIM, gpus=0, load_weights=LAST_CKPT_PATH, verbose=True)
GEN_CONF = conf.yolo_generator_config
GEN_CONF['BATCH_SIZE'] = 1
valid_batch = YOLO_BatchGenerator(val_imgs, GEN_CONF, shuffle=False, jitter=False, norm=normalize)
print('YOLO model loaded!')

inputs = valid_batch.__getitem__(np.random.randint(len(valid_batch)))[0][0]
dummy = np.empty((1, 1, 1, 1, conf.TRUE_BOX_BUFFER, 4))
demo = yolo_model.predict([inputs, dummy])[0]
out  = decode_netout(demo, conf.CLASSES, conf.OBJECT_THRESHOLD, conf.NMS_THRESHOLD, conf.ANCHORS)
print('Detected objects: %d'%len(out))

labels = [ l['name'] for l in coco_valid.loadCats(conf.CLASS_IDS) ]
img  = draw_boxes(np.round(np.clip(inputs[0]*255,0,255)).astype(np.uint8), out, labels)
plt.imshow(img)
plt.show()
