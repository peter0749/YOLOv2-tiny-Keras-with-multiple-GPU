import numpy as np
# ROOT TO DATASET (DATA & TEST)
TRAIN_ANNO = '/hdd/dataset/MS-COCO/annotations/instances_train2017.json'
TRAIN_IMG = '/hdd/dataset/MS-COCO/train2017'
VALID_ANNO = '/hdd/dataset/MS-COCO/annotations/instances_val2017.json'
VALID_IMG = '/hdd/dataset/MS-COCO/val2017'
TEST_IMG = '/hdd/dataset/MS-COCO/test2017'
TEST_ANNO = '/hdd/dataset/MS-COCO/annotations/image_info_test-dev2017.json'
CLASSES = 80 # MS COCO dataset
CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
assert len(CLASS_IDS)==CLASSES
ID_MAP = dict()
for n, cat in enumerate(CLASS_IDS):
    ID_MAP[cat] = n
CLASS_WEIGHTS = np.ones(CLASSES, dtype='float32')
CLASS_SCALE = 1.0
SUBMISSION= '/hdd/dataset/MS-COCO/sub.csv'

# U-Net for semantic segmentation
U_NET_DIM = 96

# YOLO step-by-step ref:
# https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb
YOLO_DIM = 416 ## must be integer (odd number) * 32.
OBJECT_THRESHOLD = 0.3
NMS_THRESHOLD = 0.35
U_NET_THRESHOLD = 0.5
ANCHORS = [0.53,0.80, 1.71,2.36, 2.90,6.45, 6.34,3.79, 9.03,9.75]
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
WARM_UP_BATCHES  = 1000
TRUE_BOX_BUFFER  = 50

YOLO_DRAW_LINE_W = 2
YOLO_SHOW_CONF = True

YOLO_USE_MULTI_GPU=2
U_NET_USE_MULTI_GPU=0

YOLO_BATCH_SIZE=8  ## each gpus's batch size = YOLO_BATCH_SIZE / YOLO_USE_MULTI_GPU
U_NET_BATCH_SIZE=8

GENERATOR_WORKERS=10

YOLO_EPOCHS=180
U_NET_EPOCHS=180

YOLO_CH_DIM_EPOCHS=5
U_NET_CH_DIM_EPOCHS=1

YOLO_CKPT = '/hdd/dataset/MS-COCO/yolo_weights'
YOLO_PRETRAINED = None # '../yolov2-voc.weights'

U_NET_CKPT = '/hdd/dataset/MS-COCO/unet_weights'

YOLO_OPT_ARGS = {
    'lr'              : 2e-4,
    #'clipvalue'       : 0.1 ,
    #'clipnorm'        : 1.0 ,
}
U_NET_OPT_ARGS = {
    'lr'              : 1e-3,
}

YOLO_MIN_LOSS = 0
YOLO_MAX_LOSS = 10 # This prevent nans. If your loss is not chaning, then set a higher value.

YOLO_EARLY_STOP = 50
U_NET_EARLY_STOP = 50

YOLO_OUT_DIR = '/hdd/dataset/MS-COCO/detection_output'
U_NET_OUT_DIR = '/hdd/dataset/MS-COCO/unet_out'

USE_XLA = True

### !!! DO NOT EDIT THE CONFIGURATION BELOW !!! ###

BOX = int(len(ANCHORS) // 2) # number of anchorboxes, default:5
YOLO_GRID= int(YOLO_DIM // 32)  # 19
yolo_generator_config = {
    'IMAGE_H'         : YOLO_DIM,
    'IMAGE_W'         : YOLO_DIM,
    'GRID_H'          : YOLO_GRID,
    'GRID_W'          : YOLO_GRID,
    'BOX'             : BOX,
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : YOLO_BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
    'CLASSES'         : CLASSES
}

unet_generator_config = {
    'IMAGE_H'         : U_NET_DIM,
    'IMAGE_W'         : U_NET_DIM,
    'BATCH_SIZE'      : U_NET_BATCH_SIZE,
}
