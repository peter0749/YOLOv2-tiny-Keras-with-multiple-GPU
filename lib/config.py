import numpy as np
# ROOT TO DATASET (DATA & TEST)
CLASSES = 80 # MS COCO dataset
CLASS_WEIGHTS = np.ones(CLASSES, dtype='float32')
CLASS_SCALE = 1.0
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# U-Net for semantic segmentation
U_NET_DIM = 96

# YOLO step-by-step ref:
# https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb
YOLO_DIM = 416 ## must be integer (odd number) * 32.
OBJECT_THRESHOLD = 0.6
NMS_THRESHOLD = 0.45
U_NET_THRESHOLD = 0.5
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
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

YOLO_CKPT = '../yolo_weights'
YOLO_PRETRAINED = None # '../yolov2-voc.weights'

U_NET_CKPT = '../unet_weights'

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

YOLO_OUT_DIR = './detection_output'
U_NET_OUT_DIR = './unet_out'

USE_XLA = False

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
