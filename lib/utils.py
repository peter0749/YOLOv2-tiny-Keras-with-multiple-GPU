import os
import copy
import tensorflow as tf
import copy
import cv2
import numpy as np
import config as conf
from keras.callbacks import Callback

class BoundBox:
    def __init__(self, x, y, w, h, c = None, class_probs = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        self.c     = c
        self.cla_pr = class_probs
        self.label = None
        self.scr = None
    def get_label(self):
        if self.label is None:
            self.label = np.argmax(self.cla_pr)
        return self.label
    def get_score(self):
        if self.scr is None:
            self.scr = self.cla_pr[self.get_label()]
        return self.scr

def normalize(image):
    return image.astype(np.float32) / 255.

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2

    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1.w * box1.h + box2.w * box2.h - intersect

    return float(intersect) / union

def draw_boxes(image_, boxes, labels): ## channel order: RGB / BGR
    image = copy.deepcopy(image_)
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), conf.YOLO_DRAW_LINE_W)
        if conf.YOLO_SHOW_CONF:
            cv2.putText(image,
                        str(labels[box.get_label()]) + ' ' + str(box.get_score()),
                        (xmin, ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        (0,255,0), 2)
    return image

### modified version for binary classification
### ref. from https://github.com/experiencor/basic-yolo-keras/blob/master/utils.py
def decode_netout(netout, nb_class, obj_threshold, nms_threshold, anchors):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4]) # confidence
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                classes = netout[row,col,b,5:]
                if np.sum(classes)<=0:
                    continue
                x, y, w, h, confidence = netout[row,col,b,:5]

                x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height

                box = BoundBox(x, y, w, h, confidence, classes)
                boxes.append(box)

    # suppress non-maximal boxes, NMS on boxes

    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.cla_pr[c] for box in boxes])))
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].cla_pr[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].cla_pr[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes

## ref: https://github.com/keras-team/keras/issues/8463
class multi_gpu_ckpt(Callback):
    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(multi_gpu_ckpt, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

# Run-length encoding reference from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def get_rles(lab_img, cutoff=0.5):
    for i in range(1, lab_img.max() + 1):
        test =  rle_encoding(lab_img == i)
        if len(test)==0:
            continue
        yield test

# reference from basic-yolo-keras (https://github.com/experiencor/basic-yolo-keras/blob/master/utils.py)
class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4
