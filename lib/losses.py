import config as conf
import numpy as np
import tensorflow as tf
import keras.backend as K

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def unet_loss(img_size):
    def func(y_true, y_pred):
        from metrics import dice_coef
        d = dice_coef(y_true, y_pred)
        b = K.mean(K.binary_crossentropy(y_true, y_pred))
        loss = .5 * b - d
        loss = tf.Print(loss, [d], message='\nDC:\t')
        loss = tf.Print(loss, [b], message='CE:\t')
        loss = tf.Print(loss, [tf.shape(y_true)], message='Shape:\t', summarize=10)
        return loss
    return func

def yolo_loss(true_boxes, img_size):
    YOLO_GRID = img_size // 32
    def func(y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(YOLO_GRID), [YOLO_GRID]), (1, YOLO_GRID, YOLO_GRID, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [conf.YOLO_BATCH_SIZE, 1, 1, conf.BOX, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(conf.ANCHORS, [1,1,1,conf.BOX,2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class prob.
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * conf.COORD_SCALE

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * conf.NO_OBJECT_SCALE

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * conf.OBJECT_SCALE

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(conf.CLASS_WEIGHTS, true_box_class) * conf.CLASS_SCALE

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < conf.COORD_SCALE/2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, conf.WARM_UP_BATCHES),
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                       true_box_wh + tf.ones_like(true_box_wh) * np.reshape(conf.ANCHORS, [1,1,1,conf.BOX,2]) * no_boxes_mask,
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy,
                                       true_box_wh,
                                       coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy    = tf.clip_by_value( .5 * tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-8) , conf.YOLO_MIN_LOSS, conf.YOLO_MAX_LOSS)
        loss_wh    = tf.clip_by_value( .5 * tf.reduce_sum(tf.square(tf.sqrt(true_box_wh+1e-8) - tf.sqrt(pred_box_wh+1e-8))     * coord_mask) / (nb_coord_box + 1e-8) , conf.YOLO_MIN_LOSS, conf.YOLO_MAX_LOSS)
        loss_conf  = tf.clip_by_value( .5 * tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-8) , conf.YOLO_MIN_LOSS, conf.YOLO_MAX_LOSS)
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        """
        Debugging code
        """

        loss = tf.Print(loss, [loss_xy], message='\nLoss XY \t', summarize=10)
        loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=10)
        loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=10)
        loss = tf.Print(loss, [tf.shape(y_true)], message='Shape:\t', summarize=10)
        loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=10)

        return loss
    return func

