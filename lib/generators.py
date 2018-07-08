import os
import glob
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence, to_categorical
from utils import BoundBox, normalize, bbox_iou
from reader import dataset_filepath

### YOLO generator
class YOLO_BatchGenerator(Sequence):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       jitter=True,
                       norm=None):

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        ]),
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASSES']))                # desired network output

        for instance_count, train_instance in enumerate(self.images[l_bound:r_bound]):
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)

            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.clip(np.floor(center_x), 0, self.config['GRID_W']-1))
                    grid_y = int(np.clip(np.floor(center_y), 0, self.config['GRID_H']-1))

                    center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                    center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell

                    box = [center_x, center_y, center_w, center_h]

                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou     = -1

                    shifted_box = BoundBox(0,
                                           0,
                                           center_w,
                                           center_h)

                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou    = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor = i
                            max_iou     = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 5: ] = to_categorical(int(obj['class']), self.config['CLASSES'])

                    # assign the true box to b_batch
                    b_batch[instance_count, 0, 0, 0, true_box_index] = box

                    true_box_index += 1
                    true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            x_batch[instance_count] = self.norm(img)

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['image']
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)[...,:3]
        image = image[...,::-1] ## BGR -> RGB

        if image is None: print('Cannot find ' + str(image_name))

        h, w, c = image.shape
        assert h==train_instance['height'] and w==train_instance['width']
        all_objs = copy.deepcopy(train_instance['masks'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin

        return image, all_objs
### end YOLO generator

### U-Net generator ###
class U_NET_BatchGenerator(Sequence):
    def __init__(self, images,
                       coco,
                       config,
                       shuffle=True,
                       jitter=True,
                       norm=None):
        self.coco = coco

        self.images = [] # pairs of (img, mask)
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.Affine(
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=0, # if mode is constant, use a cval between 0 and 255
                    mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03)))
            ],
            random_order=True
        )

        for img in images:
            for mask in copy.deepcopy(img['masks']):
                mask['image'] = img['image']
                mask['width'] = img['width']
                mask['height'] = img['height']
                self.images.append(mask) # list of: {'image':filepath, 'mask':filepath, width, height, xmax, ymax, ...}

        if self.shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        y_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1))                # desired network output

        for instance_count, train_instance in enumerate(self.images[l_bound:r_bound]):
            # augment input image and fix object's position and size
            img, lab = self.aug_image(train_instance, jitter=self.jitter)

            x_batch[instance_count,...]   = self.norm(img)
            y_batch[instance_count,...,0] = lab

        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['image']
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)[...,:3]
        assert image is not None
        image = image[...,::-1] ## BGR -> RGB

        annId = train_instance['id']
        mask = self.coco.annToMask(self.coco.loadAnns(annId)[0])

        assert mask is not None
        if mask.ndim==2:
            mask  = np.expand_dims(mask, -1)

        ymin, ymax = train_instance['ymin'], train_instance['ymax']
        xmin, xmax = train_instance['xmin'], train_instance['xmax']

        if jitter:
            min_size = 3
            croph, cropw = max(min_size, np.random.uniform(0.9, 1.1)*(ymax-ymin)), max(min_size, np.random.uniform(0.9, 1.1)*(xmax-xmin)) ## random scale
            xmin = np.clip(np.random.uniform(-0.1, 0.1) * cropw + xmin, 0, image.shape[1]-min_size) ## random crop
            ymin = np.clip(np.random.uniform(-0.1, 0.1) * croph + ymin, 0, image.shape[0]-min_size)
            xmax = xmin+cropw
            ymax = ymin+croph

        image = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        mask  =  mask[int(ymin):int(ymax), int(xmin):int(xmax)]

        h, w, c = image.shape

        if jitter:
            seq_det = self.aug_pipe.to_deterministic()
            image = seq_det.augment_image(image)
            mask  = seq_det.augment_image( mask)

        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W'])) # shape: (IMAGE_H, IMAGE_W, 3)
        mask  = (cv2.resize(np.squeeze(mask) , (self.config['IMAGE_H'], self.config['IMAGE_W']))>.5).astype(np.float32) # shape: (IMAGE_H, IMAGE_W)

        return image, mask
### end U-Net generator ###
