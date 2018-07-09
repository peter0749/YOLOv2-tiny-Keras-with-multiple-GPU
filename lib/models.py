import os
import gc
import config as conf
import losses
import metrics
import numpy as np
from utils import WeightReader

### Yolo model:

def get_yolo_model(img_size=conf.YOLO_DIM, gpus=1, load_weights=None, verbose=False):
    import tensorflow as tf
    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)
    from keras.models import Sequential, Model
    from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, Flatten, Dense, Lambda
    from keras.layers.advanced_activations import LeakyReLU
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    from keras.layers.merge import concatenate
    from keras.utils.training_utils import multi_gpu_model
    import keras.backend as K
    from weightnorm import AdamWithWeightnorm as Adam # weight normalization

    YOLO_GRID = conf.YOLO_GRID

    input_image = Input(shape=(img_size, img_size, 3))
    true_boxes  = Input(shape=(1, 1, 1, conf.TRUE_BOX_BUFFER , 4))

    # Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', kernel_initializer='he_normal', use_bias=False)(input_image)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', kernel_initializer='he_normal', use_bias=False)(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', kernel_initializer='he_normal', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23
    x = Conv2D(conf.BOX * (4 + 1 + conf.CLASSES), (1,1), strides=(1,1), padding='same', name='conv_23', kernel_initializer='he_normal')(x)
    output = Reshape((img_size//32, img_size//32, conf.BOX, 4 + 1 + conf.CLASSES))(x)

    # small hack to allow true_boxes to be registered when Keras build the model
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = Lambda(lambda args: args[0])([output, true_boxes])
    optimizer = Adam(**conf.YOLO_OPT_ARGS)
    with tf.device('/cpu:0'): ## prevent OOM error
        cpu_model = Model([input_image, true_boxes], output)
        cpu_model.compile(loss=losses.yolo_loss(true_boxes, img_size), optimizer=optimizer)
        if conf.YOLO_PRETRAINED is not None and os.path.exists(conf.YOLO_PRETRAINED): # load yolo pretrained weights
            weight_reader = WeightReader(conf.YOLO_PRETRAINED)
            weight_reader.reset()
            nb_conv = 22
            for i in range(1, nb_conv+1):
                conv_layer = cpu_model.get_layer('conv_' + str(i))

                if i < nb_conv:
                    norm_layer = cpu_model.get_layer('norm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta  = weight_reader.read_bytes(size)
                    gamma = weight_reader.read_bytes(size)
                    mean  = weight_reader.read_bytes(size)
                    var   = weight_reader.read_bytes(size)

                    weights = norm_layer.set_weights([gamma, beta, mean, var])

                if len(conv_layer.get_weights()) > 1:
                    bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            # Randomize weights of the last layer
            layer   = cpu_model.layers[-4] # the last convolutional layer
            weights = layer.get_weights()

            new_kernel = np.random.normal(size=weights[0].shape)/(YOLO_GRID*YOLO_GRID)
            new_bias   = np.random.normal(size=weights[1].shape)/(YOLO_GRID*YOLO_GRID)
            layer.set_weights([new_kernel, new_bias])

            if verbose:
                print('Loaded pretrained YOLO weights')

        if load_weights is not None and os.path.exists(load_weights):
            cpu_model.load_weights(load_weights)
            if verbose:
                print('Loaded weights')
    if gpus>=2:
        gpu_model = multi_gpu_model(cpu_model, gpus=gpus)
        gpu_model.compile(loss=losses.yolo_loss(true_boxes, img_size), optimizer=optimizer)
        return gpu_model, cpu_model
    return cpu_model, cpu_model
### end Yolo model

### U-Net:
def get_U_Net_model(img_size=conf.U_NET_DIM, gpus=1, load_weights=None, verbose=False):
    from keras.models import Model, load_model
    from keras.layers import Input, Add, Activation
    from keras.layers.core import Lambda
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.utils.training_utils import multi_gpu_model
    from keras import backend as K
    from weightnorm import AdamWithWeightnorm as Adam # weight normalization
    import tensorflow as tf

    IMG_WIDTH = img_size
    IMG_HEIGHT= img_size
    IMG_CHANNELS= 3

    def conv(f, k=3, act='selu'):
        return Conv2D(f, (k, k), activation=act, kernel_initializer='he_normal', padding='same')
    def _incept_conv(inputs, f, chs=[0.15, 0.5, 0.25, 0.1]):
        fs = [] # determine channel number
        for k in chs:
            t = max(int(k*f), 1) # at least 1 channel
            fs.append(t)

        fs[1] += f-np.sum(fs) # reminding channels allocate to 3x3 conv

        c1x1 = conv(fs[0], 1, act='linear') (inputs)
        c3x3 = conv(max(1, fs[1]//2), 1, act='selu') (inputs)
        c5x5 = conv(max(1, fs[2]//2), 1, act='selu') (inputs)
        cpool= MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same') (inputs)

        c3x3 = conv(fs[1], 3, act='linear') (c3x3)
        c5x5 = conv(fs[2], 5, act='linear') (c5x5)
        cpool= conv(fs[3], 1, act='linear') (cpool)

        output = concatenate([c1x1, c3x3, c5x5, cpool], axis=-1)
        return output

    def _res_conv(inputs, f, k=3): # very simple residual module
        channels = int(inputs.shape[-1])

        cs = _incept_conv(inputs, f)

        if f!=channels:
            t1 = conv(f, 1, 'linear') (inputs) # identity mapping
        else:
            t1 = inputs

        out = Add()([t1, cs]) # t1 + c2
        out = Activation('selu') (out)
        return out
    def pool():
        return MaxPooling2D((2, 2))
    def up(inputs):
        upsampled = Conv2DTranspose(int(inputs.shape[-1]), (2, 2), strides=(2, 2), padding='same') (inputs)
        return upsampled

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = _res_conv(inputs, 32, 3)
    c1 = _res_conv(c1, 32, 3)
    p1 = pool() (c1)

    c2 = _res_conv(p1, 64, 3)
    c2 = _res_conv(c2, 64, 3)
    p2 = pool() (c2)

    c3 = _res_conv(p2, 128, 3)
    c3 = _res_conv(c3, 128, 3)
    p3 = pool() (c3)

    c4 = _res_conv(p3, 256, 3)
    c4 = _res_conv(c4, 256, 3)
    p4 = pool() (c4)

    c5 = _res_conv(p4, 512, 3)
    c5 = _res_conv(c5, 512, 3)
    p5 = pool() (c5)

    c6 = _res_conv(p5, 512, 3)
    c6 = _res_conv(c6, 512, 3)

    u7 = up (c6)
    c7 = concatenate([u7, c5])
    c7 = _res_conv(c7, 512, 3)
    c7 = _res_conv(c7, 512, 3)

    u8 = up (c7)
    c8 = concatenate([u8, c4])
    c8 = _res_conv(c8, 256, 3)
    c8 = _res_conv(c8, 256, 3)

    u9 = up (c8)
    c9 = concatenate([u9, c3])
    c9 = _res_conv(c9, 128, 3)
    c9 = _res_conv(c9, 128, 3)

    u10 = up (c9)
    c10 = concatenate([u10, c2])
    c10 = _res_conv(c10, 64, 3)
    c10 = _res_conv(c10, 64, 3)

    u11 = up (c10)
    c11 = concatenate([u11, c1])
    c11 = _res_conv(c11, 32, 3)
    c11 = _res_conv(c11, 32, 3)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)
    optimizer = Adam(**conf.U_NET_OPT_ARGS)
    with tf.device('/cpu:0'): ## prevent OOM error
        cpu_model = Model(inputs=[inputs], outputs=[outputs])
        cpu_model.compile(loss=losses.unet_loss(img_size), metrics=[metrics.mean_iou], optimizer=optimizer)
        if load_weights is not None and os.path.exists(load_weights):
            cpu_model.load_weights(load_weights)
            if verbose:
                print('Loaded weights')
    if gpus>=2:
        gpu_model = multi_gpu_model(cpu_model, gpus=gpus)
        gpu_model.compile(loss=losses.unet_loss(img_size), metrics=[metrics.mean_iou], optimizer=optimizer)
        return gpu_model, cpu_model
    return cpu_model, cpu_model
### end U-Net model
