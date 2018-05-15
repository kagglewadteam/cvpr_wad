import os
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
import utils.custom_metrics as custom_metrics
"""
blocks for DCCN series: DCCN_ORI, DCCN_SYMM
"""

def denseBlock(l, k, lbda):
    def dense_block_instance(x):
        ins = [x, denseConv(k,3,lbda)(
                  denseConv(k,1, lbda)(x))]
        for i in range(l-1):
            ins.append(denseConv(k,3, lbda)(
                       denseConv(k,1, lbda)(Concatenate(axis=-1)(ins))))
        y = Concatenate(axis=-1)(ins)
        return y
    return dense_block_instance


def denseConv(k, kernel_size, lbda):
    return lambda x: Conv2D(filters=k,
                            kernel_size=2*(kernel_size,),
                            padding='same',
                            kernel_regularizer=regularizers.l2(lbda),
                            bias_regularizer=regularizers.l2(lbda))(
                     Activation('relu')(
                     BatchNormalization()(x)))

# Transition Layers
def transitionLayerPool(f, lbda):
    return lambda x: AveragePooling2D(pool_size=2*(2,))(
                     denseConv(f, 1, lbda)(x))


def transitionLayerTransposeUp(f, lbda):
    return lambda x: Conv2DTranspose(filters=f, kernel_size=(3, 3), strides=(2, 2),
                                     padding="same", kernel_regularizer = regularizers.l2(lbda))(
                     denseConv(f, 1, lbda)(x))



class DCCN_SYMM():
    '''
    __init__ function will build up the model with given hyperparams
    '''
    def __init__(self, in_shape, kls, ls, theta, k_0, lbda=0):
        self.in_shape = in_shape
        self.kls = kls
        self.ls = ls
        self.theta = theta
        self.k_0 = k_0


        in_ = Input(shape=in_shape, name='input_X')
        x = Conv2D(filters=k_0, kernel_size=(7, 7), strides=(2, 2), padding='same')(in_)  #k_0 = 32

        shortcuts = []
        kls_ls_list = list(zip(self.kls, self.ls))

        for k, l in kls_ls_list:            #ls for dccn_symm is [8,8,4,2], k is [16,16,32,64]
            x = denseBlock(l=l, k=k, lbda=lbda)(x)
            shortcuts.append(x)
            k_0 = int(round((k_0 + k * l) * theta))
            x = transitionLayerPool(f=k_0, lbda=lbda)(x)

        #add one dense conv at the bottleneck, shift the dense block for the decoder to make it symmetric
        x = denseBlock(l=1, k=128, lbda=lbda)(x)

        for k, l, shortcut in reversed(list(zip(self.kls, self.ls, shortcuts))):  #start from TLU then DB
            k_0 = int(shortcut.shape[-1])                #get the number of channels for the low level feature map
            print('k_0:', k_0)
            x = transitionLayerTransposeUp(f=k_0, lbda=lbda)(x)
            x = Add()([shortcut, x])
            x = denseBlock(l=l, k=k, lbda=lbda)(x)

        x = Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_regularizer = regularizers.l2(lbda))(x)
        out = Activation('softmax', name='output_Y')(x)
        self.model = Model(in_, out)

    '''
    compile model
    settings for true-positive-rate (TPR)
    '''
    def compile(self):
        cls = 8

        m1 = [custom_metrics.metric_tp(c) for c in range(cls)]
        for j, f in enumerate(m1):
            f.__name__ = 'm_tp_c' + str(j)

        m2 = [custom_metrics.metric_gt(c) for c in range(cls)]
        for k, f in enumerate(m2):
            f.__name__ = 'm_gt_c' + str(k)

        self.model.compile(optimizer='rmsprop',
                      loss=custom_metrics.jaccard_dist,
                      metrics=m1 + m2 + ['categorical_accuracy'] + [custom_metrics.jaccard_dist_discrete])
        print(' model compiled.')


    def train(self, gen_train, gen_valid, callbacks, config):
        print(' training...')
        histObj = self.model.fit_generator(generator=gen_train,
                                      epochs=config.epochs,
                                      steps_per_epoch=1750 // config.batch_size,
                                      validation_data=gen_valid,
                                      validation_steps=194 // config.batch_size,
                                      callbacks=callbacks)
        return histObj



