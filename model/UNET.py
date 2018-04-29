# external import
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from keras.applications.resnet50 import preprocess_input
import keras.preprocessing.image as KPImage
from PIL import Image
from skimage.io import imread
from keras.backend import tensorflow_backend as KTF
from keras.utils import multi_gpu_model

# internal import
from utils.loss import *
from utils.gen_vis import TensorBoardWrapper

class pil_image_awesome():
    @staticmethod
    def open(in_path):
        MIN_OBJ_VAL = 1000
        if 'instanceIds' in in_path:
            # we only want to keep the positive labels not the background
            in_img = imread(in_path)
            if in_img.ndim == 3:
                in_img = in_img[:, :, 0]
            return Image.fromarray((in_img > MIN_OBJ_VAL).astype(np.float32))
        else:
            return Image.open(in_path)

    fromarray = Image.fromarray


# flow from train directory
def train_generator(image_datagen, mask_datagen, batch_size):
    seed = 1
    image_flow = image_datagen.flow_from_directory('/home/liuyn/masterthesis/kaggle_sampledata/train_color', batch_size=batch_size, target_size=(384, 384), color_mode="rgb", class_mode=None, seed=seed)
    label_flow = mask_datagen.flow_from_directory('/home/liuyn/masterthesis/kaggle_sampledata/train_label', batch_size=batch_size, target_size=(384, 384), color_mode="grayscale", class_mode="categorical", seed=seed)
    return zip(image_flow,label_flow)
    '''
    while True:
        image_batch = next(image_flow)
        label_batch = next(label_flow)
        yield image_batch, label_batch
    '''



# flow from validation directory
def val_generator(image_datagen, mask_datagen, batch_size):
    seed = 1
    image_flow = image_datagen.flow_from_directory('/home/liuyn/masterthesis/kaggle_sampledata/val_color', batch_size=batch_size, target_size=(384, 384), color_mode="rgb", class_mode=None, seed=seed)
    label_flow = mask_datagen.flow_from_directory('/home/liuyn/masterthesis/kaggle_sampledata/val_label', batch_size=batch_size, target_size=(384, 384), color_mode="grayscale", class_mode="categorical", seed=seed)
    return zip(image_flow,label_flow)
    '''
    while True:
        image_batch = next(image_flow)
        label_batch = next(label_flow)
        yield image_batch, label_batch
    '''


def create_model():
    # Build U-Net model
    inputs = Input((384, 384, 3))
    s = BatchNormalization()(inputs) # we can learn the normalization step
    s = Dropout(0.5)(s)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(8, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model


def train(cfg, train_generator, val_generator):
    #with tf.device('/cpu:0'):  #save parameters in cpu to avoid chaos
    model = create_model()
    #parallel_model = multi_gpu_model(model,gpus=3)
    model.compile(optimizer='adam',
                  loss=dice_coef_loss,
                  metrics=[dice_coef, 'categorical_accuracy', 'mse'])

    weights_file = cfg['model'] + '.h5'

    # define callback function
    callback_list = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1))
    callback_list.append(ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, period=1, save_best_only=True, save_weights_only=True))
    callback_list.append(TensorBoard(log_dir='./tensorboard/log', write_images=False, histogram_freq=0))
    #callback_list.append(TensorBoardWrapper(val_generator, nb_steps=1, log_dir='./tensorboard/log', histogram_freq=1,
     #                          batch_size=cfg['batch_size'], write_graph=True, write_grads=True))  #visualize model 
    # train model
    model.fit_generator(train_generator,
                        steps_per_epoch=1750 // cfg['batch_size'],
                        validation_data=val_generator,
                        validation_steps=194 // cfg['batch_size'],
                        epochs=cfg['epoch'],
                        callbacks=callback_list)


def run(cfg):
    #allocate GPUS sources
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess =tf.Session(config = config)
    KTF.set_session(sess)
    # handle the 16bit numbers
    KPImage.pil_image = pil_image_awesome

    # data augmentation options
    data_gen_args = dict(horizontal_flip=True, zoom_range=0.05)

    if cfg['augmentation']:
        image_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
    else:
        image_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        mask_datagen = ImageDataGenerator()

    # train model
    train(cfg, train_generator(image_datagen, mask_datagen, cfg['batch_size']), val_generator(image_datagen, mask_datagen, cfg['batch_size']))

