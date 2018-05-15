import argparse
import os


# Just for settings, could be changed
#**************************************
# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import keras
import keras.backend as K
from keras import callbacks as cb
from keras.models import Sequential, Model
from keras.layers import *
K.set_image_data_format = 'channels_last'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))
#***************************************

# import own scripts

import utils.history as history
from model.dccn_symm import DCCN_SYMM
from utils.DataPreprocessing import dataset_split
from model.UNET import train_generator, val_generator
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

#hyperparams setting, needs to be changed
#*****************************************************

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default= '../patient-paths/patients_1_5T.pkl', type = str)   #need change
parser.add_argument('--label_dir', default= '../patient-paths/patients_1_5T.pkl', type = str)   #need change
parser.add_argument('--model_path', default= '/home/d1251/no_backup/d1251/models/', type = str) #change
parser.add_argument('--history_path', default= '/home/d1251/no_backup/d1251/histories/', type = str)    #change

parser.add_argument('--in_size', default = 32, type = int)
parser.add_argument('--k', default = [16, 16, 32, 64], type = int, help= 'growth rate of dense block') #don't need to change
parser.add_argument('--ls', default = [8,8,4,2], type = list, help = 'layers in dense blocks') #don't need to change
parser.add_argument('--theta', default = 0.5, type = float, help = 'compression factor for dense net') # no change
parser.add_argument('--k_0', default = 32, type = int, help = 'num of channel in input layer') # no change
parser.add_argument('--lbda', default = 0, type = float, help = 'lambda for l2 regularization')
parser.add_argument('--epochs', default = 50, type = int)
parser.add_argument('--batch_size', default= 48, type= int)
args = parser.parse_args()
#*****************************************************


# list for history-objects
lhist = []


print(' load patients')

# do the data loading
train_path, validation_path = dataset_split(args.data_dir, args.label_dir, valid_train_rate=0.05, shuffle=True, seed= 100)

#**************generator: I just copy the way done in UNET without augmentation, so very likely need some changes

image_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
mask_datagen = ImageDataGenerator()
gen_train = train_generator(image_datagen, mask_datagen, args.batch_size)
gen_valid = val_generator(image_datagen, mask_datagen, args.batch_size)


# load model with parameters
print(' load model')
network = DCCN_SYMM(in_shape=(args.in_size, args.in_size, args.in_size, 1),
                  kls = args.k,
                  ls = args.ls,
                  theta = args.theta,
                  k_0 = args.k_0,
                  lbda = args.lbda,
                  out_res= args.out_res,
                  feed_pos= args.pos,
                  pos_noise_stdv = args.pos_noise_stdv)

#compile model
network.compile()
network.model.summary()

# saves the model weights after each epoch if the validation loss decreased
path_w = args.model_path + "dccn-symm" + ".hdf5"
checkpointer = cb.ModelCheckpoint(filepath=path_w, verbose=0, monitor='val_loss', save_best_only=True)
#*****************add more callbacks if necessary, but its ok to just use ModelCheckpoint

# train
hist_object = network.train(gen_train, gen_valid, checkpointer, args)

print(' save histories')
# list of histories
lhist.append(hist_object.history)

# save history
path_hist = args.history_path + "dccn-symm"
history.save_histories(lhist=lhist, path=path_hist)